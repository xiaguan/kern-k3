// K3 intra-node collectives over peer pointers (TP group).
//
// The runtime hands a kernel the device addresses of every rank's copy of an
// `export`ed buffer (a `peer` buffer, u64[tp]).  That is enough to run a
// collective without a communicator: an allgather is a push of the caller's own
// slice into the destination slice of every rank, including its own.
//
//   [bump] kern_k3_epoch_bump(flags)          first call of every replay
//   [AG]   kern_k3_allgather_push(src, dst, dst_peer, flags, flags_peer,
//                                 count, slot, rank, nranks)
//          src   [count]            this rank's slice (bf16)
//          dst   [nranks * count]   the caller's own copy of the gathered buffer;
//                                   dst_peer[r] is rank r's copy of it
//          flags [slots]            u64 arrival counters, flags[0] is the epoch
//          flags_peer[r][slot]      rank r's counters, read over NVLink
//   [init] kern_k3_flags_init(flags, n)       called once, from the `load` program.
//
// The reduce-scatter entries are described above them, at the end of the file.
//
// Grid: one thread per K3_AG_VPT 16 B vectors, block 1024, so a block publishes
// its arrival with a single atomicAdd and the waiting block polls nranks words.
//
// Correctness.  A rank must not run past the call while a peer's slice is still
// in flight.  Every block fences its stores and bumps its own rank's flags[slot]
// by one; block 0 waits until every peer's flags[slot] has reached `flags[0]`.
//
// flags[0] accumulates `gridDim.x` once per program call, so after the i-th call
// it is the total number of block arrivals those calls will produce.  That makes
// the target exact even when the calls have different grids (a 16384-row prefill
// is 896 blocks, a single-row decode step is one), and it makes stale state
// harmless: a counter left at an earlier call's total is strictly below every
// later target, because every call contributes at least one.  The counters are
// therefore never reset -- a reset is what a first version got wrong, since the
// fastest rank could read a peer's counter before the peer's reset became
// visible to it and pass the barrier without that peer's slice.
//
// Only whole 16 B vectors are moved and `count` is a multiple of 8 at every call
// site (rows * 7168), so every slice starts 16 B aligned.
//
//   nvcc -cubin -arch=sm_103a -o k3_collectives.cubin k3_collectives.cu

#include <cuda_bf16.h>
#include <cstdint>

typedef __nv_bfloat16  bf16_t;
typedef __nv_bfloat162 bf162_t;

#define K3_AG_VPT    4            /* 16 B vectors copied per thread */
#define K3_MAX_RANKS 8
#define K3_SPIN_NS   4000000000LL /* ~2 s bail-out so a broken peer cannot wedge the GPU */


__device__ __forceinline__ bf162_t as_bf162(unsigned u) {
    return __halves2bfloat162(__ushort_as_bfloat16((unsigned short)(u & 0xffffu)),
                              __ushort_as_bfloat16((unsigned short)(u >> 16)));
}
__device__ __forceinline__ float2 bf2f(unsigned u) { return __bfloat1622float2(as_bf162(u)); }
__device__ __forceinline__ unsigned f2bf(float2 f) {
    const bf162_t h = __float22bfloat162_rn(f);
    return (unsigned)(unsigned short)__bfloat16_as_ushort(__low2bfloat16(h)) |
           ((unsigned)(unsigned short)__bfloat16_as_ushort(__high2bfloat16(h)) << 16);
}

extern "C" __global__ void __launch_bounds__(1024) kern_k3_flags_init(
    unsigned long long* __restrict__ flags, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        flags[i] = 0ull;
}

/* flags[0] += gridDim.x, once per rank: launched with the same grid expression
 * as the collectives, so it counts exactly the arrivals this call will produce
 * (each collective's blocks bump their own counter by one each). */
extern "C" __global__ void __launch_bounds__(1024) kern_k3_epoch_bump(
    unsigned long long* __restrict__ flags)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)   /* once per rank, not once per block */
        atomicAdd(&flags[0], (unsigned long long)gridDim.x);
}

extern "C" __global__ void __launch_bounds__(1024) kern_k3_allgather_push(
    const bf16_t* __restrict__ src,
    bf16_t*       __restrict__ dst,
    const unsigned long long* __restrict__ dst_peer,
    unsigned long long* __restrict__ flags,
    const unsigned long long* __restrict__ flags_peer,
    long count, int slot, int rank, int nranks)
{
    const int nr = nranks < K3_MAX_RANKS ? nranks : K3_MAX_RANKS;
    const long nvec = count >> 3;
    const long elem_off = (long)rank * count;

    uint4* dp[K3_MAX_RANKS];
    for (int q = 0; q < nr; ++q)
        dp[q] = (uint4*)((bf16_t*)(uintptr_t)dst_peer[q] + elem_off);
    dp[rank] = (uint4*)(dst + elem_off);         /* the caller's own copy */

    const uint4* __restrict__ s = (const uint4*)src;
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = tid; i < nvec; i += stride) {
        const uint4 v = s[i];
#pragma unroll
        for (int q = 0; q < K3_MAX_RANKS; ++q)
            if (q < nr) dp[q][i] = v;
    }

    /* this block's stores are system-visible before its arrival is published */
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(&flags[slot], 1ull);

    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            const unsigned long long target = flags[0];
            const long long t0 = clock64();
            for (int q = 0; q < nr; ++q) {
                volatile unsigned long long* f =
                    (volatile unsigned long long*)(flags_peer[q] + (unsigned long long)slot * 8u);
                while (*f < target) {
                    if (clock64() - t0 > K3_SPIN_NS) break;
                    __nanosleep(64);
                }
            }
            __threadfence_system();
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------ reduce-scatter
//
//   [RS] kern_k3_reducescatter_bf16(src, dst, stage, stage_peer, flags,
//                                  flags_peer, count, slot, rank, nranks)
//        src   [nranks * count]  this rank's partial (o_proj / moe_partial)
//        dst   [count]           the reduced chunk this rank keeps
//        stage [nranks * count]  this rank's staging area; stage_peer[r] is
//                                rank r's copy of it
//   launch 1 `kern_k3_rs_push` moves chunk c of src into peer c's slot `rank`;
//   launch 2 `kern_k3_rs_sum`  adds this rank's own chunk `rank` to the three
//   staged ones and lands bf16, in the ring's order and with the ring's
//   per-hop bf16 rounding, so the reduced values match ncclReduceScatter
//   bit for bit (see kern_k3_rs_sum).  The two launches are one op, so the call site
//   is unchanged: the barrier in launch 1 keeps launch 2 from reading a slot a
//   peer has not written yet.
//
//   Each rank moves (nranks-1)/nranks of `src` over the wire and the same
//   amount arrives; nothing is sent twice, unlike a ring, and there is one
//   barrier per call instead of nranks-1.

extern "C" __global__ void __launch_bounds__(1024) kern_k3_rs_push(
    const bf16_t* __restrict__ src,
    bf16_t*       __restrict__ stage,
    const unsigned long long* __restrict__ stage_peer,
    unsigned long long* __restrict__ flags,
    const unsigned long long* __restrict__ flags_peer,
    long count, int slot, int rank, int nranks)
{
    const int nr = nranks < K3_MAX_RANKS ? nranks : K3_MAX_RANKS;
    const long nvec = count >> 3;
    const long my = (long)rank * nvec;      /* this rank's slot in a peer's stage */

    uint4* dq[K3_MAX_RANKS];
    for (int c = 0; c < nr; ++c)
        if (c != rank)
            dq[c] = (uint4*)((bf16_t*)(uintptr_t)stage_peer[c] + (long)rank * count);
    (void)stage;                            /* this rank's own staging stays untouched */

    const uint4* __restrict__ s = (const uint4*)src;
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = tid; i < nvec; i += stride) {
#pragma unroll
        for (int c = 0; c < K3_MAX_RANKS; ++c)
            if (c < nr && c != rank)
                dq[c][i] = s[(long)c * nvec + i];
    }
    (void)my;

    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(&flags[slot], 1ull);

    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            const unsigned long long target = flags[0];
            const long long t0 = clock64();
            for (int q = 0; q < nr; ++q) {
                volatile unsigned long long* f =
                    (volatile unsigned long long*)(flags_peer[q] + (unsigned long long)slot * 8u);
                while (*f < target) {
                    if (clock64() - t0 > K3_SPIN_NS) break;
                    __nanosleep(64);
                }
            }
            __threadfence_system();
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(1024) kern_k3_rs_sum(
    const bf16_t* __restrict__ src,
    bf16_t*       __restrict__ dst,
    const bf16_t* __restrict__ stage,
    long count, int rank, int nranks, int order)
{
    const int nr = nranks < K3_MAX_RANKS ? nranks : K3_MAX_RANKS;
    const long nvec = count >> 3;

    const uint4* __restrict__ s = (const uint4*)src;
    const uint4* __restrict__ g = (const uint4*)stage;
    uint4* __restrict__ o = (uint4*)dst;

    /* NCCL reduce-scatter is a ring: the running sum for a chunk lives in the
     * user's bf16 buffer between hops, so every hop but the first lands one
     * bf16 rounding.  Which rank's partial is added when is a property of the
     * ring, so the order is a launch parameter (ownpos = where this rank's own
     * partial enters the chain, descend = peers in descending ring order). */
    const int ownpos = order >> 1;
    const int descend = order & 1;
    int seq[K3_MAX_RANKS];
    {
        int pi = 0;
        for (int j = 0; j < nr; ++j) {
            if (j == ownpos) {
                seq[j] = rank;
            } else {
                const int off = descend ? (nr - 1 - pi) : (1 + pi);
                seq[j] = (rank + off) % nr;
                ++pi;
            }
        }
    }

    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = tid; i < nvec; i += stride) {
        float2 a[4];
        for (int k = 0; k < nr; ++k) {
            const uint4 v = (seq[k] == rank) ? s[(long)rank * nvec + i]
                                             : g[(long)seq[k] * nvec + i];
            const unsigned vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(vv[j]);
                if (k == 0) { a[j] = f; }
                else { a[j] = bf2f(f2bf(make_float2(a[j].x + f.x, a[j].y + f.y))); }
            }
        }
        uint4 out;
        out.x = f2bf(a[0]); out.y = f2bf(a[1]); out.z = f2bf(a[2]); out.w = f2bf(a[3]);
        o[i] = out;
    }
}
