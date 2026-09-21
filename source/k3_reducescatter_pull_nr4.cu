// K3 intra-node reduce-scatter over peer pointers, pull form (TP group).
//
// The push form (source/k3_collectives.cu) splits one collective into two
// launches: `kern_k3_rs_push` moves this rank's three quarters of the partial
// into the peers' staging slots, block 0 waits for every peer's arrival, and
// only then does the second launch `kern_k3_rs_sum` read the staging back and
// land the ring's bf16 chain.  Link traffic and local traffic are therefore
// serialised: the sum's ~150 MB of local reads and writes wait for the last
// remote store of the slowest rank.
//
// A pull does the same link volume (rank r reads the three peers' slices of the
// chunk it keeps, and serves its own three slices to them) but reads the peers'
// `src` directly, so the local read/sum/store is in the same kernel as the link
// traffic and overlaps it.  The local staging disappears entirely.
//
//   [arrive] kern_k3_rs_arrive(src, dst, src_peer, flags, flags_peer,
//                              count, slot, order, rank, nranks)
//            bumps this rank's arrival, block 0 waits for the peers -- a second
//            launch, exactly like the push form, so the pull never reads a
//            peer's partial before that peer's producer has run.
//   [pull]   kern_k3_rs_pull(...) reads the four partials of the chunk this rank
//            keeps and lands the ring's bf16 chain.
//
// The arithmetic is the one scripts/gen_collectives_rs.py established: NCCL's
// reduce-scatter is a ring, so the running sum lives in the user's bf16 buffer
// between hops and every hop but the first lands one bf16 rounding.  `order`
// selects which rank's partial enters when (`ownpos = order >> 1`) and whether
// the peers are walked in ascending or descending ring order
// (`descend = order & 1`); "peers ascending, own partial last" (order 6)
// reproduces ncclReduceScatter bit for bit.
//
// Grid: one thread per 16 B vector, block 1024, the same grid expression as
// every other collective of a call (see source/k3_collectives.cu for why the
// arrival counters want that).
//
//   nvcc -cubin -arch=sm_103a -o k3_reducescatter_pull.cubin k3_reducescatter_pull.cu

#include <cuda_bf16.h>
#include <cstdint>

typedef __nv_bfloat16  bf16_t;
typedef __nv_bfloat162 bf162_t;

#define K3_MAX_RANKS 8
#define K3_SPIN_NS   4000000000LL /* ~2 s bail-out so a broken peer cannot wedge the GPU */

/* The shape is fixed (4 ranks, `order` 6 at every call site), so it is compiled
 * in rather than read back out of the launch arguments.  `nr` used to be a
 * runtime value, and `q = (rank + o) % nr` came out of nvcc as ten groups of
 * MUFU.RCP / I2F / F2I / IMAD.HI: 944 instructions in the kernel, 402 of them
 * prologue, and a conditional branch of ~250 instructions in front of every hop
 * that kept the four peer loads from being issued back to back.  With NR and
 * ORDER constant the chain is four unconditional straight lines, the rank of
 * each hop is `(rank + o) & (NR - 1)`, and the unpack of a bf16 pair is a shift
 * and a mask instead of PRMT + IMAD.  The rounding chain is untouched: peers
 * ascending, own partial last, one bf16 rounding per hop, which is what
 * ncclReduceScatter's ring does. */
#ifndef NR
#define NR 4
#endif
#ifndef ORDER
#define ORDER 6
#endif
static_assert(NR > 0 && (NR & (NR - 1)) == 0, "NR must be a power of two");
static_assert(ORDER >= 0 && ORDER < 2 * NR, "ORDER packs ownpos and descend");

__device__ __forceinline__ bf162_t as_bf162(unsigned u) {
    return __halves2bfloat162(__ushort_as_bfloat16((unsigned short)(u & 0xffffu)),
                              __ushort_as_bfloat16((unsigned short)(u >> 16)));
}
__device__ __forceinline__ float2 bf2f(unsigned u) { return __bfloat1622float2(as_bf162(u)); }
/* bf16 -> f32 is the top half of the word: a shift and a mask, no PRMT. */
__device__ __forceinline__ float2 bf2f_shift(unsigned u) {
    float2 f;
    f.x = __uint_as_float(u << 16);
    f.y = __uint_as_float(u & 0xffff0000u);
    return f;
}
__device__ __forceinline__ unsigned f2bf(float2 f) {
    const bf162_t h = __float22bfloat162_rn(f);
    return (unsigned)(unsigned short)__bfloat16_as_ushort(__low2bfloat16(h)) |
           ((unsigned)(unsigned short)__bfloat16_as_ushort(__high2bfloat16(h)) << 16);
}

/* One block per 16 B vector stride; bump this rank's arrival, block 0 waits
 * until every peer's counter has reached flags[0] (the replay's epoch).  No
 * store of this kernel's own is ordered -- the producer that wrote `src` is
 * separated from it by a kernel boundary, so a device-scope flush is already
 * in place; the fence is kept for the store-side visibility of that flush. */
extern "C" __global__ void __launch_bounds__(1024) kern_k3_rs_arrive(
    const bf16_t* __restrict__ src,
    bf16_t*       __restrict__ dst,
    const unsigned long long* __restrict__ src_peer,
    unsigned long long* __restrict__ flags,
    const unsigned long long* __restrict__ flags_peer,
    long count, int slot, int order, int rank, int nranks)
{
    (void)src; (void)dst; (void)src_peer; (void)count; (void)order; (void)rank;
    const int nr = nranks < K3_MAX_RANKS ? nranks : K3_MAX_RANKS;

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

extern "C" __global__ void __launch_bounds__(1024) kern_k3_rs_pull(
    const bf16_t* __restrict__ src,
    bf16_t*       __restrict__ dst,
    const unsigned long long* __restrict__ src_peer,
    unsigned long long* __restrict__ flags,
    const unsigned long long* __restrict__ flags_peer,
    long count, int slot, int order, int rank, int nranks)
{
    (void)flags; (void)flags_peer; (void)slot; (void)order; (void)nranks;
    /* this cubin is built for one shape; a call of another shape has to be
     * given the cubin for it (kernels.toml carries one module per shape) */
    if (nranks != NR || order != ORDER) return;
    const long nvec = count >> 3;
    const long off = (long)rank * count;

    const int ownpos = ORDER >> 1;
    const int descend = ORDER & 1;

    /* the four base pointers, one per ring offset; position k of the chain then
     * picks a constant index out of them and the hot loop has no dynamic
     * indexing and no branch */
    const uint4* own = (const uint4*)(src + off);
    const uint4* peer[NR];
#pragma unroll
    for (int j = 0; j < NR; ++j)
        peer[j] = (const uint4*)((const bf16_t*)(uintptr_t)src_peer[(rank + j) & (NR - 1)] + off);
    const uint4* base[NR];
#pragma unroll
    for (int k = 0; k < NR; ++k) {
        if (k == ownpos) {
            base[k] = own;
        } else {
            const int pi = (k < ownpos) ? k : k - 1;
            const int o = descend ? (NR - 1 - pi) : (1 + pi);
            base[k] = peer[o & (NR - 1)];
        }
    }
    uint4* __restrict__ o = (uint4*)dst;

    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = tid; i < nvec; i += stride) {
        float2 a[4];
        {
            const uint4 v = base[0][i];                 /* first hop: no rounding */
            a[0] = bf2f_shift(v.x); a[1] = bf2f_shift(v.y);
            a[2] = bf2f_shift(v.z); a[3] = bf2f_shift(v.w);
        }
#pragma unroll
        for (int k = 1; k < NR; ++k) {
            const uint4 v = base[k][i];
            const unsigned vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f_shift(vv[j]);
                a[j] = bf2f(f2bf(make_float2(a[j].x + f.x, a[j].y + f.y)));
            }
        }
        uint4 out;
        out.x = f2bf(a[0]); out.y = f2bf(a[1]); out.z = f2bf(a[2]); out.w = f2bf(a[3]);
        o[i] = out;
    }
}
