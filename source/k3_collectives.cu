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

typedef __nv_bfloat16 bf16_t;

#define K3_AG_VPT    4            /* 16 B vectors copied per thread */
#define K3_MAX_RANKS 8
#define K3_SPIN_NS   4000000000LL /* ~2 s bail-out so a broken peer cannot wedge the GPU */

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
