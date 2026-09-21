// The arrival half of the TP reduce-scatter, as one block.
//
// `kern_k3_rs_arrive` (source/k3_reducescatter_pull.cu) does no work at all: it
// fences, bumps this rank's `coll_flags[slot]` once per block, and block 0 waits
// for every peer's counter to reach `coll_flags[0]`.  It is launched with the
// collective grid -- 896 blocks of 1024 threads -- because the epoch in
// `coll_flags[0]` is bumped by `gridDim.x` once per rank per program call and
// every collective of the call has to contribute exactly that many arrivals.
//
// Nothing about that requires 896 blocks.  A launch of one block that adds the
// whole grid's worth of arrivals in a single atomic leaves the counters exactly
// where they would have been (the epoch target still rises by `gridDim.x` per
// call, and this rank's slot still reaches it only after this kernel runs, which
// is stream-ordered after the producer), and it drops
//
//   * the ramp of 917504 threads (three waves of 1024-thread blocks),
//   * 895 of the 896 atomicAdds to the same cache line,
//   * and 896 copies of the two `__threadfence_system()` -- each of which is a
//     full L1 invalidate (`CCTL.IVALL`) on the SM that runs it.
//
// Measured: removing the arrival launch from the reduce-scatter op entirely
// takes the graph from 620.5 to 613.0 ms (187 calls, ~40 us per call, of which
// ~15 us is the rank skew the wait is made of and the rest is that mechanism).
//
// The barrier's observable meaning is unchanged: the bump still happens after
// this rank's producer has completed (kernel boundary), and block 0 still waits
// for every peer.  The two fences are kept: with one block they cost two
// invalidations on one SM instead of twelve per SM, so there is nothing to win
// by arguing about them.
//
//   nvcc -cubin -arch=sm_103a -o k3_rs_arrive1.cubin k3_rs_arrive1.cu

#include <cuda_bf16.h>
#include <cstdint>

typedef __nv_bfloat16 bf16_t;

#define K3_MAX_RANKS 8
#define K3_SPIN_NS   4000000000LL /* ~2 s bail-out so a broken peer cannot wedge the GPU */

extern "C" __global__ void __launch_bounds__(32) kern_k3_rs_arrive1(
    const bf16_t* __restrict__ src,
    bf16_t*       __restrict__ dst,
    const unsigned long long* __restrict__ src_peer,
    unsigned long long* __restrict__ flags,
    const unsigned long long* __restrict__ flags_peer,
    long count, int units, int slot, int order, int rank, int nranks)
{
    (void)src; (void)dst; (void)src_peer; (void)count; (void)order; (void)rank;
    const int nr = nranks < K3_MAX_RANKS ? nranks : K3_MAX_RANKS;

    __threadfence_system();
    if (threadIdx.x == 0) {
        atomicAdd(&flags[slot], (unsigned long long)units);

        /* the same wait as kern_k3_rs_arrive, with the address arithmetic and
         * the comparison width hoisted out of the spin: every counter is far
         * below 2^32, so the low half of the u64 is the whole value */
        const unsigned long long target = flags[0];
        const unsigned* fp[K3_MAX_RANKS];
        for (int q = 0; q < nr; ++q)
            fp[q] = (const unsigned*)(flags_peer[q] + (unsigned long long)slot * 8u);
        const long long t0 = clock64();
        for (int q = 0; q < nr; ++q) {
            volatile const unsigned* f = fp[q];
            while (*f < (unsigned)target) {
                if (clock64() - t0 > K3_SPIN_NS) break;
                __nanosleep(64);
            }
        }
        __threadfence_system();
    }
}
