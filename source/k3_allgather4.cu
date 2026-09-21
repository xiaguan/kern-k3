// K3 prefill: the four small TP allgathers of a layer run as one push kernel.
//
// l<N>.gather_latent_q / gather_latent_sf / gather_topk_idx / gather_topk_weight
// move 15.7 MB of payload per rank in total (latent_q 14.0, latent_sf 0.44,
// topk_idx and topk_weight 0.25 each), but as four nccl_allgather_* calls they
// cost ~190 us per layer per rank: every call is ~25 us of fixed cost plus its
// own bandwidth, and the four are serialised.  They are the same kind of
// collective as `normed`, so they use the same peer-pointer scheme as
// kern_k3_allgather_push in source/k3_collectives.cu: each rank writes its own
// slice into every rank's copy of the gathered buffer (the runtime hands the
// kernel the device address of each rank's copy of an `export`ed buffer), and
// one barrier per call makes the call wait for every peer's stores.
//
// The four payloads are byte streams: every count is a multiple of 16 (rows *
// 3584, rows * 112, rows * 64, rows * 64), so whole 16 B vectors, and the
// vectors of the four buffers are treated as one concatenation.  The
// destination slice of rank r starts at byte rank * count, exactly like the
// bf16 allgather, so the layouts of `*_all` are unchanged.
//
// The barrier is the one described in source/k3_collectives.cu: this kernel
// bumps its own rank's coll_flags[slot] once per block, block 0 waits until
// every peer's counter has reached coll_flags[0] (the replay's epoch, bumped
// once per rank per program call by kern_k3_epoch_bump), and the epoch scheme
// only holds if this launch has the same grid as every other collective of the
// call -- so the manifest gives it the same grid expression over `tokens`.
//
//   nvcc -cubin -arch=sm_103a -o k3_allgather4.cubin k3_allgather4.cu

#include <cstdint>

#define K3_MAX_RANKS 8
#define K3_SPIN_NS   4000000000LL   /* ~2 s bail-out so a broken peer cannot wedge the GPU */

/* This rank's slice of one payload (nb bytes, a multiple of 16) to every rank's
 * copy: the destination slice of rank r starts at byte r * nb. */
__device__ __forceinline__ void push_span(
    const unsigned char* __restrict__ s, unsigned char* __restrict__ d,
    const unsigned long long* __restrict__ p, long nb, int rank, int nr,
    long tid, long stride)
{
    const long nv = nb >> 4;
    for (long i = tid; i < nv; i += stride) {
        const uint4 val = ((const uint4*)s)[i];
        const long eb = (long)rank * nb + (i << 4);
        *(uint4*)(d + eb) = val;                    /* this rank's own copy */
#pragma unroll
        for (int q = 0; q < K3_MAX_RANKS; ++q)
            if (q < nr && q != rank)
                *(uint4*)((unsigned char*)(uintptr_t)p[q] + eb) = val;
    }
}

extern "C" __global__ void __launch_bounds__(1024) kern_k3_allgather4_push(
    const unsigned char* __restrict__ s0, unsigned char* __restrict__ d0,
    const unsigned long long* __restrict__ p0, long n0,
    const unsigned char* __restrict__ s1, unsigned char* __restrict__ d1,
    const unsigned long long* __restrict__ p1, long n1,
    const unsigned char* __restrict__ s2, unsigned char* __restrict__ d2,
    const unsigned long long* __restrict__ p2, long n2,
    const unsigned char* __restrict__ s3, unsigned char* __restrict__ d3,
    const unsigned long long* __restrict__ p3, long n3,
    unsigned long long* __restrict__ flags,
    const unsigned long long* __restrict__ flags_peer,
    int slot, int rank, int nranks)
{
    const int nr = nranks < K3_MAX_RANKS ? nranks : K3_MAX_RANKS;
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;

    /* one span per payload, with the pointers scalar: an indexed array of
     * pointers would live in local memory (the whole kernel is four loads and
     * sixteen stores per thread, so a spilled address is a real cost). */
    push_span(s0, d0, p0, n0, rank, nr, tid, stride);
    push_span(s1, d1, p1, n1, rank, nr, tid, stride);
    push_span(s2, d2, p2, n2, rank, nr, tid, stride);
    push_span(s3, d3, p3, n3, rank, nr, tid, stride);

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
