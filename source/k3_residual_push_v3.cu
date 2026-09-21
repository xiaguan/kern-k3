// K3 prefill: the residual kernel that computes `normed` writes it straight to
// every rank of the TP group, so the allgather that used to follow it has no
// payload left.
//
// `l*.res_in` (kern_k3_attnres_rms in source/k3_residual_v2.cu) produces this
// rank's 4096 rows of 7168 bf16 `normed`; `l*.gather_normed`
// (kern_k3_allgather_push in source/k3_collectives.cu) then reads those 58.7 MB
// back and stores them into all four ranks' `normed_all`, which is 176 MB of
// NVLink egress per layer.  The two are consecutive calls of the same layer and
// cost 97 us + 283 us per call per rank, and neither half is anywhere near the
// bandwidth the other half is bound by: the residual kernel is HBM-bound at
// ~2.4 TB/s, the push is link-bound at ~620 GB/s of egress.
//
// This kernel is that pair in one launch: the row's normed vectors are already
// in registers when the rms pass finishes, so they are stored straight into the
// four ranks' copies instead of into one local buffer that a second kernel then
// re-reads.  The link traffic overlaps the HBM traffic instead of following it,
// and the 58.7 MB read-back disappears.
//
// The barrier stays a separate launch: an allgather of a rank's slice needs
// every peer's slice to have landed before the consumer runs, and a block that
// spins inside this kernel would never retire, so the blocks that still have to
// arrive could never be scheduled (see the note in source/k3_collectives.cu).
// The generator therefore keeps the `k3_allgather_push` call of the layer, with
// a zero payload, as the barrier.  It is stream-ordered after this kernel, so
// its arrival still means "this rank's slice has been written".
//
// The arithmetic is byte-for-byte the one of kern_k3_attnres_rms: the same
// candidate loop, the same fixed-order reductions, the same single bf16 landing
// per value.  Only the destination of the last store changed.
//
// Launch geometry: grid (B, 1, 1) = one row per block, block (RES_THREADS, 1, 1)
// = 128, smem 0, exactly as the kernel it is derived from.
//
// The second `__launch_bounds__` argument is what makes the fusion pay: without
// it the four extra destination pointers push the kernel from 122 to 146
// registers, which drops it from four blocks per SM to three, and 384 threads
// per SM cannot keep this kernel's 235 MB of stores in flight -- measured, the
// fusion then costs 352 us where the two kernels it replaces cost 378, and the
// graph does not move (625.2 vs 625.0 ms).  Asking for four blocks per SM caps
// it at 128 registers, at the cost of 24 bytes of stack, and the same graph
// drops to 621.7 ms.  The stores, not the arithmetic, are what the SM has to
// keep busy.
//
//   nvcc -cubin -arch=sm_103a -o k3_residual_push_v3.cubin k3_residual_push_v3.cu

// K3 decode, family K1: the residual stream (attnres + rms + landing adds), v2.
//
// Same ABI, math and landing points as k3_residual.cu (docs/k3-kernel-abi.md
// section K1); only the thread layout and the reduction order differ:
//
//   [K1a] kern_k3_attnres_rms(prefix, blocks, sw, gamma, normed, nb, snapshot, B)
//   [K1b] kern_k3_land_add_attnres_rms(partial, prefix, blocks, sw, gamma,
//                                      prefix2, normed, nb, snapshot, B)
//   [K1c] kern_k3_land_add2(p1, p2, prefix2, hidden, two, B)
//
// Launch geometry (all three entries):
//     grid  = (B, 1, 1)                 one row per block
//     block = (RES_THREADS, 1, 1)       default 128, must divide 896
//     smem  = 0                         static shared memory only
//
// Why v2.  v1 used one 1024-thread block per row with a single 16 B vector per
// thread and __launch_bounds__(1024, 1): one block per SM, three block-wide
// barriers and several dependent global round trips per row, so at B = 4096
// (16k-token TP4 prefill) it ran at ~100 us + 14 us per candidate row, three to
// four times off the bandwidth bound.  v2 gives each thread KVPT = 896 /
// RES_THREADS vectors of the row (vector t + v * RES_THREADS), so a row is a
// 128-thread block and four rows are resident per SM; both passes walk the
// thread's vectors in the outer loop and the nb candidates in the inner loop,
// issuing the nb 16 B loads of a vector back to back through a volatile asm
// load that the compiler cannot hoist across iterations (hoisting would keep
// every candidate vector live at once, cost ~200 registers and halve
// residency).  Only one accumulator vector and one sw vector are live at a
// time.  Measured at B = 4096 on GB300 (standalone, 20 launches): attnres_rms
// 113 -> 63 us at nb = 1 and 214 -> 128 us at nb = 8; land_add_attnres_rms
// 133 -> 94 us and 237 -> 155 us.
//
// Reductions are fixed-order (warp butterfly, then a fixed-order sum of the
// four warp partials), so a result never depends on the schedule, but the
// order differs from v1: values may differ from v1 by f32 rounding, i.e.
// occasionally one bf16 ulp of normed.  prefix2 / hidden / blocks are
// bit-identical to v1.
//
// Landing points (unchanged from v1):
//   K1a/K1b: mixed = bf16(sum_c p_c * f32(cand_c));  rms: bf16(f32(mixed) * rsqrt)
//            then * gamma as bf16 x bf16 -> bf16.  Scores/softmax stay f32.
//   K1b:     p = bf16(partial) first, then prefix2 = bf16(f32(prefix) + f32(p)).
//   K1c:     bf16(p1), bf16(p2) first, sum with f32(prefix2) in f32, one final round.
//
//   nvcc -cubin -arch=sm_103a [-DLAND_BF16=1] -o k3_residual_v2.cubin k3_residual_v2.cu

#include <cuda_bf16.h>
#include <cstdint>

typedef __nv_bfloat16  bf16_t;
typedef __nv_bfloat162 bf162_t;

#define KH        7168
#define KNB_MAX   8
#define KVEC      (KH / 8)        /* 896 sixteen-byte vectors per row */
#define KEPS      1e-5f
#define K3_MAX_RANKS 8
#define K3_SPIN_NS 4000000000LL /* ~2 s bail-out so a broken peer cannot wedge the GPU */

#ifndef RES_THREADS
#define RES_THREADS 128
#endif
#define KVPT      (KVEC / RES_THREADS)   /* vectors per thread */
#define KNWARPS   (RES_THREADS / 32)
#if (KVEC % RES_THREADS) != 0 || (RES_THREADS % 32) != 0
#error "RES_THREADS must be a multiple of 32 that divides 896"
#endif

struct V8 { unsigned w[4]; };

__device__ __forceinline__ V8 ldv(const void* p) {
    const uint4 t = *(const uint4*)p;
    V8 v; v.w[0] = t.x; v.w[1] = t.y; v.w[2] = t.z; v.w[3] = t.w;
    return v;
}
__device__ __forceinline__ void stv(void* p, const V8& v) {
    *(uint4*)p = make_uint4(v.w[0], v.w[1], v.w[2], v.w[3]);
}
__device__ __forceinline__ bf162_t as_bf162(unsigned u) {
    return __halves2bfloat162(__ushort_as_bfloat16((unsigned short)(u & 0xffffu)),
                              __ushort_as_bfloat16((unsigned short)(u >> 16)));
}
__device__ __forceinline__ unsigned from_bf162(bf162_t h) {
    return (unsigned)(unsigned short)__bfloat16_as_ushort(__low2bfloat16(h)) |
           ((unsigned)(unsigned short)__bfloat16_as_ushort(__high2bfloat16(h)) << 16);
}
__device__ __forceinline__ float2 bf2f(unsigned u)  { return __bfloat1622float2(as_bf162(u)); }
__device__ __forceinline__ unsigned f2bf(float2 f)  { return from_bf162(__float22bfloat162_rn(f)); }

// 16 B read-only global load that the compiler will not hoist across iterations.
__device__ __forceinline__ V8 ldv_nv(const void* p) {
    V8 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.w[0]), "=r"(v.w[1]), "=r"(v.w[2]), "=r"(v.w[3]) : "l"(p));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// mixed = attnres(candidates 0..nb-1 from blk_row, candidate nb = pv held in registers)
// then normed_row = rms(mixed, gamma).  Called with the whole block; t = threadIdx.x.
// pv is overwritten with the mixed row (both are this thread's KVPT vectors).
//
// Both passes walk this thread's vectors in the outer loop and the candidates
// in the inner loop, so only one vector of sw / one accumulator vector is live
// at a time and each candidate's nb loads of a vector are issued back to back.
__device__ __forceinline__ void attnres_rms_row(
    const bf16_t* __restrict__ blk_row,   // &blocks[b * NB_MAX * H]  (unread when nb == 0)
    V8*                        pv,        // this thread's KVPT vectors of the prefix candidate
    const float*  __restrict__ sw,        // [H]   (unread when nb == 0)
    const bf16_t* __restrict__ gamma,     // [H]
    bf16_t*       __restrict__ dst_row,   // &normed_all[(rank * B + b) * H] of this rank
    const unsigned long long* __restrict__ dst_peer,  // every rank's normed_all
    long dst_off,                         // (rank * B + b) * H in every rank's normed_all
    int rank, int nranks,
    int nb, int t)
{
    const int  lane = t & 31, warp = t >> 5;

    __shared__ float s_red[(KNB_MAX + 1) * 2 * KNWARPS];   // [value][warp] partials
    __shared__ float s_red2[KNWARPS];

    const int ncand = nb + 1;

    if (nb > 0) {
        // pass 1: (sum x^2, sum x*sw) per candidate.
        float sq[KNB_MAX + 1], dp[KNB_MAX + 1];
#pragma unroll
        for (int c = 0; c <= KNB_MAX; ++c) { sq[c] = 0.f; dp[c] = 0.f; }
#pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const int u = (t + v * RES_THREADS) * 8;
            const float4* sp4 = (const float4*)(sw + u);
            const float4 a = sp4[0], b = sp4[1];
            const float swv[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
            V8 x[KNB_MAX];
#pragma unroll
            for (int c = 0; c < KNB_MAX; ++c)
                if (c < nb) x[c] = ldv_nv(blk_row + (size_t)c * KH + u);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(pv[v].w[j]);
                sq[KNB_MAX] += f.x * f.x;  sq[KNB_MAX] += f.y * f.y;
                dp[KNB_MAX] += f.x * swv[2 * j];  dp[KNB_MAX] += f.y * swv[2 * j + 1];
            }
#pragma unroll
            for (int c = 0; c < KNB_MAX; ++c) {
                if (c >= nb) break;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = bf2f(x[c].w[j]);
                    sq[c] += f.x * f.x;  sq[c] += f.y * f.y;
                    dp[c] += f.x * swv[2 * j];  dp[c] += f.y * swv[2 * j + 1];
                }
            }
        }
#pragma unroll
        for (int c = 0; c < KNB_MAX; ++c) {
            if (c >= nb) break;
            const float s = warp_sum(sq[c]);
            const float d = warp_sum(dp[c]);
            if (lane == 0) { s_red[(c * 2) * KNWARPS + warp] = s; s_red[(c * 2 + 1) * KNWARPS + warp] = d; }
        }
        {
            const float s = warp_sum(sq[KNB_MAX]);
            const float d = warp_sum(dp[KNB_MAX]);
            if (lane == 0) { s_red[(nb * 2) * KNWARPS + warp] = s; s_red[(nb * 2 + 1) * KNWARPS + warp] = d; }
        }
        __syncthreads();

        // scores + softmax, recomputed identically by every warp from the warp
        // partials (fixed order): lane c ends up holding p_c.
        float sc = -3.0e38f;
        if (lane < ncand) {
            float sqv = 0.f, dpv = 0.f;
#pragma unroll
            for (int w = 0; w < KNWARPS; ++w) {
                sqv += s_red[(2 * lane) * KNWARPS + w];
                dpv += s_red[(2 * lane + 1) * KNWARPS + w];
            }
            sc = dpv * rsqrtf(sqv * (1.0f / (float)KH) + KEPS);
        }
        float mx = sc;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float ex = (lane < ncand) ? __expf(sc - mx) : 0.f;
        float den = ex;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
        const float pmine = ex / den;                 // lane c: p_c
        float p[KNB_MAX + 1];
#pragma unroll
        for (int c = 0; c <= KNB_MAX; ++c) p[c] = __shfl_sync(0xffffffffu, pmine, c);
        const float pp = __shfl_sync(0xffffffffu, pmine, nb);   // the prefix candidate

        // pass 2: mix (f32 accumulate, one bf16 landing).  Candidates are re-read
        // (they are in L2 from pass 1); the prefix is still in registers.
#pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const int u = (t + v * RES_THREADS) * 8;
            V8 x[KNB_MAX];
#pragma unroll
            for (int c = 0; c < KNB_MAX; ++c)
                if (c < nb) x[c] = ldv_nv(blk_row + (size_t)c * KH + u);
            float acc[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(pv[v].w[j]);
                acc[2 * j]     = pp * f.x;
                acc[2 * j + 1] = pp * f.y;
            }
#pragma unroll
            for (int c = 0; c < KNB_MAX; ++c) {
                if (c >= nb) break;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = bf2f(x[c].w[j]);
                    acc[2 * j]     += p[c] * f.x;
                    acc[2 * j + 1] += p[c] * f.y;
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
                pv[v].w[j] = f2bf(make_float2(acc[2 * j], acc[2 * j + 1]));
        }
    }
    // nb == 0: mixed = prefix, since bf16(1.0f * f32(prefix)) == prefix.

    // rms(mixed, gamma)
    float sq = 0.f;
#pragma unroll
    for (int v = 0; v < KVPT; ++v) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = bf2f(pv[v].w[j]);
            sq += f.x * f.x;  sq += f.y * f.y;
        }
    }
    sq = warp_sum(sq);
    if (lane == 0) s_red2[warp] = sq;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < KNWARPS; ++w) tot += s_red2[w];
    const float r = rsqrtf(tot * (1.0f / (float)KH) + KEPS);

#pragma unroll
    for (int v = 0; v < KVPT; ++v) {
        const int u = (t + v * RES_THREADS) * 8;
        const V8 g = ldv(gamma + u);
        V8 o;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = bf2f(pv[v].w[j]);
            f.x *= r;  f.y *= r;
            o.w[j] = from_bf162(__hmul2(__float22bfloat162_rn(f), as_bf162(g.w[j])));
        }
        stv(dst_row + u, o);
#pragma unroll
        for (int q = 0; q < K3_MAX_RANKS; ++q)
            if (q < nranks && q != rank)
                stv((bf16_t*)(uintptr_t)dst_peer[q] + dst_off + u, o);
    }
}

// ---------------------------------------------------------------- push variant
// Same ABI as kern_k3_attnres_rms, with `normed` replaced by the four ranks'
// `normed_all`: this rank's own slice is written through `normed_all`, the other
// three through `dst_peer`, exactly like kern_k3_allgather_push does.  `rank`
// and `nranks` arrive as launch arguments, as they do there.
extern "C" __global__ void __launch_bounds__(RES_THREADS, 4) kern_k3_attnres_rms_push(
    const bf16_t* __restrict__ prefix,     // [B, H]
    bf16_t*       __restrict__ blocks,     // [B, NB_MAX, H]
    const float*  __restrict__ sw,         // [H]
    const bf16_t* __restrict__ gamma,      // [H]
    bf16_t*       __restrict__ normed_all, // [nranks * B, H]
    const unsigned long long* __restrict__ dst_peer,  // every rank's normed_all
    int nb, int snapshot, int B, int rank, int nranks)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int t = threadIdx.x;
    const int nr = nranks < K3_MAX_RANKS ? nranks : K3_MAX_RANKS;

    /* the destination slice of this rank's chunk is the same row everywhere */
    const long dst_off = ((long)rank * B + b) * (long)KH;
    bf16_t* dst_row = normed_all + dst_off;

    V8 pv[KVPT];
#pragma unroll
    for (int v = 0; v < KVPT; ++v) pv[v] = ldv(prefix + (size_t)b * KH + (t + v * RES_THREADS) * 8);

    if (snapshot && nb < KNB_MAX) {
#pragma unroll
        for (int v = 0; v < KVPT; ++v)
            stv(blocks + ((size_t)b * KNB_MAX + nb) * KH + (t + v * RES_THREADS) * 8, pv[v]);
    }

    attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, pv, sw, gamma,
                    dst_row, dst_peer, dst_off, rank, nr, nb, t);

    /* No fence here.  The peers read these stores over NVLink once the barrier
     * of the next call passes, and that barrier is a launch of its own
     * (l*.normed_sync -> kern_k3_rs_arrive1, one block), which executes
     * __threadfence_system() before it bumps this rank's arrival.  A kernel
     * boundary already flushes these stores to the device's coherence point --
     * the L2 that a peer's read is serviced from -- so that fence, which runs
     * after them, is the release that orders them; the per-thread fence this
     * kernel used to execute (524288 of them per call, each a CCTL.IVALL) was
     * ordering stores that are ordered already, and cost 40 us per call of the
     * 315.  Measured: l*.res_in 314.9 -> 274.9 us per call, which is the store
     * floor this kernel reaches when it has no candidates at all. */
}

