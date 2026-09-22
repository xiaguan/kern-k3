// res_mlp reads the candidates' statistics from res_in (round 20)
// K3 prefill: the residual + RMSNorm kernel of a layer, one cubin entry per
// candidate count.
//
// `kern_k3_land_add_attnres_rms` (source/k3_residual_v2.cu) reads `nb` candidates
// out of `blocks` at run time, so nvcc has to size `V8 x[KNB_MAX]`,
// `sq/dp[KNB_MAX+1]` and `p[KNB_MAX+1]` for nb = 8 whatever the call asks for,
// and every candidate loop keeps its `if (c < nb)` guard.  `nb` is not a run-time
// property: twelve layers of the model use each value 1..7 and nine use 8, and
// the call site knows which.
//
// So this file is that kernel written out once per candidate count, each with
// `const int nb = k` in place of the parameter; the inliner propagates the
// constant into `attnres_rms_row` and nvcc folds it.  Measured with cuobjdump,
// per entry:
//
//   nb   registers   instructions      blocks per SM at block 128
//    1        88          1848            5      (was 4 at 124 registers)
//    4       102          3000            5
//    8       108          4456            4
//
// With the candidates loaded CH = 2 at a time (round 15) the nb = 3, 5 and 7
// entries also reach five blocks per SM, at identical results:
//
//   nb   registers          blocks per SM
//    1    88 ->  88            5 -> 5
//    2    86 ->  86            5 -> 5
//    3   104 -> 102            4 -> 5
//    4   102 -> 102            5 -> 5
//    5   102 -> 100            5 -> 5
//    6   105 -> 106            4 -> 4
//    7   112 -> 102            4 -> 5
//    8   108 -> 104            4 -> 4
//
// The arithmetic is untouched: same candidate loop, same fixed-order reductions,
// same single bf16 landing per value.  A call whose shape is not the entry's
// returns early.
//
// The thread's row slice `pv[KVPT]` (prefix + partial, then the mixed row) is 28
// registers at KVPT = 7 and it is live through pass 1, pass 2 and the RMS phase.
// It lives in shared memory now -- one 16 B slot per (vector, thread), read and
// written through volatile `ld.shared`/`st.shared` so the compiler cannot cache
// it back into registers.  Registers, per entry, before -> after: nb 1 88 -> 40,
// 2 86 -> 48, 3 102 -> 48, 4 102 -> 54, 5 100 -> 60, 6 106 -> 66, 7 102 -> 70,
// 8 104 -> 68, no spill, 15.7 KB of shared memory per block.  Same values, same
// order, so still bit-identical (`test16k`: every span, the logits and the KV/KDA
// state).
//
//   nvcc -cubin -arch=sm_103a -DLAND_BF16=1 -o k3_residual_nb+LAND_BF16=1.cubin k3_residual_nb.cu

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

typedef __nv_bfloat16  bf16_t;
typedef __nv_bfloat162 bf162_t;

#define KH        7168
#define KNB_MAX   8
#define KVEC      (KH / 8)        /* 896 sixteen-byte vectors per row */
#define KEPS      1e-5f

#ifndef RES_THREADS
#define RES_THREADS 128
#endif
#ifndef CH
#define CH 2
#endif
#define KVPT      (KVEC / RES_THREADS)   /* vectors per thread */
#define KNWARPS   (RES_THREADS / 32)
#if (KVEC % RES_THREADS) != 0 || (RES_THREADS % 32) != 0
#error "RES_THREADS must be a multiple of 32 that divides 896"
#endif

struct V8 { unsigned w[4]; };

// 16 B shared-memory load/store that the compiler will not hoist or cache in
// registers across the kernel (same trick as ldv_nv for the global candidates).
__device__ __forceinline__ V8 ldsv(const V8* p) {
    V8 v;
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.w[0]), "=r"(v.w[1]), "=r"(v.w[2]), "=r"(v.w[3]) : "r"(a));
    return v;
}
__device__ __forceinline__ void stsv(V8* p, const V8& v) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "r"(a), "r"(v.w[0]), "r"(v.w[1]), "r"(v.w[2]), "r"(v.w[3]) : "memory");
}


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
    bf16_t*       __restrict__ normed_row,// &normed[b * H]
    const float*  __restrict__ stats,     // [B, KNB_MAX, 2] from res_in
    int nb_stats,                         // candidates those statistics cover
    int row, int nb, int t)
{
    const int  lane = t & 31, warp = t >> 5;

    __shared__ float s_red[(KNB_MAX + 1) * 2 * KNWARPS];   // [value][warp] partials
    __shared__ float s_red2[KNWARPS];

    const int ncand = nb + 1;

    if (nb > 0) {
        // pass 1: (sum x^2, sum x*sw) per candidate.
        /* the candidates' (sq, dp_mlp) arrive from res_in (which reads the same
         * rows anyway); only the prefix candidate's statistics are left, and
         * they need the reduce-scatter's output, so they stay here.  The
         * accumulation, the butterfly and the wave of warp partials are the
         * same code as before, so the value is bit-identical. */
        float sq[KNB_MAX + 1], dp[KNB_MAX + 1];
#pragma unroll
        for (int c = 0; c <= KNB_MAX; ++c) { sq[c] = 0.f; dp[c] = 0.f; }
#pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const int u = (t + v * RES_THREADS) * 8;
            const float4* sp4 = (const float4*)(sw + u);
            const float4 a = sp4[0], b = sp4[1];
            const float swv[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
            const V8 pvv = ldsv(&pv[v * RES_THREADS]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(pvv.w[j]);
                sq[KNB_MAX] += f.x * f.x;  sq[KNB_MAX] += f.y * f.y;
                dp[KNB_MAX] += f.x * swv[2 * j];  dp[KNB_MAX] += f.y * swv[2 * j + 1];
            }
#pragma unroll
            for (int c0 = 0; c0 < KNB_MAX; c0 += CH) {
                if (c0 >= nb) break;
                if (c0 + CH <= nb_stats) continue;          /* res_in covered these */
                V8 x[CH];
#pragma unroll
                for (int q = 0; q < CH; ++q)
                    if (c0 + q >= nb_stats && c0 + q < nb)
                        x[q] = ldv_nv(blk_row + (size_t)(c0 + q) * KH + u);
#pragma unroll
                for (int q = 0; q < CH; ++q) {
                    if (c0 + q < nb_stats || c0 + q >= nb) continue;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = bf2f(x[q].w[j]);
                        sq[c0 + q] += f.x * f.x;  sq[c0 + q] += f.y * f.y;
                        dp[c0 + q] += f.x * swv[2 * j];  dp[c0 + q] += f.y * swv[2 * j + 1];
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < KNB_MAX; ++c) {
            if (c < nb_stats || c >= nb) continue;          /* covered by res_in */
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
        if (lane < nb_stats) {
            /* the candidate's statistics, computed by res_in with this kernel's
             * own tree and summation order */
            const float sqv = __ldg(&stats[((size_t)row * KNB_MAX + lane) * 2 + 0]);
            const float dpv = __ldg(&stats[((size_t)row * KNB_MAX + lane) * 2 + 1]);
            sc = dpv * rsqrtf(sqv * (1.0f / (float)KH) + KEPS);
        } else if (lane <= nb) {
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
            float acc[8];
            const V8 pv0 = ldsv(&pv[v * RES_THREADS]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(pv0.w[j]);
                acc[2 * j]     = pp * f.x;
                acc[2 * j + 1] = pp * f.y;
            }
#pragma unroll
            for (int c0 = 0; c0 < KNB_MAX; c0 += CH) {
                if (c0 >= nb) break;
                V8 x[CH];
#pragma unroll
                for (int q = 0; q < CH; ++q)
                    if (c0 + q < nb) x[q] = ldv_nv(blk_row + (size_t)(c0 + q) * KH + u);
#pragma unroll
                for (int q = 0; q < CH; ++q) {
                    if (c0 + q >= nb) break;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = bf2f(x[q].w[j]);
                        acc[2 * j]     += p[c0 + q] * f.x;
                        acc[2 * j + 1] += p[c0 + q] * f.y;
                    }
                }
            }
            V8 m;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                m.w[j] = f2bf(make_float2(acc[2 * j], acc[2 * j + 1]));
            stsv(&pv[v * RES_THREADS], m);
        }
    }
    // nb == 0: mixed = prefix, since bf16(1.0f * f32(prefix)) == prefix.

    // rms(mixed, gamma)
    float sq = 0.f;
#pragma unroll
    for (int v = 0; v < KVPT; ++v) {
        const V8 mv = ldsv(&pv[v * RES_THREADS]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = bf2f(mv.w[j]);
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
        const V8 mv = ldsv(&pv[v * RES_THREADS]);
        V8 o;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = bf2f(mv.w[j]);
            f.x *= r;  f.y *= r;
            o.w[j] = from_bf162(__hmul2(__float22bfloat162_rn(f), as_bf162(g.w[j])));
        }
        stv(normed_row + u, o);
    }
}
// ---------------------------------------------------------------- K1b
// -DLAND_BF16: the landing arrives as bf16 (a reduce-scatter's sum of the
// tray's o_proj partials), which is the value the f32 form rounds to first.
#ifdef LAND_BF16
typedef bf16_t land_t;
#else
typedef float land_t;
#endif

// ---------------------------------------------------------------- nb = 1
extern "C" __global__ void __launch_bounds__(RES_THREADS)
kern_k3_land_add_attnres_rms_nb1(
    const land_t* __restrict__ partial,   // [B, H]  partial of o_proj
    const bf16_t* __restrict__ prefix,    // [B, H]
    const bf16_t* __restrict__ blocks,    // [B, NB_MAX, H]
    const float*  __restrict__ sw,        // [H]
    const bf16_t* __restrict__ gamma,     // [H]
    bf16_t*       __restrict__ prefix2,   // [B, H]
    bf16_t*       __restrict__ normed,    // [B, H]
    int nb_in, int snapshot, int B,
    const float*  __restrict__ stats,     // [B, KNB_MAX, 2] from res_in
    int nb_stats)                         // candidates those statistics cover
{
    const int nb = 1;        /* compile-time: the candidate guards fold away */
    if (nb_in != 1) return;  /* this entry is built for one shape */
        const int b = blockIdx.x;
        if (b >= B) return;
        const int t = threadIdx.x;

        __shared__ V8 s_pv[KVPT * RES_THREADS];
        V8 pv[KVPT];
    #pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const size_t off = (size_t)b * KH + (t + v * RES_THREADS) * 8;
            float pf[8];
    #ifdef LAND_BF16
            const V8 lp = ldv(partial + off);
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(lp.w[j]);
                pf[2 * j] = f.x; pf[2 * j + 1] = f.y;
            }
    #else
            const float4* q = (const float4*)(partial + off);
            const float4 a = q[0], c = q[1];
            pf[0] = a.x; pf[1] = a.y; pf[2] = a.z; pf[3] = a.w;
            pf[4] = c.x; pf[5] = c.y; pf[6] = c.z; pf[7] = c.w;
    #endif
            if (snapshot) {
    #pragma unroll
                for (int j = 0; j < 4; ++j)
                    pv[v].w[j] = f2bf(make_float2(pf[2 * j], pf[2 * j + 1]));
            } else {
                const V8 pr = ldv(prefix + off);
    #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 rf = bf2f(pr.w[j]);
                    const float2 lf = bf2f(f2bf(make_float2(pf[2 * j], pf[2 * j + 1])));
                    pv[v].w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
                }
            }
            stv(prefix2 + off, pv[v]);
            stsv(&s_pv[v * RES_THREADS + t], pv[v]);
        }

        attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, &s_pv[t], sw, gamma,
                        normed + (size_t)b * KH, stats, nb_stats, b, nb, t);
}

// ---------------------------------------------------------------- nb = 2
extern "C" __global__ void __launch_bounds__(RES_THREADS)
kern_k3_land_add_attnres_rms_nb2(
    const land_t* __restrict__ partial,   // [B, H]  partial of o_proj
    const bf16_t* __restrict__ prefix,    // [B, H]
    const bf16_t* __restrict__ blocks,    // [B, NB_MAX, H]
    const float*  __restrict__ sw,        // [H]
    const bf16_t* __restrict__ gamma,     // [H]
    bf16_t*       __restrict__ prefix2,   // [B, H]
    bf16_t*       __restrict__ normed,    // [B, H]
    int nb_in, int snapshot, int B,
    const float*  __restrict__ stats,     // [B, KNB_MAX, 2] from res_in
    int nb_stats)                         // candidates those statistics cover
{
    const int nb = 2;        /* compile-time: the candidate guards fold away */
    if (nb_in != 2) return;  /* this entry is built for one shape */
        const int b = blockIdx.x;
        if (b >= B) return;
        const int t = threadIdx.x;

        __shared__ V8 s_pv[KVPT * RES_THREADS];
        V8 pv[KVPT];
    #pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const size_t off = (size_t)b * KH + (t + v * RES_THREADS) * 8;
            float pf[8];
    #ifdef LAND_BF16
            const V8 lp = ldv(partial + off);
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(lp.w[j]);
                pf[2 * j] = f.x; pf[2 * j + 1] = f.y;
            }
    #else
            const float4* q = (const float4*)(partial + off);
            const float4 a = q[0], c = q[1];
            pf[0] = a.x; pf[1] = a.y; pf[2] = a.z; pf[3] = a.w;
            pf[4] = c.x; pf[5] = c.y; pf[6] = c.z; pf[7] = c.w;
    #endif
            if (snapshot) {
    #pragma unroll
                for (int j = 0; j < 4; ++j)
                    pv[v].w[j] = f2bf(make_float2(pf[2 * j], pf[2 * j + 1]));
            } else {
                const V8 pr = ldv(prefix + off);
    #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 rf = bf2f(pr.w[j]);
                    const float2 lf = bf2f(f2bf(make_float2(pf[2 * j], pf[2 * j + 1])));
                    pv[v].w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
                }
            }
            stv(prefix2 + off, pv[v]);
            stsv(&s_pv[v * RES_THREADS + t], pv[v]);
        }

        attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, &s_pv[t], sw, gamma,
                        normed + (size_t)b * KH, stats, nb_stats, b, nb, t);
}

// ---------------------------------------------------------------- nb = 3
extern "C" __global__ void __launch_bounds__(RES_THREADS)
kern_k3_land_add_attnres_rms_nb3(
    const land_t* __restrict__ partial,   // [B, H]  partial of o_proj
    const bf16_t* __restrict__ prefix,    // [B, H]
    const bf16_t* __restrict__ blocks,    // [B, NB_MAX, H]
    const float*  __restrict__ sw,        // [H]
    const bf16_t* __restrict__ gamma,     // [H]
    bf16_t*       __restrict__ prefix2,   // [B, H]
    bf16_t*       __restrict__ normed,    // [B, H]
    int nb_in, int snapshot, int B,
    const float*  __restrict__ stats,     // [B, KNB_MAX, 2] from res_in
    int nb_stats)                         // candidates those statistics cover
{
    const int nb = 3;        /* compile-time: the candidate guards fold away */
    if (nb_in != 3) return;  /* this entry is built for one shape */
        const int b = blockIdx.x;
        if (b >= B) return;
        const int t = threadIdx.x;

        __shared__ V8 s_pv[KVPT * RES_THREADS];
        V8 pv[KVPT];
    #pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const size_t off = (size_t)b * KH + (t + v * RES_THREADS) * 8;
            float pf[8];
    #ifdef LAND_BF16
            const V8 lp = ldv(partial + off);
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(lp.w[j]);
                pf[2 * j] = f.x; pf[2 * j + 1] = f.y;
            }
    #else
            const float4* q = (const float4*)(partial + off);
            const float4 a = q[0], c = q[1];
            pf[0] = a.x; pf[1] = a.y; pf[2] = a.z; pf[3] = a.w;
            pf[4] = c.x; pf[5] = c.y; pf[6] = c.z; pf[7] = c.w;
    #endif
            if (snapshot) {
    #pragma unroll
                for (int j = 0; j < 4; ++j)
                    pv[v].w[j] = f2bf(make_float2(pf[2 * j], pf[2 * j + 1]));
            } else {
                const V8 pr = ldv(prefix + off);
    #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 rf = bf2f(pr.w[j]);
                    const float2 lf = bf2f(f2bf(make_float2(pf[2 * j], pf[2 * j + 1])));
                    pv[v].w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
                }
            }
            stv(prefix2 + off, pv[v]);
            stsv(&s_pv[v * RES_THREADS + t], pv[v]);
        }

        attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, &s_pv[t], sw, gamma,
                        normed + (size_t)b * KH, stats, nb_stats, b, nb, t);
}

// ---------------------------------------------------------------- nb = 4
extern "C" __global__ void __launch_bounds__(RES_THREADS)
kern_k3_land_add_attnres_rms_nb4(
    const land_t* __restrict__ partial,   // [B, H]  partial of o_proj
    const bf16_t* __restrict__ prefix,    // [B, H]
    const bf16_t* __restrict__ blocks,    // [B, NB_MAX, H]
    const float*  __restrict__ sw,        // [H]
    const bf16_t* __restrict__ gamma,     // [H]
    bf16_t*       __restrict__ prefix2,   // [B, H]
    bf16_t*       __restrict__ normed,    // [B, H]
    int nb_in, int snapshot, int B,
    const float*  __restrict__ stats,     // [B, KNB_MAX, 2] from res_in
    int nb_stats)                         // candidates those statistics cover
{
    const int nb = 4;        /* compile-time: the candidate guards fold away */
    if (nb_in != 4) return;  /* this entry is built for one shape */
        const int b = blockIdx.x;
        if (b >= B) return;
        const int t = threadIdx.x;

        __shared__ V8 s_pv[KVPT * RES_THREADS];
        V8 pv[KVPT];
    #pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const size_t off = (size_t)b * KH + (t + v * RES_THREADS) * 8;
            float pf[8];
    #ifdef LAND_BF16
            const V8 lp = ldv(partial + off);
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(lp.w[j]);
                pf[2 * j] = f.x; pf[2 * j + 1] = f.y;
            }
    #else
            const float4* q = (const float4*)(partial + off);
            const float4 a = q[0], c = q[1];
            pf[0] = a.x; pf[1] = a.y; pf[2] = a.z; pf[3] = a.w;
            pf[4] = c.x; pf[5] = c.y; pf[6] = c.z; pf[7] = c.w;
    #endif
            if (snapshot) {
    #pragma unroll
                for (int j = 0; j < 4; ++j)
                    pv[v].w[j] = f2bf(make_float2(pf[2 * j], pf[2 * j + 1]));
            } else {
                const V8 pr = ldv(prefix + off);
    #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 rf = bf2f(pr.w[j]);
                    const float2 lf = bf2f(f2bf(make_float2(pf[2 * j], pf[2 * j + 1])));
                    pv[v].w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
                }
            }
            stv(prefix2 + off, pv[v]);
            stsv(&s_pv[v * RES_THREADS + t], pv[v]);
        }

        attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, &s_pv[t], sw, gamma,
                        normed + (size_t)b * KH, stats, nb_stats, b, nb, t);
}

// ---------------------------------------------------------------- nb = 5
extern "C" __global__ void __launch_bounds__(RES_THREADS)
kern_k3_land_add_attnres_rms_nb5(
    const land_t* __restrict__ partial,   // [B, H]  partial of o_proj
    const bf16_t* __restrict__ prefix,    // [B, H]
    const bf16_t* __restrict__ blocks,    // [B, NB_MAX, H]
    const float*  __restrict__ sw,        // [H]
    const bf16_t* __restrict__ gamma,     // [H]
    bf16_t*       __restrict__ prefix2,   // [B, H]
    bf16_t*       __restrict__ normed,    // [B, H]
    int nb_in, int snapshot, int B,
    const float*  __restrict__ stats,     // [B, KNB_MAX, 2] from res_in
    int nb_stats)                         // candidates those statistics cover
{
    const int nb = 5;        /* compile-time: the candidate guards fold away */
    if (nb_in != 5) return;  /* this entry is built for one shape */
        const int b = blockIdx.x;
        if (b >= B) return;
        const int t = threadIdx.x;

        __shared__ V8 s_pv[KVPT * RES_THREADS];
        V8 pv[KVPT];
    #pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const size_t off = (size_t)b * KH + (t + v * RES_THREADS) * 8;
            float pf[8];
    #ifdef LAND_BF16
            const V8 lp = ldv(partial + off);
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(lp.w[j]);
                pf[2 * j] = f.x; pf[2 * j + 1] = f.y;
            }
    #else
            const float4* q = (const float4*)(partial + off);
            const float4 a = q[0], c = q[1];
            pf[0] = a.x; pf[1] = a.y; pf[2] = a.z; pf[3] = a.w;
            pf[4] = c.x; pf[5] = c.y; pf[6] = c.z; pf[7] = c.w;
    #endif
            if (snapshot) {
    #pragma unroll
                for (int j = 0; j < 4; ++j)
                    pv[v].w[j] = f2bf(make_float2(pf[2 * j], pf[2 * j + 1]));
            } else {
                const V8 pr = ldv(prefix + off);
    #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 rf = bf2f(pr.w[j]);
                    const float2 lf = bf2f(f2bf(make_float2(pf[2 * j], pf[2 * j + 1])));
                    pv[v].w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
                }
            }
            stv(prefix2 + off, pv[v]);
            stsv(&s_pv[v * RES_THREADS + t], pv[v]);
        }

        attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, &s_pv[t], sw, gamma,
                        normed + (size_t)b * KH, stats, nb_stats, b, nb, t);
}

// ---------------------------------------------------------------- nb = 6
extern "C" __global__ void __launch_bounds__(RES_THREADS)
kern_k3_land_add_attnres_rms_nb6(
    const land_t* __restrict__ partial,   // [B, H]  partial of o_proj
    const bf16_t* __restrict__ prefix,    // [B, H]
    const bf16_t* __restrict__ blocks,    // [B, NB_MAX, H]
    const float*  __restrict__ sw,        // [H]
    const bf16_t* __restrict__ gamma,     // [H]
    bf16_t*       __restrict__ prefix2,   // [B, H]
    bf16_t*       __restrict__ normed,    // [B, H]
    int nb_in, int snapshot, int B,
    const float*  __restrict__ stats,     // [B, KNB_MAX, 2] from res_in
    int nb_stats)                         // candidates those statistics cover
{
    const int nb = 6;        /* compile-time: the candidate guards fold away */
    if (nb_in != 6) return;  /* this entry is built for one shape */
        const int b = blockIdx.x;
        if (b >= B) return;
        const int t = threadIdx.x;

        __shared__ V8 s_pv[KVPT * RES_THREADS];
        V8 pv[KVPT];
    #pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const size_t off = (size_t)b * KH + (t + v * RES_THREADS) * 8;
            float pf[8];
    #ifdef LAND_BF16
            const V8 lp = ldv(partial + off);
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(lp.w[j]);
                pf[2 * j] = f.x; pf[2 * j + 1] = f.y;
            }
    #else
            const float4* q = (const float4*)(partial + off);
            const float4 a = q[0], c = q[1];
            pf[0] = a.x; pf[1] = a.y; pf[2] = a.z; pf[3] = a.w;
            pf[4] = c.x; pf[5] = c.y; pf[6] = c.z; pf[7] = c.w;
    #endif
            if (snapshot) {
    #pragma unroll
                for (int j = 0; j < 4; ++j)
                    pv[v].w[j] = f2bf(make_float2(pf[2 * j], pf[2 * j + 1]));
            } else {
                const V8 pr = ldv(prefix + off);
    #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 rf = bf2f(pr.w[j]);
                    const float2 lf = bf2f(f2bf(make_float2(pf[2 * j], pf[2 * j + 1])));
                    pv[v].w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
                }
            }
            stv(prefix2 + off, pv[v]);
            stsv(&s_pv[v * RES_THREADS + t], pv[v]);
        }

        attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, &s_pv[t], sw, gamma,
                        normed + (size_t)b * KH, stats, nb_stats, b, nb, t);
}

// ---------------------------------------------------------------- nb = 7
extern "C" __global__ void __launch_bounds__(RES_THREADS)
kern_k3_land_add_attnres_rms_nb7(
    const land_t* __restrict__ partial,   // [B, H]  partial of o_proj
    const bf16_t* __restrict__ prefix,    // [B, H]
    const bf16_t* __restrict__ blocks,    // [B, NB_MAX, H]
    const float*  __restrict__ sw,        // [H]
    const bf16_t* __restrict__ gamma,     // [H]
    bf16_t*       __restrict__ prefix2,   // [B, H]
    bf16_t*       __restrict__ normed,    // [B, H]
    int nb_in, int snapshot, int B,
    const float*  __restrict__ stats,     // [B, KNB_MAX, 2] from res_in
    int nb_stats)                         // candidates those statistics cover
{
    const int nb = 7;        /* compile-time: the candidate guards fold away */
    if (nb_in != 7) return;  /* this entry is built for one shape */
        const int b = blockIdx.x;
        if (b >= B) return;
        const int t = threadIdx.x;

        __shared__ V8 s_pv[KVPT * RES_THREADS];
        V8 pv[KVPT];
    #pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const size_t off = (size_t)b * KH + (t + v * RES_THREADS) * 8;
            float pf[8];
    #ifdef LAND_BF16
            const V8 lp = ldv(partial + off);
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(lp.w[j]);
                pf[2 * j] = f.x; pf[2 * j + 1] = f.y;
            }
    #else
            const float4* q = (const float4*)(partial + off);
            const float4 a = q[0], c = q[1];
            pf[0] = a.x; pf[1] = a.y; pf[2] = a.z; pf[3] = a.w;
            pf[4] = c.x; pf[5] = c.y; pf[6] = c.z; pf[7] = c.w;
    #endif
            if (snapshot) {
    #pragma unroll
                for (int j = 0; j < 4; ++j)
                    pv[v].w[j] = f2bf(make_float2(pf[2 * j], pf[2 * j + 1]));
            } else {
                const V8 pr = ldv(prefix + off);
    #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 rf = bf2f(pr.w[j]);
                    const float2 lf = bf2f(f2bf(make_float2(pf[2 * j], pf[2 * j + 1])));
                    pv[v].w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
                }
            }
            stv(prefix2 + off, pv[v]);
            stsv(&s_pv[v * RES_THREADS + t], pv[v]);
        }

        attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, &s_pv[t], sw, gamma,
                        normed + (size_t)b * KH, stats, nb_stats, b, nb, t);
}

// ---------------------------------------------------------------- nb = 8
extern "C" __global__ void __launch_bounds__(RES_THREADS)
kern_k3_land_add_attnres_rms_nb8(
    const land_t* __restrict__ partial,   // [B, H]  partial of o_proj
    const bf16_t* __restrict__ prefix,    // [B, H]
    const bf16_t* __restrict__ blocks,    // [B, NB_MAX, H]
    const float*  __restrict__ sw,        // [H]
    const bf16_t* __restrict__ gamma,     // [H]
    bf16_t*       __restrict__ prefix2,   // [B, H]
    bf16_t*       __restrict__ normed,    // [B, H]
    int nb_in, int snapshot, int B,
    const float*  __restrict__ stats,     // [B, KNB_MAX, 2] from res_in
    int nb_stats)                         // candidates those statistics cover
{
    const int nb = 8;        /* compile-time: the candidate guards fold away */
    if (nb_in != 8) return;  /* this entry is built for one shape */
        const int b = blockIdx.x;
        if (b >= B) return;
        const int t = threadIdx.x;

        __shared__ V8 s_pv[KVPT * RES_THREADS];
        V8 pv[KVPT];
    #pragma unroll
        for (int v = 0; v < KVPT; ++v) {
            const size_t off = (size_t)b * KH + (t + v * RES_THREADS) * 8;
            float pf[8];
    #ifdef LAND_BF16
            const V8 lp = ldv(partial + off);
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(lp.w[j]);
                pf[2 * j] = f.x; pf[2 * j + 1] = f.y;
            }
    #else
            const float4* q = (const float4*)(partial + off);
            const float4 a = q[0], c = q[1];
            pf[0] = a.x; pf[1] = a.y; pf[2] = a.z; pf[3] = a.w;
            pf[4] = c.x; pf[5] = c.y; pf[6] = c.z; pf[7] = c.w;
    #endif
            if (snapshot) {
    #pragma unroll
                for (int j = 0; j < 4; ++j)
                    pv[v].w[j] = f2bf(make_float2(pf[2 * j], pf[2 * j + 1]));
            } else {
                const V8 pr = ldv(prefix + off);
    #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 rf = bf2f(pr.w[j]);
                    const float2 lf = bf2f(f2bf(make_float2(pf[2 * j], pf[2 * j + 1])));
                    pv[v].w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
                }
            }
            stv(prefix2 + off, pv[v]);
            stsv(&s_pv[v * RES_THREADS + t], pv[v]);
        }

        attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, &s_pv[t], sw, gamma,
                        normed + (size_t)b * KH, stats, nb_stats, b, nb, t);
}
