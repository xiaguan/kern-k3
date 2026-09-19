// K11 `k3_kda_out_gate` v2 — the K3 epilogue for span rows, same ABI, math and
// landing points as k3_kda_out_gate.cu (docs/k3-kernel-abi.md section K11);
// only the thread layout, the launch geometry and the reduction order differ.
//
//   extern "C" __global__ void kern_k3_kda_out_gate(
//       const bf16* attn,          // [span, INNER]
//       const float* gate_partial, // [rows, KDA_FUSED]  band 3 only, rows at..at+span
//       const float* gamma_o,      // [128]
//       bf16* gated,               // [rows, INNER]  rows at..at+span written
//       const int* span_at,        // [1]  the span's first batch row
//       int span);
//
//   grid (span, HEADS/K11_HPB, 1)   block (K11_HPB * K11_LPH, 1, 1)   smem 0
//   defaults K11_HPB = 8 heads per block, K11_LPH = 16 lanes per head:
//   grid (span, 3, 1) block 128 at HEADS = 24
//   (i = blockIdx.x, h = K11_HPB * blockIdx.y + threadIdx.x / K11_LPH,
//    dv = K11_EPL * (threadIdx.x % K11_LPH) .. + K11_EPL; b = at + i)
//
//   a       = f32(attn[i, h*128 + dv])
//   r       = rsqrt(mean_dv(a^2) + 1e-5)
//   o       = bf16(a * r * gamma_o[dv])
//   gt      = bf16(sigmoid(f32(bf16(gate_partial[b, 3*INNER + h*128 + dv]))))
//   gated[b, h*128 + dv] = bf16(f32(o) * f32(gt))
//
// Why v2.  v1 ran one 128-thread block per (row, head) with one element per
// thread: 2 B / 4 B / 2 B per thread and a shared-memory reduction with a
// block barrier, i.e. 24 * span = 393k tiny blocks per call.  At span = 16384
// that took ~207 us in the prefill graph for ~400 MB of traffic (~60 us of
// DRAM time).  v2 gives each head K11_LPH lanes of one warp (default 16, so
// one warp finishes two heads) with K11_EPL = 128 / K11_LPH consecutive
// elements per lane: 16 B vector loads and stores, a shuffle-only reduction,
// no shared memory, no barrier, and 8 heads per block, so the block count
// drops 8x and each lane keeps 48 B of loads in flight.  Standalone at
// span = 16384, HEADS = 24 on GB300: 204 us -> 72 us (32 lanes per head 80 us,
// 4 or 8 heads per block equal, 24 per block 74 us).
//
// The sum of squares is fixed-order (K11_EPL serial squares per lane, then a
// butterfly over the K11_LPH lanes), so a result never depends on the
// schedule, but the order differs from v1 (one square per thread, warp
// butterfly, then the four warp partials): r may differ by f32 rounding, so
// `gated` may occasionally differ from v1 by one bf16 ulp.
//
//   nvcc -cubin -arch=sm_103a -DHEADS=24 -o k3_kda_out_gate_v2.cubin k3_kda_out_gate_v2.cu
#include <cuda_bf16.h>

#ifndef HEADS
#define HEADS 96
#endif
#ifndef K11_LPH
#define K11_LPH 16                 /* lanes per head: 16 or 32 */
#endif
#define K11_EPL (128 / K11_LPH)    /* elements per lane: 8 or 4 */
#ifndef K11_HPB
#define K11_HPB 8                  /* heads per block */
#endif
#define K11_THREADS (K11_HPB * K11_LPH)
#define K11_INNER (HEADS * 128)
#define K11_KDA_FUSED (4 * K11_INNER)
#define K11_RMS_EPS 1e-5f
#if (HEADS % K11_HPB) != 0 || (K11_LPH != 16 && K11_LPH != 32)
#error "HEADS must be a multiple of 8 and K11_LPH 16 or 32"
#endif

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float k11_bf16_bits_to_f32(unsigned u) {
  return __bfloat162float(__ushort_as_bfloat16((unsigned short)u));
}

extern "C" __global__ __launch_bounds__(K11_THREADS) void kern_k3_kda_out_gate(
    const bf16* __restrict__ attn, const float* __restrict__ gate_partial, const float* __restrict__ gamma_o,
    bf16* __restrict__ gated, const int* __restrict__ span_at, int span) {
  const int i = blockIdx.x;
  const int h = blockIdx.y * K11_HPB + threadIdx.x / K11_LPH;
  const int lane = threadIdx.x % K11_LPH;
  const int d0 = lane * K11_EPL;
  const size_t col = (size_t)h * 128 + d0;

  // attn: K11_EPL bf16 in one 8 B / 16 B vector
  float a[K11_EPL];
  {
    const unsigned* src = (const unsigned*)(attn + (size_t)i * K11_INNER + col);
#if K11_EPL == 8
    const uint4 w = *(const uint4*)src;
    const unsigned u[4] = {w.x, w.y, w.z, w.w};
#else
    const uint2 w = *(const uint2*)src;
    const unsigned u[2] = {w.x, w.y};
#endif
#pragma unroll
    for (int k = 0; k < K11_EPL / 2; ++k) {
      a[2 * k] = k11_bf16_bits_to_f32(u[k] & 0xffffu);
      a[2 * k + 1] = k11_bf16_bits_to_f32(u[k] >> 16);
    }
  }
  const int b = span_at[0] + i;
  // gate band and gamma_o: K11_EPL f32 each, 16 B vectors
  float g[K11_EPL], gm[K11_EPL];
  {
    const float4* gp = (const float4*)(gate_partial + (size_t)b * K11_KDA_FUSED + 3 * K11_INNER + col);
    const float4* gmp = (const float4*)(gamma_o + d0);
#pragma unroll
    for (int k = 0; k < K11_EPL / 4; ++k) {
      const float4 v = gp[k];
      g[4 * k] = v.x; g[4 * k + 1] = v.y; g[4 * k + 2] = v.z; g[4 * k + 3] = v.w;
      const float4 m = gmp[k];
      gm[4 * k] = m.x; gm[4 * k + 1] = m.y; gm[4 * k + 2] = m.z; gm[4 * k + 3] = m.w;
    }
  }

  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < K11_EPL; ++k) ss += a[k] * a[k];
#pragma unroll
  for (int o = K11_LPH / 2; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float r = rsqrtf(ss * (1.0f / 128.0f) + K11_RMS_EPS);

  unsigned out[K11_EPL / 2];
#pragma unroll
  for (int k = 0; k < K11_EPL / 2; ++k) {
    unsigned short lo, hi;
    {
      const bf16 o = __float2bfloat16(a[2 * k] * r * gm[2 * k]);
      const float gg = __bfloat162float(__float2bfloat16(g[2 * k]));
      const bf16 gt = __float2bfloat16(1.0f / (1.0f + __expf(-gg)));
      lo = __bfloat16_as_ushort(__float2bfloat16(__bfloat162float(o) * __bfloat162float(gt)));
    }
    {
      const bf16 o = __float2bfloat16(a[2 * k + 1] * r * gm[2 * k + 1]);
      const float gg = __bfloat162float(__float2bfloat16(g[2 * k + 1]));
      const bf16 gt = __float2bfloat16(1.0f / (1.0f + __expf(-gg)));
      hi = __bfloat16_as_ushort(__float2bfloat16(__bfloat162float(o) * __bfloat162float(gt)));
    }
    out[k] = (unsigned)lo | ((unsigned)hi << 16);
  }
  unsigned* dst = (unsigned*)(gated + (size_t)b * K11_INNER + col);
#if K11_EPL == 8
  *(uint4*)dst = make_uint4(out[0], out[1], out[2], out[3]);
#else
  *(uint2*)dst = make_uint2(out[0], out[1]);
#endif
}
