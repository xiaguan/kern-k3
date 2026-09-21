// K11 `k3_kda_out_gate` v3 -- v2 with a bf16 gate band.
//
// Same ABI, math, landing points, thread layout and launch geometry as v2
// (docs/k3-kernel-abi.md section K11); the only difference is that the second
// argument `gate_partial` is already landed to bf16 by its producer.  v2
// loads f32 and rounds every element through bf16 before the sigmoid
// (`gg = f32(bf16(g[...]))`); v3 loads those same bf16 values directly, so
// the gate it applies is bit-identical.

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
    const bf16* __restrict__ attn, const bf16* __restrict__ gate_partial, const float* __restrict__ gamma_o,
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
  // gate band: K11_EPL bf16 in one 16 B vector; gamma_o: K11_EPL f32, 16 B vectors
  float g[K11_EPL], gm[K11_EPL];
  {
    const unsigned* gp = (const unsigned*)(gate_partial + (size_t)b * K11_KDA_FUSED + 3 * K11_INNER + col);
    const float4* gmp = (const float4*)(gamma_o + d0);
#if K11_EPL == 8
    const uint4 w = *(const uint4*)gp;
    const unsigned u[4] = {w.x, w.y, w.z, w.w};
#else
    const uint2 w = *(const uint2*)gp;
    const unsigned u[2] = {w.x, w.y};
#endif
#pragma unroll
    for (int k = 0; k < K11_EPL / 2; ++k) {
      g[2 * k] = k11_bf16_bits_to_f32(u[k] & 0xffffu);
      g[2 * k + 1] = k11_bf16_bits_to_f32(u[k] >> 16);
    }
#pragma unroll
    for (int k = 0; k < K11_EPL / 4; ++k) {
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
