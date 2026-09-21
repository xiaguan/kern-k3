// Landing kernels that take an already-landed bf16 partial, so the GEMM that
// produces it can write bf16 instead of f32.
//
//   extern "C" __global__ void kern_k3_land_situ_bf16(
//       const bf16* p, bf16* act, int n, int B);
//   grid (B, ceil(n/1024))   block 128 (or any (grid.y, block) covering the row)
//   gate in the first n columns of the row, up in the next n (ldc == 2n):
//       g = f32(bf16(p[b*2n + i]));  u = f32(bf16(p[b*2n + n + i]))
//       act[b,i] = bf16( 4*tanh(g/4) * sigmoid(g) * 25*tanh(u/25) )
//
// This is k3_land.cu's kern_k3_land_situ with the input read as bf16.  That
// kernel rounds both operands to bf16 before the activation anyway
// (situ_f does `f32(bf16(p))`), so when its producer lands bf16 the values
// entering situ_f are the same numbers and the output is bit-identical -- the
// same argument as k3_span_gather_v3 / k3_kda_out_gate_v3.  The redundant
// round trip is kept in situ_f below so the two kernels stay literally the same
// arithmetic, and the (grid.y, block) sweep is free: the kernel is elementwise
// and every geometry computes each output element from the same inputs.
//
// The other landing kernels (rms, land_add2 and the residual stream) live in
// k3_land.cu / k3_residual_v2.cu; this module exists so that adding bf16
// variants does not move the pinned SHA-256 of a module other manifests and
// other agents' areas also pin.
//
//   nvcc -cubin -arch=sm_103a -O3 -o k3_land_bf16.cubin k3_land_bf16.cu

#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float tanh_approx_bf16(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// land both operands to bf16 first, then the activation in f32
__device__ __forceinline__ float situ_f_bf16(float pg, float pu) {
  float g = __bfloat162float(__float2bfloat16(pg));
  float u = __bfloat162float(__float2bfloat16(pu));
  float a = 4.0f * tanh_approx_bf16(g * 0.25f);
  float s = __frcp_rn(1.0f + __expf(-g));
  float c = 25.0f * tanh_approx_bf16(u * 0.04f);
  return (a * s) * c;
}

// 4 consecutive bf16 as a float4 (the shape kern_k3_land_situ loads)
__device__ __forceinline__ float4 bf16x4_to_f32(uint2 raw) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
  const float2 l = __bfloat1622float2(lo);
  const float2 h = __bfloat1622float2(hi);
  return make_float4(l.x, l.y, h.x, h.y);
}

extern "C" __global__ void __launch_bounds__(1024, 2) kern_k3_land_situ_bf16(
    const bf16* __restrict__ p, bf16* __restrict__ act, int n, int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const bf16* __restrict__ pg = p + (long long)b * 2 * n;
  const bf16* __restrict__ pu = pg + n;
  bf16* __restrict__ orow = act + (long long)b * n;
  const int stride = gridDim.y * blockDim.x;
  const int i0 = blockIdx.y * blockDim.x + threadIdx.x;

  if ((n & 3) == 0) {
    const int nv = n >> 2;                       // 4 bf16 per uint2
    const uint2* gv = (const uint2*)pg;
    const uint2* uv = (const uint2*)pu;
    for (int k = i0; k < nv; k += stride) {
      const float4 g = bf16x4_to_f32(gv[k]);
      const float4 u = bf16x4_to_f32(uv[k]);
      __nv_bfloat162 out2[2];
      out2[0] = __floats2bfloat162_rn(situ_f_bf16(g.x, u.x), situ_f_bf16(g.y, u.y));
      out2[1] = __floats2bfloat162_rn(situ_f_bf16(g.z, u.z), situ_f_bf16(g.w, u.w));
      *(uint2*)(orow + (k << 2)) = *(const uint2*)out2;
    }
  } else {
    for (int i = i0; i < n; i += stride)
      orow[i] = __float2bfloat16(situ_f_bf16(__bfloat162float(pg[i]), __bfloat162float(pu[i])));
  }
}

// ------------------------------------------------------------------ land_add2
// kern_k3_land_add2 (k3_residual_v2.cu) with bf16 partials:
//     out[b,i] = bf16( f32(prefix2[b,i]) + f32(bf16(p1[b,i]))
//                      [+ f32(bf16(p2[b,i])) when two != 0] )
// computed in that order, exactly as K1c documents it.  Its two partial inputs
// are rounded through bf16 before the sum anyway, so when their producers land
// bf16 the round trips are no-ops and `hidden` is bit-identical -- and the sum
// is per element with no cross-thread reduction, so the (grid.y, block)
// geometry is free.
#define KH 7168
#define KHV (KH / 8)   /* 896 vectors of 8 bf16 per row */

extern "C" __global__ void __launch_bounds__(1024, 2) kern_k3_land_add2_bf16(
    const bf16* __restrict__ p1,      // [B, KH]
    const bf16* __restrict__ p2,      // [B, KH]  read only when two != 0
    const bf16* __restrict__ prefix2, // [B, KH]
    bf16* __restrict__ hidden,        // [B, KH]
    int two, int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const int stride = gridDim.y * blockDim.x;
  const long long row = (long long)b * KH;
  const uint4* v1 = (const uint4*)(p1 + row);
  const uint4* v2 = (const uint4*)(p2 + row);
  const uint4* vp = (const uint4*)(prefix2 + row);
  uint4* vo = (uint4*)(hidden + row);

  for (int v = blockIdx.y * blockDim.x + threadIdx.x; v < KHV; v += stride) {
    float f1[8], f2[8], fp[8];
    {
      const __nv_bfloat162* q = (const __nv_bfloat162*)&v1[v];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 t = __bfloat1622float2(q[j]);
        f1[2 * j] = t.x; f1[2 * j + 1] = t.y;
      }
    }
    {
      const __nv_bfloat162* q = (const __nv_bfloat162*)&vp[v];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 t = __bfloat1622float2(q[j]);
        fp[2 * j] = t.x; fp[2 * j + 1] = t.y;
      }
    }
    if (two) {
      const __nv_bfloat162* q = (const __nv_bfloat162*)&v2[v];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 t = __bfloat1622float2(q[j]);
        f2[2 * j] = t.x; f2[2 * j + 1] = t.y;
      }
    }
    uint4 o;
    __nv_bfloat162* oq = (__nv_bfloat162*)&o;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float x = fp[2 * j] + f1[2 * j];
      float y = fp[2 * j + 1] + f1[2 * j + 1];
      if (two) { x += f2[2 * j]; y += f2[2 * j + 1]; }
      oq[j] = __floats2bfloat162_rn(x, y);
    }
    vo[v] = o;
  }
}
