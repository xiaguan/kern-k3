// bf16 -> e4m3 per-tensor quantization, the operand half of the dense fp8 GEMMs.  Two launches per
// tensor, so no atomics and no pre-zeroed scratch:
//
//   kern_quant_fp8_amax(x, n, partials)              partials[b] = max |x| over block b's slice
//   kern_quant_fp8_apply(x, y, n, partials, nb, sc)  scale = amax/448 into sc[0], y = e4m3(x/scale)
//
// e4m3's largest normal is 448; the GEMM's epilogue multiplies by scale_a * scale_b, so the pair
// (fp8 operand, scale) reproduces the bf16 operand to within one e4m3 ulp of amax/448.  Launch both
// with the same grid; `nb` is that grid width.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

typedef __nv_bfloat16 bf16;
typedef unsigned char u8;

#define NTHREADS 256

extern "C" __global__ void kern_quant_fp8_amax(const bf16* __restrict__ x, int n, float* __restrict__ partials) {
  float m = 0.f;
  for (int i = blockIdx.x * NTHREADS + threadIdx.x; i < n; i += gridDim.x * NTHREADS)
    m = fmaxf(m, fabsf(__bfloat162float(x[i])));
  __shared__ float s[NTHREADS];
  s[threadIdx.x] = m;
  __syncthreads();
  for (int off = NTHREADS >> 1; off > 0; off >>= 1) {
    if (threadIdx.x < off) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + off]);
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = s[0];
}

extern "C" __global__ void kern_quant_fp8_apply(const bf16* __restrict__ x, u8* __restrict__ y, int n,
                                                const float* __restrict__ partials, int nb,
                                                float* __restrict__ scale_out) {
  __shared__ float s[NTHREADS];
  float m = 0.f;
  for (int b = threadIdx.x; b < nb; b += NTHREADS) m = fmaxf(m, partials[b]);
  s[threadIdx.x] = m;
  __syncthreads();
  for (int off = NTHREADS >> 1; off > 0; off >>= 1) {
    if (threadIdx.x < off) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + off]);
    __syncthreads();
  }
  float scale = s[0] > 0.f ? s[0] / 448.f : 1.f;
  if (blockIdx.x == 0 && threadIdx.x == 0 && scale_out) *scale_out = scale;
  float inv = 1.f / scale;
  for (int i = blockIdx.x * NTHREADS + threadIdx.x; i < n; i += gridDim.x * NTHREADS) {
    __nv_fp8_e4m3 q = __nv_fp8_e4m3(__bfloat162float(x[i]) * inv);
    y[i] = *reinterpret_cast<u8*>(&q);
  }
}
