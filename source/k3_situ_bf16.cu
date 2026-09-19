#include <cuda_bf16.h>

__device__ __forceinline__ float tanh_approx(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

extern "C" __global__ void kern_k3_situ_bf16(
    const __nv_bfloat16* __restrict__ p, __nv_bfloat16* __restrict__ out,
    int n, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const __nv_bfloat16* gate = p + (long long)row * 2 * n;
  const __nv_bfloat16* up = gate + n;
  for (int i = blockIdx.y * blockDim.x + threadIdx.x;
       i < n; i += gridDim.y * blockDim.x) {
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    float a = 4.0f * tanh_approx(g * 0.25f);
    float s = __frcp_rn(1.0f + __expf(-g));
    float c = 25.0f * tanh_approx(u * 0.04f);
    out[(long long)row * n + i] = __float2bfloat16((a * s) * c);
  }
}
