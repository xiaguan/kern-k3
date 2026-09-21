// The shared-expert SiTU landing and the latent RMSNorm, in one launch (v2).
//
//   extern "C" __global__ void kern_k3_situ_rms_bf16(
//       const bf16* sp, bf16* act, const bf16* lat, const bf16* gam,
//       bf16* ln, int n, int B, int h, int L);
//   grid ((RATIO + 1) * ceil_div(L, RMS_R), 1, 1)   block THREADS
//
// One block does one job.  Of every RATIO + 1 consecutive blocks, block 0 of
// the group does RMS_R consecutive rows of the latent RMSNorm and the RATIO
// blocks after it do SITU_R consecutive rows of the shared-expert SiTU landing
// each, so the two streams are resident on an SM at the same time and the
// latency-bound rms stream runs under the DRAM/SFU-bound situ stream.
//
// Both jobs keep the arithmetic of the kernels they replace element for
// element: the situ part is elementwise (as kern_k3_land_situ with the
// operands already landed to bf16), and the rms part keeps warp w on vectors
// [32w, 32w + 32), the idle warps contributing 0, the same block_sum butterfly
// over the warps, the same rsqrtf(sum/h + 1e-5f) and the same
// __hmul2(__floats2bfloat162_rn(x*rs), gamma) landing as kern_k3_rms at block
// 512.  THREADS must stay 512 for that reason.
//
// v1 used RMS_R = 4, RATIO = 2 (3 * ceil_div(tokens, 16) blocks, SITU_R = 2).
// Measured standalone over the same data: v1 51.7 us, this 45.1 us for the
// same 209.7 MB (4096 situ rows of 6144, 4096 rms rows of 3584, bf16) --
// six percent off the situ-only floor of 38.7 us.  The situ stream uses
// __ldcs/__stcs (evict-first): both operands are read exactly once and the
// rows are 12 KB, so keeping them out of L2 costs nothing and the cache stays
// with the rms stream; 46.2 -> 43.6 us over three alternating runs each.
//
//   nvcc -cubin -arch=sm_103a [-DRMS_R=n -DRATIO=n -DSITU_R=n] \
//        -o k3_land_fused_v2.cubin k3_land_fused_v2.cu
#include <cuda_bf16.h>

#ifndef RMS_R
#define RMS_R 1
#endif
#ifndef RATIO
#define RATIO 1
#endif
#ifndef SITU_R
#define SITU_R 1
#endif
#ifndef THREADS
#define THREADS 512
#endif

typedef unsigned int u32;
typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void bf16x8_to_f32(const uint4& w, float* f) {
  const __nv_bfloat162* h2 = (const __nv_bfloat162*)&w;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 t = __bfloat1622float2(h2[k]);
    f[2 * k] = t.x;
    f[2 * k + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 rms_scale8(const uint4& xw, const uint4& gw,
                                            float rs) {
  const __nv_bfloat162* x2 = (const __nv_bfloat162*)&xw;
  const __nv_bfloat162* g2 = (const __nv_bfloat162*)&gw;
  uint4 ow;
  __nv_bfloat162* o2 = (__nv_bfloat162*)&ow;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 f = __bfloat1622float2(x2[k]);
    o2[k] = __hmul2(__floats2bfloat162_rn(f.x * rs, f.y * rs), g2[k]);
  }
  return ow;
}

__device__ __forceinline__ float block_sum(float v, float* sm) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  if (lane == 0) sm[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < nwarps) ? sm[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) sm[32] = v;
  }
  __syncthreads();
  return sm[32];
}

__device__ __forceinline__ float tanh_approx(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float situ1(float g, float u) {
  float a = 4.0f * tanh_approx(g * 0.25f);
  float s = __frcp_rn(1.0f + __expf(-g));
  float c = 25.0f * tanh_approx(u * 0.04f);
  return (a * s) * c;
}

// one bf16x8 in, one bf16x8 out; `g` are the gate lanes, `u` the up lanes
__device__ __forceinline__ uint4 situ8(const uint4& gw, const uint4& uw) {
  const __nv_bfloat162* g2 = (const __nv_bfloat162*)&gw;
  const __nv_bfloat162* u2 = (const __nv_bfloat162*)&uw;
  __nv_bfloat162 r2[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 gf = __bfloat1622float2(g2[k]);
    float2 uf = __bfloat1622float2(u2[k]);
    float r[2];
    const float gv[2] = {gf.x, gf.y};
    const float uv[2] = {uf.x, uf.y};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      r[j] = situ1(gv[j], uv[j]);
    }
    r2[k] = __floats2bfloat162_rn(r[0], r[1]);
  }
  return *(const uint4*)r2;
}

extern "C" __global__ void __launch_bounds__(THREADS, 3) kern_k3_situ_rms_bf16(
    const bf16* __restrict__ sp, bf16* __restrict__ act,
    const bf16* __restrict__ lat, const bf16* __restrict__ gam,
    bf16* __restrict__ ln, int n, int B, int h, int L) {
  __shared__ float sm[RMS_R * 33];

  const int tid = threadIdx.x;
  const int q = blockIdx.x / (RATIO + 1);
  const int job = blockIdx.x - (RATIO + 1) * q;

  if (job == 0) {  // ---------------------------------------------------- rms
    const int r0 = RMS_R * q;
    const bool vok = (h & 7) == 0;
    const int hv = h >> 3;
    const bool tok = vok && (tid < hv);
    uint4 rx[RMS_R], gw;
    float s[RMS_R];
    if (tok) {
#pragma unroll
      for (int r = 0; r < RMS_R; ++r) {
        const int row = r0 + r;
        rx[r] = (row < L) ? ((const uint4*)lat)[(long long)row * hv + tid]
                          : make_uint4(0u, 0u, 0u, 0u);
      }
      gw = ((const uint4*)gam)[tid];
    } else {
#pragma unroll
      for (int r = 0; r < RMS_R; ++r) rx[r] = make_uint4(0u, 0u, 0u, 0u);
      gw = make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int r = 0; r < RMS_R; ++r) {
      float f[8];
      float t = 0.0f;
      if (tok) {
        bf16x8_to_f32(rx[r], f);
#pragma unroll
        for (int k = 0; k < 8; ++k) t += f[k] * f[k];
      }
      s[r] = t;
    }
#pragma unroll
    for (int r = 0; r < RMS_R; ++r) {
      const int row = r0 + r;
      float rs = rsqrtf(block_sum(s[r], sm + 33 * r) / (float)h + 1e-5f);
      if (tok && row < L)
        ((uint4*)ln)[(long long)row * hv + tid] = rms_scale8(rx[r], gw, rs);
    }
  } else {  // ------------------------------------------------------------ situ
    const int j0 = RATIO * q + (job - 1);
    if ((n & 7) == 0) {
      const int nv = n >> 3;
#pragma unroll
      for (int r = 0; r < SITU_R; ++r) {
        const int row = SITU_R * j0 + r;
        if (row >= B) break;
        const bf16* __restrict__ grow = sp + (long long)row * 2 * n;
        const bf16* __restrict__ urow = grow + n;
        bf16* __restrict__ orow = act + (long long)row * n;
        const uint4* __restrict__ gv = (const uint4*)grow;
        const uint4* __restrict__ uv = (const uint4*)urow;
        uint4* __restrict__ ov = (uint4*)orow;
        for (int v = tid; v < nv; v += THREADS)
          __stcs(&ov[v], situ8(__ldcs(&gv[v]), __ldcs(&uv[v])));
      }
    } else {
#pragma unroll
      for (int r = 0; r < SITU_R; ++r) {
        const int row = SITU_R * j0 + r;
        if (row >= B) break;
        const bf16* __restrict__ grow = sp + (long long)row * 2 * n;
        const bf16* __restrict__ urow = grow + n;
        bf16* __restrict__ orow = act + (long long)row * n;
        for (int i = tid; i < n; i += THREADS)
          orow[i] = __float2bfloat16(situ1(__bfloat162float(grow[i]),
                                           __bfloat162float(urow[i])));
      }
    }
  }
}
