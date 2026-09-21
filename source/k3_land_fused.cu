// The shared-expert SiTU landing and the MoE latent RMSNorm, in one launch.
//
//   extern "C" __global__ void kern_k3_situ_rms_bf16(
//       const bf16* sp, bf16* act, const bf16* lat, const bf16* gam,
//       bf16* ln, int n, int B, int h, int L);
//   grid (ceil_div(tokens, 4) / RMS_R, 1, 1)   block 512
//
// One block does RMS_R consecutive rows of both jobs:
//
//   situ   (as kern_k3_land_situ with the operands already landed to bf16, so
//          `<row, i>` of `act`  = bf16( (4*tanh(g/4) * sigmoid(g)) * 25*tanh(u/25) ),
//          g = f32(sp[row*2n + i]), u = f32(sp[row*2n + n + i]))
//   rms    (as kern_k3_rms at block 512, `<row, i>` of `ln` =
//          bf16(f32(lat[row*h+i]) * rsqrt(sum(lat^2)/h + 1e-5)) * gam[i])
//
// Both parts keep the arithmetic of the kernels they replace element for
// element: the same per-thread partial (one 16 B vector of 8 bf16 per thread,
// squared and summed in the same order), the same `block_sum` butterfly over
// the 16 warps (warp w still holds vectors [32w, 32w+32) and the idle warps
// still contribute 0), the same rsqrt/h landing, and the same
// `__hmul2(__floats2bfloat162_rn(x*rs), gamma)` landing.  The situ part is
// elementwise with an f32 activation and one bf16 landing, so any geometry
// computes the same values.  So the whole kernel is bit-identical to
// `kern_k3_land_situ` on bf16 input plus `kern_k3_rms` at block 512.
//
// Why fuse: `rms` (lat_norm) is 28.6 us per call in the graph for 58.7 MB
// (2.06 TB/s) -- it is latency-bound, one 16 B load and two block-wide
// __syncthreads per row, and it is idle bandwidth; `land_situ` is 47.3 us for
// 251.7 MB of f32 in and bf16 out (5.3 TB/s).  Run one after the other the
// pair costs 75.9 us; run in the same block, with the rms loads issued before
// the situ loop and the rms reduction after it, the two streams overlap.
// RMS_R rows per block also give each thread RMS_R 16 B loads in flight
// instead of one, which is what the standalone rms was missing.
#include <cuda_bf16.h>

#define RMS_R 4

typedef unsigned int u32;

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

// --- job layout --------------------------------------------------------------
// One launch, three blocks out of every three doing different jobs, so an SM
// holds both kinds at once and the latency-bound rms stream (2.2 TB/s standalone)
// runs under the DRAM/SFU-bound situ stream (5.6 TB/s) instead of after it:
//   blockIdx.x % 3 == 0  -> rms rows [4q, 4q + 4),    q = blockIdx.x / 3
//   blockIdx.x % 3 != 0  -> situ rows [2j, 2j + 2),   j = 2q + blockIdx.x % 3 - 1
// grid = 3 * ceil_div(tokens, 16) blocks of 512 threads.
extern "C" __global__ void __launch_bounds__(512, 3) kern_k3_situ_rms_bf16(
    const __nv_bfloat16* __restrict__ sp, __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ lat, const __nv_bfloat16* __restrict__ gam,
    __nv_bfloat16* __restrict__ ln, int n, int B, int h, int L) {
  __shared__ float sm[RMS_R * 33];

  const int tid = threadIdx.x;
  const int q = blockIdx.x / 3;
  const int job = blockIdx.x - 3 * q;

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
    const int j0 = 2 * q + (job - 1);
    if ((n & 7) == 0) {
      const int nv = n >> 3;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int row = 2 * j0 + r;
        if (row >= B) break;
        const __nv_bfloat16* grow = sp + (long long)row * 2 * n;
        const __nv_bfloat16* urow = grow + n;
        __nv_bfloat16* orow = act + (long long)row * n;
        for (int v = tid; v < nv; v += blockDim.x) {
          const uint4 g = ((const uint4*)grow)[v];
          const uint4 u = ((const uint4*)urow)[v];
          ((uint4*)orow)[v] = situ8(g, u);
        }
      }
    } else {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int row = 2 * j0 + r;
        if (row >= B) break;
        const __nv_bfloat16* grow = sp + (long long)row * 2 * n;
        const __nv_bfloat16* urow = grow + n;
        __nv_bfloat16* orow = act + (long long)row * n;
        for (int i = tid; i < n; i += blockDim.x)
          orow[i] = __float2bfloat16(situ1(__bfloat162float(grow[i]),
                                           __bfloat162float(urow[i])));
      }
    }
  }
}
