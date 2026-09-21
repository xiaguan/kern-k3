// K4 `k3_mla_prep` v2 -- the same fused MLA prep as two prefill-shaped kernels.
//
// v1 (`source/k3_mla_prep.cu`, module k3_mla_prep+INNER=3072+MLA_FUSED=5184) is
// one launch, grid (tokens, 3), block 512: blockIdx.y == 0 does the row head
// (q_norm, the latent row, the rope half) and the two gate blocks each land
// 1536 of the 3072 gate columns.  At a prefill batch of 16384 rows that is
// 65536 blocks of ~9-17 KB each -- three (gate) or four (head) 4 B values per
// thread -- and the op measures 180 us per call for 573 MB of traffic, i.e.
// 3.2 TB/s, the furthest any kernel in this area sits from the machine's
// streaming rate (~6 TB/s, measured on the same graph by flash_kda's K1).
//
// v2 splits the same work into two kernels with the same entry points the
// manifest calls, each shaped for a large batch:
//
//   * `kern_k3_mla_prep_gate` (grid (ceil(B / 4), 1, 1), block 384): a flat
//     landing copy of the gate band -- thread t owns 8 columns of four
//     consecutive rows, so every thread has eight 16 B loads and four 16 B
//     stores in flight instead of one 16 B pair.
//   * `kern_k3_mla_prep_head` (grid (ceil(B / 4), 1, 1), block 512): four rows
//     per block, four warps per row.  A lane owns one 32-unit group per warp
//     (three q groups and one kv group), sums that unit's four landed squares
//     and reduces inside the warp; the 16 group partials go through 80 bytes of
//     shared memory in group order and every thread adds them in that order,
//     which is exactly what the current kernel's 512-thread row does.  The row
//     is read twice (8.4 KB, L1-resident the second time) instead of being
//     staged through 2112 f32 of shared memory.
//
// Landing points are v1's, element for element: every f32 partial column is
// rounded to bf16 before it is used (`landf`), rms is round-before-scale
// (y = bf16(x * rsqrt(mean(x^2) + 1e-5))), the gamma is applied as a bf16 x
// bf16 -> bf16 product (__hmul2), and rope / the gate band are a single bf16
// landing with no arithmetic.  The only difference is the order in which the
// sums of squares accumulate: v1 sums four elements per thread, then a warp
// tree, then the per-warp partials of the block; v2 sums 48 elements per lane
// and then one warp tree, which moves `q_norm` / the latent row by at most one
// bf16 ulp in the elements near a rounding boundary.
//
//   grid (ceil(B/4), 1, 1)   block 384   kern_k3_mla_prep_gate
//   grid (ceil(B/4), 1, 1)   block 512   kern_k3_mla_prep_head
//
//   nvcc -cubin -arch=sm_103a -DINNER=3072 -DMLA_FUSED=5184 -o out.cubin this
#include <cuda_bf16.h>

#define Q_LORA 1536
#define KV_LORA 512
#define ROPE 64
#define KV_A (KV_LORA + ROPE)
#define HEADC (Q_LORA + KV_LORA + ROPE)   // 2112 head columns of the fused row
#ifndef INNER
#define INNER 3072
#endif
#ifndef MLA_FUSED
#define MLA_FUSED (HEADC + INNER)
#endif
#define EPSV 1e-5f

#define QU (Q_LORA / 4)               // 384 float4 units of q
#define KU ((Q_LORA + KV_LORA) / 4)   // 512 end of the kv units
#define HU (HEADC / 4)                // 528 end of the rope units
#define KU_ (KU - QU)                 // 128 kv units per row
#define HU_ (HU - KU)                 // 16 rope units per row
#define HUQ (QU / 32)                 // 12 q units per lane
#define HUK (KU_ / 32)                // 4 kv units per lane

#define PG_THREADS 384                // 8 gate columns per thread
#define PG_ROWS 4                     // rows per gate block
#define PG_COLS (INNER / PG_THREADS)
#define PH_ROWS 4                     // rows per head block
#define PH_WPR 4                      // warps per row: 3 q groups + 1 kv group each
#define PH_THREADS (PH_ROWS * PH_WPR * 32)

typedef __nv_bfloat16 bf16;

// bf16 pairs as raw 32-bit words so ptxas folds the accesses into LDG/STG.E.64.
__device__ __forceinline__ unsigned pack2(float lo, float hi) {
  __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&p);
}

__device__ __forceinline__ unsigned mul2(unsigned a, unsigned g) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a),
                             *reinterpret_cast<const __nv_bfloat162*>(&g));
  return *reinterpret_cast<unsigned*>(&r);
}

__device__ __forceinline__ float landf(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

// ---- gate band: mla_gate[b, 0 .. INNER) = bf16(P[b, HEADC + 0 .. INNER)) ----
extern "C" __global__ __launch_bounds__(PG_THREADS) void kern_k3_mla_prep_gate(
    const float* __restrict__ partial, bf16* __restrict__ mla_gate, int B) {
  const int c = PG_COLS * threadIdx.x;
  const int r0 = blockIdx.x * PG_ROWS;
  const float* src[PG_ROWS];
  bf16* dst[PG_ROWS];
  float4 v0[PG_ROWS], v1[PG_ROWS];
#pragma unroll
  for (int r = 0; r < PG_ROWS; ++r) {
    const int b = r0 + r;
    const bool ok = b < B;
    src[r] = ok ? partial + (long long)b * MLA_FUSED + HEADC + c : partial;
    dst[r] = ok ? mla_gate + (long long)b * INNER + c : mla_gate;
    v0[r] = make_float4(0.f, 0.f, 0.f, 0.f);
    v1[r] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (ok) {
      v0[r] = *reinterpret_cast<const float4*>(src[r]);
      v1[r] = *reinterpret_cast<const float4*>(src[r] + 4);
    }
  }
#pragma unroll
  for (int r = 0; r < PG_ROWS; ++r) {
    if (r0 + r >= B) continue;
    *reinterpret_cast<uint4*>(dst[r]) =
        make_uint4(pack2(v0[r].x, v0[r].y), pack2(v0[r].z, v0[r].w),
                   pack2(v1[r].x, v1[r].y), pack2(v1[r].z, v1[r].w));
  }
}

// ---- row head: q_norm, the latent row (kv_norm | rope) ----
//
// The reduction reproduces the v1 head bit for bit: the row's 528 float4 units
// are cut into 12 q groups and 4 kv groups of 32 units, a lane owns one unit of
// each of its row's groups and sums that unit's four landed squares in the same
// order v1 does (0 + x0^2 + x1^2 + x2^2 + x3^2), the 32 lanes of the group
// reduce with the same association (a butterfly and v1's shfl_down tree give
// lane 0 the identical value), the group partials are stored in group order and
// every thread then adds them in that order.  PH_WPR warps cover a row (three q
// groups each, one kv group each) and PH_ROWS rows share a block.
extern "C" __global__ __launch_bounds__(PH_THREADS) void kern_k3_mla_prep_head(
    const float* __restrict__ partial, const bf16* __restrict__ gamma_q_a,
    const bf16* __restrict__ gamma_kv_a, const long long* __restrict__ slot_mapping,
    bf16* __restrict__ slab, long long layer_off, long long page_stride,
    bf16* __restrict__ q_norm, int B) {
  __shared__ float red_q[PH_ROWS][12];
  __shared__ float red_k[PH_ROWS][4];
  const int tid = threadIdx.x;
  const int row = tid / (PH_WPR * 32);
  const int sub = (tid / 32) % PH_WPR;
  const int lane = tid & 31;
  const int b = blockIdx.x * PH_ROWS + row;
  const bool live = b < B;
  const float* __restrict__ P = partial + (long long)(live ? b : 0) * MLA_FUSED;

#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int u = 32 * (3 * sub + k) + lane;
    const float4 v = *reinterpret_cast<const float4*>(P + 4 * u);
    const float x0 = landf(v.x), x1 = landf(v.y), x2 = landf(v.z), x3 = landf(v.w);
    float p = 0.0f;
    p += x0 * x0;
    p += x1 * x1;
    p += x2 * x2;
    p += x3 * x3;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) p += __shfl_xor_sync(0xffffffffu, p, off);
    if (lane == 0) red_q[row][3 * sub + k] = p;
  }
  {
    const int u = QU + 32 * sub + lane;
    const float4 v = *reinterpret_cast<const float4*>(P + 4 * u);
    const float x0 = landf(v.x), x1 = landf(v.y), x2 = landf(v.z), x3 = landf(v.w);
    float p = 0.0f;
    p += x0 * x0;
    p += x1 * x1;
    p += x2 * x2;
    p += x3 * x3;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) p += __shfl_xor_sync(0xffffffffu, p, off);
    if (lane == 0) red_k[row][sub] = p;
  }
  __syncthreads();
  float tq = 0.0f, tk = 0.0f;
#pragma unroll
  for (int w = 0; w < 12; ++w) tq += red_q[row][w];
#pragma unroll
  for (int w = 0; w < 4; ++w) tk += red_k[row][w];
  const float scq = rsqrtf(tq * (1.f / Q_LORA) + EPSV);
  const float sck = rsqrtf(tk * (1.f / KV_LORA) + EPSV);
  if (!live) return;

  const long long slot = slot_mapping[b];
  const bool append = slot >= 0;
  // Same slab addressing as v1: a negative slot is "no slot", never used.
  bf16* const rowp = slab + (slot / 64) * page_stride + layer_off + (slot % 64) * KV_A;

#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int u = 32 * (3 * sub + k) + lane;
    const float4 v = *reinterpret_cast<const float4*>(P + 4 * u);
    const uint2 g = *reinterpret_cast<const uint2*>(gamma_q_a + 4 * u);
    *reinterpret_cast<uint2*>(q_norm + (long long)b * Q_LORA + 4 * u) =
        make_uint2(mul2(pack2(landf(v.x) * scq, landf(v.y) * scq), g.x),
                   mul2(pack2(landf(v.z) * scq, landf(v.w) * scq), g.y));
  }
  if (!append) return;
  {
    const int u = QU + 32 * sub + lane;
    const float4 v = *reinterpret_cast<const float4*>(P + 4 * u);
    const uint2 g = *reinterpret_cast<const uint2*>(gamma_kv_a + 4 * (u - QU));
    *reinterpret_cast<uint2*>(rowp + 4 * (u - QU)) =
        make_uint2(mul2(pack2(landf(v.x) * sck, landf(v.y) * sck), g.x),
                   mul2(pack2(landf(v.z) * sck, landf(v.w) * sck), g.y));
  }
  if (sub == 0 && lane < HU_) {   // this row's 16 rope units, a landing only
    const int u = KU + lane;
    const float4 v = *reinterpret_cast<const float4*>(P + 4 * u);
    *reinterpret_cast<uint2*>(rowp + KV_LORA + 4 * lane) =
        make_uint2(pack2(landf(v.x), landf(v.y)), pack2(landf(v.z), landf(v.w)));
  }
}
