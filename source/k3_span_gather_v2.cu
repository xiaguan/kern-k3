// K9 `k3_span_gather` v2 — the K2 conv + SiLU for a span, same contract as
// k3_span_gather.cu (docs/k3-kernel-abi.md section K9): batch rows
// [*span_at, *span_at + span) are consecutive tokens of one sequence, the conv
// taps of span row i are the sequence's window (the line of the span's first
// row) for i < 3 and the span's own earlier rows after that; the window leaves
// holding the span's last three inputs. The results land in the span's own
// buffers, rows 0..span, together with beta as [HEADS][span] and the f_a flow
// as bf16 [span][128].
//
//   extern "C" __global__ void kern_k3_span_gather(
//       const float* partial,      // [rows, KDA_FUSED]  rows at..at+span read
//       const float* cw,           // [3 stream][4 tap][INNER]
//       void* kda_base, const int* line_index, long long line_bytes,  // line_index[at]'s line
//       const float* wsm_partial,  // [rows, WSM=256]  col h = b_proj, 96.. = f_a
//       bf16* span_q, bf16* span_k, bf16* span_v,   // [span, INNER]
//       bf16* span_beta,           // [HEADS * span]   h*span + i
//       bf16* span_flow,           // [span, 128]
//       const int* span_at,        // [1]  the span's first batch row
//       int span);
//
//   grid  (INNER/(128*K9_VEC), 4, ceil(span/K9_ROWS))   block 128   smem 0
//   blockIdx.y < 3: stream y, K9_VEC consecutive columns per thread, rows
//   K9_ROWS*blockIdx.z .. +K9_ROWS;  blockIdx.y == 3: beta / flow for the
//   same rows (128/K9_ROWS threads per row).
//
// Per stream s, column c, with x_{-3..-1} = win_s[0..2][c] and
// x_i = bf16(partial[i, s*INNER + c]) for 0 <= i < span:
//   y_i        = sum_{t<3} f32(x_{i-3+t}) * cw[s][t][c] + f32(x_i) * cw[s][3][c]
//   sb         = bf16(y_i);  out_s[i, c] = bf16(sb * sigmoid(sb))
//   win_s[t][c] = x_{span-3+t}                                   (t < 3)
// as in v1 (same landing points and window protocol: only the z == 0 block
// reads the old window and writes the new one after its reads).
//
// What changed against v1: v1 was issue-bound, not DRAM-bound (250 us per
// 16k-token call for ~0.9 GB): its SiLU used the IEEE division sequence
// (rcp + 5 fma + range check) and the non-flushing ex2 expansion, ~30 SASS
// instructions per element, and its 8-row blocks re-read 3 tap rows per 8
// (11/8 of the input). v2 computes sigmoid with ex2.approx.ftz and
// rcp.approx.ftz (the fast-math forms; sb is already bf16-rounded and the
// result is rounded to bf16, so the output can differ from v1 by one bf16
// ulp in rare elements), rounds pairs of values through bf16x2, gives each
// block K9_ROWS = 16 rows (19/16 of the input read) and walks them in
// chunks of K9_CHUNK = 4 rows whose loads issue back to back (the three
// taps carry over in registers), with __launch_bounds__(128, 8) so eight
// blocks are resident per SM. Same grid shape as v1 with z = ceil(span/16).
#include <cuda_bf16.h>

#ifndef HEADS
#define HEADS 96
#endif
#ifndef K9_VEC
#define K9_VEC 4
#endif
#ifndef K9_ROWS
#define K9_ROWS 16
#endif
#ifndef K9_FAST_SILU
#define K9_FAST_SILU 1
#endif
#ifndef K9_LOOP
#define K9_LOOP 2   // 2: chunks of K9_CHUNK rows loaded up front; 0: all K9_ROWS rows up front; 1: v1's sequential loop
#endif
#ifndef K9_CHUNK
#define K9_CHUNK 4  // rows per load batch when K9_LOOP == 2
#endif
#ifndef K9_MINB
#define K9_MINB 8   // __launch_bounds__ minimum blocks per SM (register cap, 60 registers)
#endif
#define K9_INNER (HEADS * 128)
#define K9_KDA_FUSED (4 * K9_INNER)
#define K9_REC_BYTES ((long long)HEADS * 128 * 128 * 4)
#define K9_WIN_BYTES ((long long)3 * K9_INNER * 2)
#define K9_WSM 256
#define K9_WSM_FA 96
#define K9_BLOCK 128
#define K9_TPR (K9_BLOCK / K9_ROWS)   // beta / flow threads per row
#define K9_FPT (128 / K9_TPR)         // flow columns per thread

static_assert(K9_VEC == 4 || K9_VEC == 8, "K9_VEC must be 4 or 8");
static_assert(K9_BLOCK % K9_ROWS == 0 && 128 % K9_TPR == 0 && K9_FPT % 4 == 0, "K9_ROWS layout");

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float k9_ex2_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float k9_rcp_ftz(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// sb * sigmoid(sb) for a bf16-rounded sb.
__device__ __forceinline__ float k9_silu(float sb) {
#if K9_FAST_SILU
  const float e = k9_ex2_ftz(sb * -1.4426950408889634f);
  return sb * k9_rcp_ftz(1.0f + e);
#else
  return sb / (1.0f + __expf(-sb));
#endif
}

// Round a pair through bf16 (one F2FP for two values).
__device__ __forceinline__ void k9_round2(float a, float b, float* ra, float* rb) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
  *ra = __low2float(p);
  *rb = __high2float(p);
}

// bf16 landing of a float4 (the K2 input rounding).
__device__ __forceinline__ void k9_land4(float4 p, float* x) {
  k9_round2(p.x, p.y, x, x + 1);
  k9_round2(p.z, p.w, x + 2, x + 3);
}

__device__ __forceinline__ void k9_unpack4(uint2 raw, float* x) {
  x[0] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.x & 0xffffu)));
  x[1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.x >> 16)));
  x[2] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.y & 0xffffu)));
  x[3] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.y >> 16)));
}

// K9_VEC consecutive columns of one partial row (K9_VEC/4 float4 loads).
__device__ __forceinline__ void k9_load_partial(const float* __restrict__ p, float* x) {
#pragma unroll
  for (int q = 0; q < K9_VEC / 4; ++q) k9_land4(__ldg((const float4*)(p + 4 * q)), x + 4 * q);
}

// K9_VEC consecutive columns of one window row.
__device__ __forceinline__ void k9_load_win(const bf16* w, float* x) {
#pragma unroll
  for (int q = 0; q < K9_VEC / 4; ++q) k9_unpack4(*(const uint2*)(w + 4 * q), x + 4 * q);
}

// x_i for i in [-3, span): the window for i < 0, the partial otherwise.
__device__ __forceinline__ void k9_input(const float* __restrict__ col, const bf16* win, int c, int i, float* x) {
  if (i < 0) {
    k9_load_win(win + (size_t)(i + 3) * K9_INNER + c, x);
  } else {
    k9_load_partial(col + (size_t)i * K9_KDA_FUSED, x);
  }
}

__device__ __forceinline__ void k9_pack(const float* x, bf16* o) {
#pragma unroll
  for (int k = 0; k < K9_VEC; k += 2) *(__nv_bfloat162*)(o + k) = __floats2bfloat162_rn(x[k], x[k + 1]);
}

__device__ __forceinline__ void k9_store_bf16(bf16* o, const bf16* v) {
  if (K9_VEC == 8) {
    *(uint4*)o = *(const uint4*)v;
  } else {
    *(uint2*)o = *(const uint2*)v;
  }
}

// One output row: y = conv taps, sb = bf16(y), out = bf16(silu(sb)).
__device__ __forceinline__ void k9_row(const float* t0, const float* t1, const float* t2, const float* x,
                                       const float (*wt)[K9_VEC], bf16* o) {
  float y[K9_VEC];
#pragma unroll
  for (int k = 0; k < K9_VEC; ++k) y[k] = t0[k] * wt[0][k] + t1[k] * wt[1][k] + t2[k] * wt[2][k] + x[k] * wt[3][k];
#pragma unroll
  for (int k = 0; k < K9_VEC; k += 2) {
    float sa, sb;
    k9_round2(y[k], y[k + 1], &sa, &sb);
    *(__nv_bfloat162*)(o + k) = __floats2bfloat162_rn(k9_silu(sa), k9_silu(sb));
  }
}

// The stream body for one block; FIRST selects the z == 0 block whose first
// three taps come from the window (and which writes the new window).
template <bool FIRST>
__device__ __forceinline__ void k9_stream(const float* __restrict__ partial, const float* __restrict__ cw,
                                          bf16* win, bf16* __restrict__ out, int s, int c, int row0, int span) {
  const float* __restrict__ w = cw + (size_t)s * 4 * K9_INNER + c;
  float wt[4][K9_VEC];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
#pragma unroll
    for (int q = 0; q < K9_VEC / 4; ++q) {
      const float4 v = __ldg((const float4*)(w + (size_t)t * K9_INNER + 4 * q));
      wt[t][4 * q] = v.x; wt[t][4 * q + 1] = v.y; wt[t][4 * q + 2] = v.z; wt[t][4 * q + 3] = v.w;
    }
  }
  const float* __restrict__ col = partial + (size_t)s * K9_INNER + c;

#if K9_LOOP == 1
  // Sequential rows with sliding taps (one row load in flight per thread).
  float t0[K9_VEC], t1[K9_VEC], t2[K9_VEC], x[K9_VEC];
  if (FIRST) {
    k9_load_win(win + c, t0);
    k9_load_win(win + K9_INNER + c, t1);
    k9_load_win(win + 2 * K9_INNER + c, t2);
  } else {
    k9_load_partial(col + (size_t)(row0 - 3) * K9_KDA_FUSED, t0);
    k9_load_partial(col + (size_t)(row0 - 2) * K9_KDA_FUSED, t1);
    k9_load_partial(col + (size_t)(row0 - 1) * K9_KDA_FUSED, t2);
  }
  const int rows = min(K9_ROWS, span - row0);
  for (int r = 0; r < rows; ++r) {
    k9_load_partial(col + (size_t)(row0 + r) * K9_KDA_FUSED, x);
    bf16 o[K9_VEC];
    k9_row(t0, t1, t2, x, wt, o);
#pragma unroll
    for (int k = 0; k < K9_VEC; ++k) { t0[k] = t1[k]; t1[k] = t2[k]; t2[k] = x[k]; }
    k9_store_bf16(out + (size_t)(row0 + r) * K9_INNER + c, o);
  }
#elif K9_LOOP == 2
  // K9_ROWS rows in chunks of K9_CHUNK: each chunk's loads issue back to back,
  // the three taps carry over in registers.
  float x[K9_CHUNK + 3][K9_VEC];
  if (FIRST) {
#pragma unroll
    for (int t = 0; t < 3; ++t) k9_load_win(win + (size_t)t * K9_INNER + c, x[t]);
  } else {
#pragma unroll
    for (int t = 0; t < 3; ++t) k9_load_partial(col + (size_t)(row0 - 3 + t) * K9_KDA_FUSED, x[t]);
  }
  for (int r0 = row0; r0 < min(row0 + K9_ROWS, span); r0 += K9_CHUNK) {
#pragma unroll
    for (int r = 0; r < K9_CHUNK; ++r) {
      if (r0 + r < span) k9_load_partial(col + (size_t)(r0 + r) * K9_KDA_FUSED, x[3 + r]);
    }
#pragma unroll
    for (int r = 0; r < K9_CHUNK; ++r) {
      if (r0 + r < span) {
        bf16 o[K9_VEC];
        k9_row(x[r], x[r + 1], x[r + 2], x[r + 3], wt, o);
        k9_store_bf16(out + (size_t)(r0 + r) * K9_INNER + c, o);
      }
    }
#pragma unroll
    for (int t = 0; t < 3; ++t) {
#pragma unroll
      for (int k = 0; k < K9_VEC; ++k) x[t][k] = x[K9_CHUNK + t][k];
    }
  }
#else
  // x[j] = input row row0 - 3 + j, j in [0, K9_ROWS + 3); all loads issue before any use.
  float x[K9_ROWS + 3][K9_VEC];
  if (FIRST) {
#pragma unroll
    for (int t = 0; t < 3; ++t) k9_load_win(win + (size_t)t * K9_INNER + c, x[t]);
  } else {
#pragma unroll
    for (int t = 0; t < 3; ++t) k9_load_partial(col + (size_t)(row0 - 3 + t) * K9_KDA_FUSED, x[t]);
  }
#pragma unroll
  for (int r = 0; r < K9_ROWS; ++r) {
    if (row0 + r < span) k9_load_partial(col + (size_t)(row0 + r) * K9_KDA_FUSED, x[3 + r]);
  }
#pragma unroll
  for (int r = 0; r < K9_ROWS; ++r) {
    if (row0 + r < span) {
      bf16 o[K9_VEC];
      k9_row(x[r], x[r + 1], x[r + 2], x[r + 3], wt, o);
      k9_store_bf16(out + (size_t)(row0 + r) * K9_INNER + c, o);
    }
  }
#endif

  if (FIRST) {
    // New window = x_{span-3..span-1}; every old tap this thread needs it
    // already read above, so the writes race with nothing.
    float nt[3][K9_VEC];
#pragma unroll
    for (int t = 0; t < 3; ++t) k9_input(col, win, c, span - 3 + t, nt[t]);
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      bf16 o[K9_VEC];
      k9_pack(nt[t], o);
      k9_store_bf16(win + (size_t)t * K9_INNER + c, o);
    }
  }
}

extern "C" __global__ __launch_bounds__(K9_BLOCK, K9_MINB) void kern_k3_span_gather(
    const float* __restrict__ partial,
    const float* __restrict__ cw,
    void* __restrict__ kda_base,
    const int* __restrict__ line_index,
    long long line_bytes,
    const float* __restrict__ wsm_partial,
    bf16* __restrict__ span_q, bf16* __restrict__ span_k, bf16* __restrict__ span_v,
    bf16* __restrict__ span_beta,
    bf16* __restrict__ span_flow,
    const int* __restrict__ span_at,
    int span) {
  const int at = span_at[0];
  const int row0 = blockIdx.z * K9_ROWS;
  partial += (size_t)at * K9_KDA_FUSED;
  wsm_partial += (size_t)at * K9_WSM;
  if (blockIdx.y == 3) {
    // K9_TPR threads per row: thread k writes flow[K9_FPT*k..+K9_FPT) and beta
    // for heads k, k + K9_TPR, ...
    const int i = row0 + (threadIdx.x / K9_TPR), k = threadIdx.x % K9_TPR;
    if (i >= span) return;
    const float* row = wsm_partial + (size_t)i * K9_WSM;
    for (int h = k; h < HEADS; h += K9_TPR) span_beta[(size_t)h * span + i] = __float2bfloat16(row[h]);
#pragma unroll
    for (int q = 0; q < K9_FPT / 4; ++q) {
      __nv_bfloat162 f[2];
      f[0] = __floats2bfloat162_rn(row[K9_WSM_FA + k * K9_FPT + q * 4], row[K9_WSM_FA + k * K9_FPT + q * 4 + 1]);
      f[1] = __floats2bfloat162_rn(row[K9_WSM_FA + k * K9_FPT + q * 4 + 2], row[K9_WSM_FA + k * K9_FPT + q * 4 + 3]);
      *(uint2*)(span_flow + (size_t)i * 128 + k * K9_FPT + q * 4) = *(const uint2*)f;
    }
    return;
  }
  const int s = blockIdx.y;
  const int c = (int)(blockIdx.x * K9_BLOCK + threadIdx.x) * K9_VEC;
  bf16* __restrict__ out = s == 0 ? span_q : s == 1 ? span_k : span_v;
  bf16* win = (bf16*)((char*)kda_base + (long long)line_index[at] * line_bytes + K9_REC_BYTES +
                      (long long)s * K9_WIN_BYTES);
  if (blockIdx.z == 0) {
    k9_stream<true>(partial, cw, win, out, s, c, row0, span);
  } else {
    k9_stream<false>(partial, cw, win, out, s, c, row0, span);
  }
}
