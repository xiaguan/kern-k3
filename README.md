# kern-k3

K3 manifests and kernel sources/build instructions, used with the
[kern runtime](https://github.com/pegainfer-project/kern).

- [`manifests/k3-tp4-prefill-16k.json`](manifests/k3-tp4-prefill-16k.json): the
  93-layer **pruned-75pct** K3 prefill manifest. EP4/TP4, 16,384-token chunk,
  one sequence, maximum context 262,144. This is the 224-expert checkpoint, not
  the full 896-expert checkpoint. Since 2026-09-19 it runs the residual kernels
  on `k3_residual_v2` (see [Residual kernels v2](#residual-kernels-v2-adopted-2026-09-19))
  the KDA output gate on `k3_kda_out_gate_v2` (see
  [KDA output gate v2](#kda-output-gate-v2-adopted-2026-09-19)) and the KDA
  span conv + SiLU on `k3_span_gather_v2` (see
  [Span gather v2](#span-gather-v2-adopted-2026-09-19)). Since 2026-09-21 the
  KDA q/k/v/gate projection lands bf16 directly, so the span conv and the
  output gate read it through `k3_span_gather_v3` / `k3_kda_out_gate_v3` (see
  [KDA partial lands bf16](#kda-partial-lands-bf16-adopted-2026-09-21)). FlashKDA uses the
  vLLM fork with V-split (see [FlashKDA upgrade](#flashkda-upgrade));
  the originally supplied manifest is the initial reference of the optimization
  runs and is reproducible from it with the v1 residual, output-gate and
  span-gather ops.
- [`kernels.toml`](kernels.toml): every module pinned by a manifest in `manifests/`
  (20 per manifest), their SHA-256 values, local `source` paths and compile-time
  `defines`, or bundled `prebuilt` paths and upstream origin. Paths are relative
  to this repository root.
- [`source/`](source/): the handwritten CUDA files and the vendored FlashKDA sources.
  These produce the handwritten cubins and the FlashKDA cubins. The remaining three
  modules are prebuilt TRT-LLM kernels.
- [`prebuilt/`](prebuilt/): the exact FlashKDA and three TRT-LLM cubins, bundled with
  licenses, plus the bundled builds of `k3_situ_bf16` and `k3_residual_v2`.

The manifest itself is the calling example: `ops` contains kernel entry names,
parameter types/order, packed arguments, tensor maps and launch dimensions;
`programs.load` and `programs.prefill` show the calls and buffer bindings.

## Build handwritten kernels

Python 3.11+ and CUDA 13.0 nvcc are required to reproduce the pinned cubins.
The recorded compiler is V13.0.88, build `cuda_13.0.r13.0/compiler.36424714_0`.

```sh
NVCC=/path/to/cuda-13.0/bin/nvcc python3 scripts/build.py
```

The script copies the bundled cubins and builds the remaining handwritten module variants into `build/`, using
`nvcc -cubin -arch=sm_103a` and the defines in `kernels.toml`, then compares each
result with the manifest's SHA-256. All 20 match after building with the recorded compiler.
It exits unsuccessfully if the build fails or any hash differs. A different
compiler or changed source may produce different bytes; update and validate
the artifact and manifest together when intentionally changing a kernel.

If the handwritten kernels are already built, copy only the missing prebuilt artifacts:

```sh
python3 scripts/build.py --prebuilt-only
python3 scripts/check.py build
```

## Rebuild FlashKDA (optional)

The current manifest uses [`source/flash-kda-vllm/`](source/flash-kda-vllm/),
pinned to vLLM FlashKDA `dev@b59532f1`. Its
[provenance](source/flash-kda-vllm/PROVENANCE.md) records the source modifications
and exact CUTLASS revision `5c149f52a436782210263fb2f19b354443a61c6a`.

```sh
CUTLASS_INCLUDE=/path/to/pinned-cutlass/include NVCC=/path/to/cuda-13.0/bin/nvcc \
  bash source/flash-kda-vllm/build.sh build/flash_kda_vllm_d128.cubin
```

The bundled cubin and `kernels.toml` pin the resulting hash. The original
[`source/flash-kda/`](source/flash-kda/) and `flash_kda_d128.cubin` remain for
reference manifests; their old build used CUDA 13.1 and an unrecorded CUTLASS revision.

## Obtain the three TRT-LLM kernels

These three cubins are included in `prebuilt/` and copied by `scripts/build.py`.
They were extracted from upstream bundles; no source build recipe is available here.
`kernels.toml` records their FlashInfer 0.6.18 bundle identities:

| Module | Upstream bundle |
|---|---|
| MoE FC1, `bmm_MxE4m3_…` | `8ec29a98612c3670f9f28825d1ed19f09496073b/batched_gemm-fa419f4-31ee4e5` |
| MoE FC2, `bmm_Bfloat16_…` | Same batched GEMM bundle |
| `trtllm_fmha_ctx_h192_v128` | `2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen` |

The BMM filenames are their full module names plus `.cubin`. The FMHA filename is
`fmhaSm103aKernel_QkvBfloat16OBfloat16HQk192HV128SeparateQkvCausalVarSeqQ256Kv128PersistentContext.cubin`.
To replace the bundled copies from upstream, extract those filenames and verify
their SHA-256 against `kernels.toml`.

The exact bytes for all 20 modules are also available from the private
[HF blob store](https://huggingface.co/Pegainfer/kern-kernels), under
`blobs/<sha256>`. The manifest already points there. Running it requires access
to those blobs (via `HF_TOKEN`) or a populated kern cache, plus model weights.
cuBLAS/cuBLASLt GEMMs and NCCL collectives are runtime dependencies, not cubins
that this repository builds.

## Check

```sh
python3 scripts/check.py          # references, source paths and bundled cubin hashes
python3 scripts/check.py build    # additionally check all 20 local cubins
```

CI runs the first command, including SHA-256 checks of all four bundled cubins;
it does not build CUDA or run inference.
The initial import was checked byte for byte against the supplied manifest,
and all 16 handwritten cubins were rebuilt with matching hashes. Full-model
GPU inference has not been rerun as part of this repository import.

## Provenance

The initial CUDA source snapshot is copied unchanged from kern commit
[`ebffb9d33766f2407b6713d996f65792d267cae3`](https://github.com/pegainfer-project/kern/tree/ebffb9d33766f2407b6713d996f65792d267cae3/tools).
The manifest was originally generated by that implementation's `tools/gen_k3.py`
with `--layers 93 --ranks 4 --tp 4 --chunk 16384 --max-ctx 262144`.
Kernel hashes were cross-checked against the private
[`kern-kernels`](https://github.com/xiaguan/kern-kernels) index.

Imported kern sources retain their [Apache-2.0 license](source/LICENSE).
Vendored FlashKDA retains its [MIT license](source/flash-kda/LICENSE).

## Candidate: layer 0 BF16 gate/up output

[`manifests/k3-tp4-prefill-16k-l0-bf16.json`](manifests/k3-tp4-prefill-16k-l0-bf16.json)
changes `l0.wgu` to the existing BF16-output cuBLASLt op and `l0.situ` to a
BF16-input kernel. `dense_partial` becomes BF16. The original SITU already
rounds its FP32 inputs to BF16 before arithmetic; the candidate preserves its
activation formula and approximations. Different GEMM algorithms can still
produce different rounding, so this is an A/B candidate, not an approved replacement.

The new cubin is bundled; the original manifest is unchanged. Reproduce the JSON
with `python3 scripts/gen_l0_bf16.py`. Rebuild the candidate cubin with CUDA 13.0:

```sh
nvcc -cubin -arch=sm_103a -o build/k3_situ_bf16.cubin source/k3_situ_bf16.cu
```

To test after preparing the baseline cubins:

```sh
python3 scripts/build.py --prebuilt-only
python3 scripts/check.py build
kern test \
  --reference manifests/k3-tp4-prefill-16k.json \
  --manifest manifests/k3-tp4-prefill-16k-l0-bf16.json \
  --kernels build --weights /path/to/kimi-k3-pruned-75pct \
  --gpu 0,1,2,3 --capacity 16448 --prefill 16384 --chunk 16384 \
  --decode-steps 1 --no-sweep --no-graph-step --iters 10 \
  --out test-l0-bf16.json
```

`kern test --diff-only` identifies a single two-call span; the shared output to
compare is `dense_act`. The internal `dense_partial` has different dtypes and
is not a like-for-like comparison. The test also checks end-to-end logits.

### Initial A/B result

[`results/test-l0-bf16.json`](results/test-l0-bf16.json): **PASS**, 8/8 local
comparisons and all eight end-to-end logits rows bit-identical; states and
next-token outputs also match. One seeded workload (`0x5eed`), 16,384-token
prefill plus one continuation token, four GB300 GPUs. This verifies that
workload only, not arbitrary inputs or alternative compiler/runtime versions.

| Measured section | Reference | Candidate |
|---|---:|---:|
| GEMM | 1.991 ms | 1.966 ms |
| SITU | 0.441 ms | 0.577 ms |
| Changed two-call span | 2.432 ms | 2.543 ms |

The candidate is **4.5% slower in the changed span** and remains an experiment.
The new SITU uses scalar element loads, whereas the baseline processes four
FP32 elements per thread; further kernel tuning would be needed before judging
the best achievable BF16 path. `kern test` also reports eager whole-program
809.0 → 819.1 ms; this is not the CUDA graph timing used by the 749.5 ms benchmark.

Runtime: `kern 0.2.3 (9d1230f-dirty, cuda 13.0)`, the historical benchmark binary;
SHA-256 `7e5b1f63545efb343f93633f821112b8aaa9de62dcc90cb3119838afdba22fd1`. The test was not rerun with a fresh master build.

## Residual kernels v2 (adopted 2026-09-19)

The default manifest now runs the K1 residual family (`attnres_rms`,
`attnres_rms_first`, `land_add_attnres_rms_bf16`, `land_add2`) on
[`source/k3_residual_v2.cu`](source/k3_residual_v2.cu), modules `k3_residual_v2`
and `k3_residual_v2+LAND_BF16=1`, with 128-thread blocks instead of the
1024-thread blocks of `k3_residual.cu`. Same entries, ABI, grid, math and
landing points; each thread owns seven 16 B vectors of the row and the two
passes issue the nb candidate loads of a vector back to back, so four rows are
resident per SM and their DRAM latencies overlap. Reductions stay fixed-order
but the order differs from v1, so `normed` can differ by one bf16 ulp in about
0.02% of elements; `prefix2`, `hidden` and `blocks` are bit-identical.
`python3 scripts/gen_residual_v2.py --reference <v1 manifest>` derives it from a
v1 manifest (the one this replaced is the previous commit's file and the
initial reference of the optimization runs).
The v1 modules remain in `kernels.toml` and `build/` for the initial reference.

The four ops carry a `_v2` suffix (`attnres_rms_v2`, ...): `kern test` counts
a call as changed when its op name differs, and the runner keeps every changed
span's outputs on the device, which runs out of memory with all 187 changed
spans of this manifest at once. `gen_residual_v2.py --v1-layers 47-92` makes
an intermediate manifest that keeps the reference's ops on layers 47-92, so
the change is validated in two hops of ~94 spans (reference → intermediate,
intermediate → this manifest), each with the fixed entry and thresholds.

Measured with `kern bench` (12 samples, 16,384 tokens, empty KV cache, four
GB300, graph p50 of the slowest rank) on the same node, alternating:

| manifest | run 1 | run 2 |
|---|---:|---:|
| v1 residual (previous default) | 756.694 ms | 755.987 ms |
| v2 residual, same launches as the committed file | 746.196 ms | 746.206 ms |
| v2 residual, the committed file (`_v2` op names) | 746.150 ms | 745.849 ms (variant with an `attnres_rms` snapshot split, identical launches) |

`attnres_rms` 15.19 → 9.26 ms and `land_add_attnres_rms_bf16` 17.05 → 12.04 ms
summed over the 186 calls. `kern test` in two hops against the initial
reference (byte-identical to the previous default), 16,384-token prefill plus
one decode step, seed `0x5eed`: both **PASS** with the fixed thresholds.
Hop 1 (layers 0-46): 1880 local comparisons, 1422 bit-identical, 0 violations,
end-to-end KL ≤ 1.39e-3 (limit 1e-2), 8/8 argmax agree. Hop 2 (layers 47-92 and
the final norm): 1856 comparisons, 1345 bit-identical, 0 violations, KL ≤
1.61e-3, 8/8 argmax agree. Every differing comparison is `normed`, at most
0.023% of its elements and 0.008 absolute; `next_token` is identical on all
ranks in both hops. The eager per-span time the test reports fell 27-30%.

## KDA output gate v2 (adopted 2026-09-19)

The default manifest now runs the K11 epilogue `kda_out_gate` (one call per
KDA layer, 69 calls) on [`source/k3_kda_out_gate_v2.cu`](source/k3_kda_out_gate_v2.cu),
module `k3_kda_out_gate_v2+HEADS=24`, as op `kda_out_gate_v2` with grid
(tokens, 3) and block 128 instead of grid (tokens, 24) and block 128. Same
entry, ABI, arguments, math and landing points; v1 gave every (row, head) a
128-thread block with one element per thread and a shared-memory reduction,
393k tiny blocks per call. v2 gives each head 16 lanes of a warp with eight
consecutive elements per lane (16 B loads and stores), a shuffle-only
reduction and eight heads per block. The sum of squares stays fixed-order but
its order differs from v1, so `gated` can differ by up to two bf16 ulps in
about 0.0002% of its elements (at most 80 of 50M per call in the test).
`python3 scripts/gen_kda_out_gate_v2.py --reference <manifest with v1>`
derives it (`--v1-layers A-B` makes hop manifests for `kern test`). The v1
module remains in `kernels.toml` and `build/` for the initial reference.

Measured with `kern bench` on the same node, alternating (graph p50 of the
slowest rank, 12 samples, 16,384 tokens, empty KV cache, four GB300):

| manifest | run 1 | run 2 |
|---|---:|---:|
| residual v2 (previous default) | 745.861 ms | 746.247 ms |
| residual v2 + output gate v2 (this file) | 736.160 ms | 737.383 ms |

−9.7 ms / −8.9 ms (−1.30% / −1.19%); `kda_out_gate` 14.31 → 5.10 ms summed
over 69 calls (207 → 74 µs per call).

`kern test` keeps the 805 MB `kda_partial` input of every changed span, so a
direct test of all 69 calls runs out of device memory; the change was
validated in three hops of 24 / 23 / 22 spans (v2 on layers 0-30, 0-61, all)
from the previous default, all **PASS** with the fixed thresholds: 0
violations, every differing local comparison is `gated` (≤ 80 of 50,331,648
elements, ≤ 2 ulp, ≤ 0.0078 absolute), end-to-end KL ≤ 2.38e-3 / 1.35e-3 /
6.47e-4 (limit 1e-2), 8/8 argmax agree, `next_token` identical on all ranks.
The chain from the initial reference (the two residual-v2 hops) was rerun in
the same session and passes as before (KL ≤ 1.39e-3 and 1.61e-3).

## Span gather v2 (adopted 2026-09-19)

The default manifest now runs the K9 span kernel `span_gather` (the K2 causal
conv + SiLU of the q/k/v streams, one call per KDA layer, 69 calls) on
[`source/k3_span_gather_v2.cu`](source/k3_span_gather_v2.cu), module
`k3_span_gather_v2+HEADS=24`, as op `span_gather_v2` with grid
(6, 4, ceil(tokens/16)) and block 128 instead of (6, 4, ceil(tokens/8)) and
block 128. Same entry, ABI, arguments, per-element expression, landing points
and window protocol. v1 was issue-bound rather than DRAM-bound (250 µs per
call for ~0.9 GB): its SiLU used the IEEE division sequence and the
non-flushing `__expf` expansion (~30 SASS instructions per element), and its
8-row blocks re-read three tap rows per eight. v2 computes the sigmoid with
`ex2.approx.ftz` / `rcp.approx.ftz`, rounds pairs of values through bf16x2,
gives each block 16 rows loaded in chunks of four (the taps carried in
registers) and is capped at 60 registers by `__launch_bounds__(128, 8)` so
eight blocks are resident per SM. `sb` is bf16-rounded before the SiLU and
the result is rounded to bf16, so the output can differ from v1 by one bf16
ulp in rare elements; none were observed on random data or in the test.
`python3 scripts/gen_span_gather_v2.py --reference <manifest with v1>`
derives it (`--v1-layers A-B` makes hop manifests for `kern test`). The v1
module remains in `kernels.toml` and `build/` for the initial reference.

Measured with `kern bench` on the same node, alternating (graph p50 of the
slowest rank, 12 samples, 16,384 tokens, empty KV cache, four GB300):

| manifest | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| residual v2 + output gate v2 (previous default) | 736.458 ms | 736.576 ms | 736.746 ms |
| + span gather v2 (this file) | | | 730.484 ms |

−6.26 ms (−0.85%) against the same-session baseline run 3 (−6.09 ms against
run 2); `span_gather` 17.22 → 11.03 ms summed over 69 calls (250 → 160 µs
per call). Earlier layouts of the same kernel measured 198 / 183 / 166 µs per
call in the graph (8 rows loaded up front, 32-row sequential loop, 32 rows in
chunks of four); standalone timings did not predict the in-graph ranking.

`kern test` (16k prefill + one decode step, seed 0x5eed) records the 805 MB
`kda_partial` input of every changed span (~4.6 GB per span), so the change
was validated in three hops of 23 spans with the same entry and thresholds
(v2 on layers 0-29, then 0-60, then all): every hop is **PASS** with all
1012 local comparisons bit-identical, logits bit-identical on 8/8 rows and
next_token / KDA / KV states bit-identical on all ranks. The chain from the
initial reference (residual v2 on layers 0-46 → round-1 default → output gate
v2 on layers 0-30 → 0-61 → previous default) was rerun in the same session
and passes as before (KL ≤ 1.39e-3, 1.61e-3, 2.38e-3, 1.35e-3, 6.47e-4).

## KDA partial lands bf16 (adopted 2026-09-21)

`l*.qkvg` (69 calls, the KDA q/k/v/gate projection) ran `gemm_f32`, cuBLAS
`cublas_bf16_tn_f32` writing the f32 `kda_partial` [tokens, 12288] - 805 MB per
call. Both of its readers rounded every element through bf16 first anyway: the
span conv through `k9_land4` (`x_i = f32(bf16(partial[i, c]))` in
`source/k3_span_gather_v2.cu`) and the output gate through
`gg = f32(bf16(gate_partial[...]))` before the sigmoid. The GEMM epilogue now
lands bf16 into the same buffer, which removes half of that buffer's write and
of both reads with the same rounding, and the call sites keep their arguments:

- `buffers.kda_partial.dtype`: f32 -> bf16;
- the 69 `l*.qkvg` calls: `gemm_f32` -> `gemm_bf16` (same output buffer);
- `span_gather_v2` reads `buffer<bf16>` and runs
  [`source/k3_span_gather_v3.cu`](source/k3_span_gather_v3.cu), module
  `k3_span_gather_v3+HEADS=24` - v2 with the `float4` partial load replaced by a
  bf16 unpack (`k9_unpack4`), same entry, geometry, args, math and landing
  points;
- `kda_out_gate_v2` reads `buffer<bf16>` and runs
  [`source/k3_kda_out_gate_v3.cu`](source/k3_kda_out_gate_v3.cu), module
  `k3_kda_out_gate_v3+HEADS=24` - v2 with the f32 gate band load replaced by the
  bf16 vector it produced.

Measured on the TP4 16k prefill graph (slowest rank, 12 samples, seed 24301),
alternating with the previous default: 679.865 / 679.745 / 679.849 ms (mean
679.820) to 675.387 / 675.631 / 675.622 ms (mean 675.547), i.e. **-4.27 ms
(-0.63%)**, consistent in all three pairs. Instrumented per-label sums over the
four ranks: `span_gather` 44.13 -> 34.50 ms, `qkvg` 402.54 -> 397.41 ms,
`span_out_gate` 20.37 -> 19.44 ms; every other op is unchanged within its
run-to-run spread. The judge over the recorded reference passes on all 816
positions with the reference's own float precision (prefill KL p50 1.2e-15, max
4.2e-14; decode p50 2.0e-15, max 2.5e-12; 48/48 and 768/768 argmax agree) and
the same per-position detail as the previous default, because the change only
moves an existing bf16 round-to-nearest from the readers into the GEMM epilogue.
`python3 scripts/gen_kda_partial_bf16.py` derives the change from the previous
default; the v2 modules remain in `kernels.toml` and `build/` for that
reference. A span-by-span A/B (`kern test` against the initial reference, the
v2 route) cannot complete for this change: saving the 805 MB input of the
changed spans runs the device out of memory, as it did for the v2 span gather,
so the evidence is the op attribution plus the judge.

## SITU launch geometry (adopted 2026-09-21)

The shared expert's SITU (`land_situ_n6144`, `kern_k3_land_situ` in
`k3_land`) now runs with 128-thread blocks instead of 1024. The op is
elementwise (`out[i] = situ(gate[i], up[i])`, `source/k3_land.cu`), so the
thread mapping does not enter any value it computes; the 1024-thread launch was
inherited from the supplied manifest, where a block covered a decode-sized row
and each thread computed at most one element. With 128-thread blocks each
thread computes eight elements of the row, the per-thread register budget buys
eight times as many resident blocks per SM, and the same grid (tokens/4, 6)
covers the row. Measured on the TP4 16k prefill graph (slowest rank, 12
samples, seed 24301): 682.713 ms (three runs of the previous default,
683.060/682.393/682.686) to 678.891 ms (two runs, 678.958/678.824), i.e.
-3.82 ms / -0.56%, entirely in the 92 `shared_situ` calls (8.29 -> 4.36 ms per
rank of instrumented time). The judge over the recorded reference passes on all
816 positions with the same per-position detail as the previous default, and
the output is bit-identical because no cross-thread reduction is involved.
`python3 scripts/gen_situ_geom.py` derives the change from the previous default;
it is one field, `ops.land_situ_n6144.impl.launches[0].block`. The `rms` op of
the same module has the same problem (1024 threads for a 3584-element row, 5.6x
off its bandwidth roofline) but a different fix, because its block-wide
reduction makes the thread count part of the result.

## Optimization agent

See [Humanize setup](agent/README.md) and the [optimization task](agent/TASK.md). The host Claude Code binary and login are reused; Humanize, compilation and evaluation run in a dedicated GPU container.

## FlashKDA upgrade

The default manifest uses vLLM FlashKDA `dev@b59532f1` with K2 V-split.
The upstream kernels retain recurrent-state fragments in registers and transfer
workspace with bulk copies. V-split gives each head two independent 64-row
value slices. At TP4 this doubles K2's grid from 24 to 48 CTAs.

Four-GPU 16k prefill graph p50, slowest rank, 12 samples per run:

| Version | First run | Repeat |
|---|---:|---:|
| Previous default (`81c451f`) | 730.183 ms | 730.690 ms |
| vLLM FlashKDA, no V-split | 705.938 ms | — |
| vLLM FlashKDA, V-split (default) | 689.924 ms | 689.984 ms |

The paired final runs improve by 40.706 ms (5.57%). Instrumented KDA call
p50 sums fall from approximately 103 ms to 60 ms across 69 calls.
See [`results/flashkda-vllm.json`](results/flashkda-vllm.json).

`kern test` directly compares the final candidate to both the previous default
and the fixed initial reference, at 16k prefill plus one decode step:

- Previous default: PASS, compared outputs/states and logits bit-identical.
- Initial reference: PASS, max logit KL 0.00175971 (limit 0.01), 8/8 argmax agree.

Per-call recording exceeds GPU memory because it retains every layer's KDA
workspace. `scripts/whole_graph_test.py` creates a semantically equivalent test
manifest: it aliases operations to form one full-program comparison span and
renames internal workspace buffers so they are not retained for cross-version
scratch comparisons. It checks that reversing the names recovers the original
manifest. Kernels, call order, arguments, buffer shapes, input/output identities,
persistent states and tolerances are unchanged. This is a direct end-to-end
comparison, not a chain through intermediate manifests; it does not establish
per-layer scratch equality.

```sh
python3 scripts/whole_graph_test.py manifests/k3-tp4-prefill-16k.json /tmp/candidate-test.json
test16k /path/to/reference.json /tmp/candidate-test.json /tmp/test.json
```

For a new candidate derived from a manifest using the original FlashKDA ABI:

```sh
python3 scripts/gen_flash_kda_vllm.py --reference /path/to/baseline.json \
  --vsplit --out /tmp/candidate.json
```

One initial V-split bench exited 139 before timing. Its launch ABI was checked
against a captured upstream launch; two later bench runs and both direct tests
completed. The failed attempt is retained in the experiment records.
