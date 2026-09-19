# kern-k3

K3 manifests and kernel sources/build instructions, used with the
[kern runtime](https://github.com/pegainfer-project/kern).

- [`manifests/k3-tp4-prefill-16k.json`](manifests/k3-tp4-prefill-16k.json): the
  93-layer **pruned-75pct** K3 prefill manifest. EP4/TP4, 16,384-token chunk,
  one sequence, maximum context 262,144. This is the 224-expert checkpoint, not
  the full 896-expert checkpoint. Since 2026-09-19 it runs the residual kernels
  on `k3_residual_v2` (see [Residual kernels v2](#residual-kernels-v2-adopted-2026-09-19));
  the originally supplied manifest is the initial reference of the optimization
  runs and is reproducible from it with the v1 residual ops.
- [`kernels.toml`](kernels.toml): every module pinned by a manifest in `manifests/`
  (20 per manifest), their SHA-256 values, local `source` paths and compile-time
  `defines`, or bundled `prebuilt` paths and upstream origin. Paths are relative
  to this repository root.
- [`source/`](source/): the handwritten CUDA files and the vendored FlashKDA sources.
  These produce the handwritten cubins and one FlashKDA cubin. The remaining three
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

Source and recipe are in [`source/flash-kda/`](source/flash-kda/), with upstream
commit and modifications described in its [`PROVENANCE.md`](source/flash-kda/PROVENANCE.md).
The recorded build used CUDA 13.1 and CUTLASS 4.x headers from FlashInfer 0.6.
The exact CUTLASS revision was not recorded, so this recipe alone does not
guarantee reproducing the pinned hash.

```sh
mkdir -p build
CUTLASS_INCLUDE=/path/to/cutlass/include NVCC=/path/to/cuda-13.1/bin/nvcc \
  bash source/flash-kda/build.sh build/flash_kda_d128.cubin
sha256sum build/flash_kda_d128.cubin
```

Expected SHA-256: `34b83d875a418f63d14daf73984c1b8de0d1616250915a7d7e4810f327ce3617`.

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

## Optimization agent

See [Humanize setup](agent/README.md) and the [optimization task](agent/TASK.md). The host Claude Code binary and login are reused; Humanize, compilation and evaluation run in a dedicated GPU container.
