# Prebuilt kernels

These cubins are the exact artifacts pinned by `../kernels.toml` and the
prefill manifest. `scripts/build.py` copies them into `build/` and checks their
SHA-256 alongside the handwritten kernels. CI checks these bundled bytes too.

- The two `bmm_*` cubins and `trtllm_fmha_ctx_h192_v128.cubin` come from the
  FlashInfer 0.6.18 TRT-LLM bundles identified in `../kernels.toml`.
  Their upstream license is included as `LICENSE.trtllm-gen.txt`.
- `flash_kda_d128.cubin` is the existing build of the vendored FlashKDA sources
  in `../source/flash-kda/`, under the MIT license included in that directory.

`k3_situ_bf16.cubin` is the candidate BF16-input SITU kernel, built from
`../source/k3_situ_bf16.cu` with CUDA 13.0, under `../source/LICENSE`.

`k3_residual_v2.cubin` and `k3_residual_v2+LAND_BF16=1.cubin` are the residual
kernels used by the default prefill manifest since 2026-09-19, built from
`../source/k3_residual_v2.cu` with CUDA 13.0 (`nvcc -cubin -arch=sm_103a`,
the second with `-DLAND_BF16=1`), under `../source/LICENSE`.

`k3_kda_out_gate_v2+HEADS=24.cubin` is the KDA output-gate kernel used by the
default prefill manifest since 2026-09-19 (round 2), built from
`../source/k3_kda_out_gate_v2.cu` with CUDA 13.0 (`nvcc -cubin -arch=sm_103a
-DHEADS=24`), under `../source/LICENSE`.

`k3_span_gather_v2+HEADS=24.cubin` is the KDA span conv + SiLU kernel used by
the default prefill manifest since 2026-09-19 (round 3), built from
`../source/k3_span_gather_v2.cu` with CUDA 13.0 (`nvcc -cubin -arch=sm_103a
-DHEADS=24`), under `../source/LICENSE`.

`flash_kda_vllm_d128.cubin` is built from `../source/flash-kda-vllm/`,
vLLM FlashKDA `dev@b59532f1`, with CUDA 13.0 and CUTLASS
`5c149f52a436782210263fb2f19b354443a61c6a`. It contains K1 plus regular and
V-split K2, fixed-length D=128 with FP32 state. The default manifest selects
V-split. Source and MIT license are bundled in that directory.
