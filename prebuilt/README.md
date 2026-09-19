# Prebuilt kernels

These four cubins are the exact artifacts pinned by `../kernels.toml` and the
prefill manifest. `scripts/build.py` copies them into `build/` and checks their
SHA-256 alongside the handwritten kernels. CI checks these bundled bytes too.

- The two `bmm_*` cubins and `trtllm_fmha_ctx_h192_v128.cubin` come from the
  FlashInfer 0.6.18 TRT-LLM bundles identified in `../kernels.toml`.
  Their upstream license is included as `LICENSE.trtllm-gen.txt`.
- `flash_kda_d128.cubin` is the existing build of the vendored FlashKDA sources
  in `../source/flash-kda/`, under the MIT license included in that directory.
