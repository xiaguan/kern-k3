attention / KDA 段：span_gather、flash_kda（KDA 递推）、kda_out_gate、mla_fmha、o_proj，以及它们之间的
中间 buffer、融合边界、launch 几何、量化。不碰残差 / RMSNorm、通信 op、MoE、shared expert、dense 投影 GEMM（qkvg 除外的部分归 c）。

FlashInfer 里对你有用的：flashinfer/kda_prefill.py（_select_flash_kda_prefill_variant、_PersistentM128Roofline：
按 prefill 形状选 KDA 变体和 schedule 的表）、flashinfer/kda_kernels/、csrc/kda/、jit/cake_kda*.py、
jit/cake_flash_kda_packed_t1.py。我们 flash_kda 的 K2 递归 742 us/层，串行链理论只要几十 us，先对它们的 chunk /
persistent 调度看差在哪。mla_fmha 是 flashinfer-cubin 0.6.18 的 trtllm-gen cubin（kernels.toml upstream），
csrc/cake_fmha/ 是有源码的 FMHA，可以对 schedule。
