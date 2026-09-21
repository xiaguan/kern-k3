attention / KDA 段：span_gather、flash_kda（KDA 递推）、kda_out_gate、mla_fmha、o_proj，以及它们之间的
中间 buffer、融合边界、launch 几何、量化。不碰残差 / RMSNorm、通信 op、MoE、shared expert、dense 投影 GEMM（qkvg 除外的部分归 c）。

FlashInfer 里对你有用的：flashinfer/kda_prefill.py（_select_flash_kda_prefill_variant、_PersistentM128Roofline：
按 prefill 形状选 KDA 变体和 schedule 的表）、flashinfer/kda_kernels/、csrc/kda/、jit/cake_kda*.py、
jit/cake_flash_kda_packed_t1.py。我们 flash_kda 的 K2 递归 742 us/层，串行链理论只要几十 us，先对它们的 chunk /
persistent 调度看差在哪。mla_fmha 是 flashinfer-cubin 0.6.18 的 trtllm-gen cubin（kernels.toml upstream），
csrc/cake_fmha/ 是有源码的 FMHA，可以对 schedule。
备选融合已作废：c 实测 cuBLASLt 操作数在 peer 内存上慢 11 倍，o_proj 直写 owner 的四次调用方案不成立。
原文：o_proj 直写 owner。reduce_attn 归约的是 o_proj 的输出，把 o_proj 拆成四次 M=4096 的
cuBLASLt 调用，第 r 次的 D 指向 rank r 的 rs slot（peer 地址），后面 landing 固定顺序加四个 slot，reduce_attn
的 push 就没了。

**K2 用的是 Ampere 的 `mma.sync` m16n8k16（`MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>`，寄存器累加），整个
source/flash-kda-vllm 没有一处 tcgen05。** 这就是 436 条指令的循环体、IPC 0.34、去掉全部 MMA 还剩 34 ms 地板的根源。
FlashInfer 有 tcgen05 版本的 KDA prefill：/opt/flashinfer/csrc/kda/ 里 `cake_flashkda_bf16_persistent_m128.cu`、
`cake_flashkda_bf16_fused_m128_*.cu`、`cake_flashkda_bf16_bt16_prepare*.cu` + `cake_flashkda_bf16_bt16_chain_m64*.cu`
（prepare/chain 正对我们的 K1/K2），以及一批 `cake_flashkda_blackwell_evolution_*_h96_*.cu`（h96 = 96 头，我们 TP4
每 rank 24 头 × d128，正是这个族）。它们都是自包含的生成文件（只 include cuda_bf16.h），状态常驻 TMEM
（`TMEM_TMEM_STATE_OFFSET`），UMMA 单线程异步发射。先读 `flashinfer/kda_prefill.py` 的 `_select_flash_kda_prefill_variant`
搞清每个变体的契约（q/k/v/g/beta 布局、chunk、state 布局、输出），对上我们 span_* buffer 的布局后，用 nvcc 编成
cubin，manifest 里换掉 flash_kda 那个 op（op 序列可以改，K1/K2 可以拆成多个 op）。judge 决定数值。这一刀值
K2 的大半（52 ms/rank 里估计 −25 到 −35），比继续调 vLLM 那份核的 TMA 次数值钱得多。
核对过的三点（人）：(1) 门控别选 `*_unbounded_softplus`，那是另一种门（softplus）；`persistent_m128` /
`piece_persistent_m128` 这一族算的就是我们 K1 的门：`lower_bound * log2e * sigmoid(exp(A_log) * (g + dt_bias))`，
连 `tanh.approx(x*0.5)` 的近似都一样。(2) 直接的 fused m128 一个 (seq, head) 一个 block，我们单序列 24 头只占
24 个 SM；`piece_persistent_m128` 把序列切成 piece 用 `mid_state` 交接，才能铺满 148 个 SM，这是给我们这种
形状准备的。(3) 它们的 initial/final state 是 bf16，我们的 span_state 是 f32：只影响 decode 接续那一处，由
judge 的 decode 位置判。K2 的瓶颈是 mma.sync 的发射，不是 workspace 字节（去掉全部 K1 workspace 拷贝只值
3.3 ms），收益来自 tcgen05 和 chunk 16→32 把串行深度减半。

**TensorRT-LLM 里正对着你这段的核**（记录目录/deps/tensorrt-llm/cpp/tensorrt_llm/kernels）：
* `kdaDecode/kdaDecode.cu`：K3 KDA 的 decode 步，门、beta、状态更新的约定可以拿来核对 FlashInfer 版本；只是 decode。
* `flashMLA/`（sm90 FlashMLA 源码，wgmma 不是 tcgen05）、`mlaChunkedPrefill.cu`、`mlaKernels.cu`：MLA prefill 的
  结构；我们的 mla_fmha 是 trtllm-gen 的 fmha cubin，`contextFusedMultiHeadAttention/` 是它的 dispatcher，
  `trtllmGenKernels/fmha/KernelRunner.h` 能看到有哪些 fmha 变体（cubin 不在树里）。
* `deepseekV4QNormKernel.cu`：q norm 融合。
