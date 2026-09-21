MoE 专家段：router、lat_down / latent / moe_quant、路由表、fc1 / fc2（trtllm-gen batched GEMM 的配置与换核）、
finalize、scatter_moe 之前的重排。可以改 MoE 内部的量化格式和布局。不碰 nccl 通信 op 本身、残差 / RMSNorm、
attention、shared expert。

FlashInfer 里对你有用的：csrc/cake_grouped_mxfp8_quantize/ 和 jit/cake_grouped_mxfp8_quantize.py（和 moe_quant
同一件事，直接对 SASS）；fused_moe/ 与 flashinfer/artifacts.py 的 TRTLLM_GEN_BMM（trtllm-gen batched GEMM 的
版本；我们的 fc1/fc2 cubin 来自 0.6.18，看有没有更新的 tile / cluster 变体——cubin 要联网下载，镜像里没有，
记下名字和路径由人拉）。fused_moe/prepare.py 是权重 + scale 一起打包的契约，对照 w_shuffle / sf_shuffle。
你的融合：finalize 直写 owner。现在 l*.finalize 把 [tokens, 3584] 的加权和写到本地，l*.scatter_moe 再整个读一遍
推给 owner。让 finalize 的每个线程把加权和直接写进 owner rank 的 rs slot（d 的 rs_stage_peer / coll_flags 机制，
source/k3_collectives.cu），后面的 landing 按固定顺序加四个 slot，舍入链和现在的 rs_sum 保持一致。省一次
117 MB 的写读，约 -0.15 ms/层。scatter_moe 这个 call 因此归你改。

**TensorRT-LLM 里正对着你这段的核**（记录目录/deps/tensorrt-llm/cpp/tensorrt_llm/kernels）：
* `moe/cutlass/`：CUTLASS sm100 TMA warp-specialized 的 grouped GEMM，fp8×fp8 有源码
  （`moe_gemm_kernels_fp8_fp8.cu`、`moe_gemm_template_dispatch_tma_ws.h`，TileN∈{64,128,192,256}，cluster 1x1x1 /
  2x1x1），CUTLASS 在 /opt/cutlass。这是 fc1/fc2 唯一有源码的替代：trtllm-gen 3.0–3.3 PF 对 4.48 PF 上限之间的差
  只能从这里追，但要先量它在我们形状上的实际速率，别先信 tile 表。
* `cuda_graph_grouped_gemm.cu`：problem size 从 device 指针读、图内可重放的 grouped GEMM——你第 4 轮说"没有按
  work item 循环的核所以拿不到 dynamic-batch 的 13.7 ms"，这就是那种核的结构。
* `moe/trtllmGen/routing/RoutingDeepSeek.cu`、`noAuxTcKernels.cu`（里面点名 kimi）：router top-k 的融合写法；
  `dsv3MinLatencyKernels/dsv3RouterGemm.cu`：N=256 的 router GEMM 当带宽核写（kNumTokens 模板），是 decode 形状，
  prefill 下只参考它的归约布局。
* `moe/communication/moeAllReduceFusionKernels.cu`：finalize + 归约融合的另一种结构。

**分区调整（2026-09-21）：`moe_fc1` 的 GEMM 核留给你，`moe_fc2` 划给 c，两人并行。** fc1 要融 siTuGlu 和 fp8 输出
量化，比 fc2 难；c 先写 CUTLASS block-scaled grouped GEMM 探针，你手上的 landing 轮次收尾后直接用它的探针数据起步。
其余（router、路由表、moe_quant、finalize、land）不变。
事实（从默认 manifest 的 ops.moe_fc1 / moe_fc2 和 kernels.toml 读）：两个核都是 trtllm-gen 的 cubin
（`bmm_MxE4m3_MxE2m1MxE4m3_…siTuGlu…` 和 `bmm_Bfloat16_MxE2m1MxE4m3_…`），激活 MxE4m3（fp8，32 元素一个 E8M0 scale），
**权重 MxE2m1（NVFP4，u4 tensormap，每 rank 56 个专家）**，tile 128x128x256、cluster 2x1x1、384 线程、smem 215–228 KB，
fc1 融了 siTuGlu 与输出的 fp8 量化（fCp），fc2 出 bf16；网格按 `ceil(tokens*16/128)+56` 的 dynamic batch 展开，参数是
一个 17472 字节的 pack（tensormap + 路由表指针），用 python 把 pack 的 fields 打出来读。b 量到的：3.0–3.3 PF 对
fp8 上限 4.48 PF，dynamic-batch 的 padding 值 13.7 ms，pdl 不可用。激活格式（moe_quant 的输出）和权重格式（checkpoint）
都不许动。

源码都给了：CUTLASS /opt/cutlass 的 examples/92_blackwell_moe_gemm（`_blockscaled_rcgrouped.cu`、`_fp4_grouped.cu`）、
75_blackwell_grouped_gemm_block_scaled.cu、89_sm103_fp4_ultra_gemm.cu（tcgen05 `kind::mxf8f6f4` 的 block-scaled
builder 就在这些例子里）；TensorRT-LLM 的 `moe/cutlass/`（sm100 TMA warp-specialized grouped GEMM 的整套 dispatch，
但它的 fp8 路径是 per-expert scale 不是 MX）和 `cuda_graph_grouped_gemm.cu`（problem size 从 device 读、按 work item
循环，是吃掉 padding 的那种结构）。

做法（先量后写）：先用 CUTLASS 写独立的 block-scaled grouped GEMM 探针，A = MxE4m3、B = MxE2m1、block-32 scale，
按真实的 per-expert 行数分布（从 moe.blockcount / route_map 读一次真实路由）跑自己那个 op 的形状，报 TF 对照 3.0–3.3 PF；
达不到 3.5 PF 就写下结论停手。达到了再对 ABI：接同一份 route_map / cta_batch / total_padded，输出比特相同或
test16k 可解释。探针是两人共用的，先写出来的放自己的记录目录，另一个人直接读结果不重做。
