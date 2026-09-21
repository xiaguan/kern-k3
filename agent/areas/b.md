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
