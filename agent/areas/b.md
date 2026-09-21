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
