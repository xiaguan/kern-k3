MoE 专家段：router、lat_down / latent / moe_quant、路由表、fc1 / fc2（trtllm-gen batched GEMM 的配置与换核）、
finalize、scatter_moe 之前的重排。可以改 MoE 内部的量化格式和布局。不碰 nccl 通信 op 本身、残差 / RMSNorm、
attention、shared expert。
