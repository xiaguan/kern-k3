dense 投影与 shared expert：qkvg、wsh、sh_down、wfu、lat_up、lat_norm、situ、add2 这些 gemm_* 与 landing kernel，
包括换成 fp8 / 量化 GEMM、改输出 dtype、融合 landing。不碰 attention kernel、MoE 专家、通信 op、残差 / RMSNorm。

FlashInfer 里对你有用的：csrc/norm.cu、csrc/rmsnorm_silu.cu、jit/rmsnorm_silu.py（fused_add_rmsnorm_quant 把
add + rms + fp8/nvfp4 输出做在一个核里；按 (C, tokens) 选几何的框架，它的调参表是给 64–1024 宽扫的，7168 走
启发式，抄结构不抄参数）。
你的融合（已被你自己的探针否掉，留档）：gather_normed 融进 qkvg。cuBLASLt 的 A 指向 peer 内存只有 200 TF
（本地 2249），四次调用那条路不通；要做只能自己写 peer-aware 的 tcgen05 GEMM，先追到 cuBLASLt 的速率，是长期项目，
不是下一轮。下一轮按你 STATE 里列的候选走（wfu bf16 这条要 a 的 mla_prep 配合，先在记录里写清楚接口再动）。
原方案：gather_normed 融进 qkvg。把一次 M=16k 的 cuBLASLt 调用拆成四次 M=4096，第 r 次的 A 直接指向 rank r 的
normed（peer 地址，TMA 走 NVLink；d 的 *_peer buffer / coll_flags 机制，source/k3_collectives.cu），前面一个小核
等对方的 ready flag，gather_normed 这个 call 就没了。GEMM 消耗 A 只有 170 GB/s，链路藏得住；先实测 cuBLASLt 读
peer 内存的速率。

**TensorRT-LLM 里正对着你这段的核**（记录目录/deps/tensorrt-llm/cpp/tensorrt_llm/kernels）：
* `fusedGatedRMSNormQuant/`（SiLU 门 + group RMSNorm + 量化一核，Nemotron-H 的 NVFP4 路径）、`groupRmsNormKernels/`、
  `rmsnormKernels.cu`、`fusedLayernormKernels/`：门控 + norm + 落地的融合结构，对着我们 land_situ_rms / lat_norm /
  land_add2 看它怎么排一行的两遍。量化那半是 b 的，结构可以借。
* `dsv3MinLatencyKernels/dsv3FusedAGemm.cu`：小 N GEMM 融合的写法。

**分区调整（2026-09-21）：`moe_fc2` 的 GEMM 核划给你**（fc1 归 b，两人并行）。你的 dense 段按你自己的结论已到墙，
而 MoE 的两个 batched GEMM 是图里最大的可动项（合计 1476 us/层/rank，21%）。fc2 更简单（无 gate 融合、bf16 输出），
CUTLASS 探针由你先写，b 复用。
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
