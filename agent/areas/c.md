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

**分区调整（2026-09-21）：fc1 / fc2 的 GEMM 核从 b 划给你。** 你自己的判断是 dense 段已经榨干（cuBLASLt 的墙、
落地核都在噪声带内），而图里最大的一块可动项在 MoE 的两个 batched GEMM：`moe_fc1` / `moe_fc2` 合计约 1476 us/层/rank，
占图 21%，b 量到它们跑在 3.0–3.3 PF，对 fp8 上限 4.48 PF 还差 25–30%，其中 dynamic-batch 的 padding 就值 13.7 ms。
b 继续管 router / 路由表 / moe_quant / finalize / land，只把这两个 op 的 GEMM 本体交给你；moe_quant 出的激活格式和
权重格式都不许动（权重是 checkpoint 的一部分）。

事实（从默认 manifest 的 ops.moe_fc1 / moe_fc2 和 kernels.toml 读）：两个核都是 trtllm-gen 的 cubin
（`bmm_MxE4m3_MxE2m1MxE4m3_…siTuGlu…` 和 `bmm_Bfloat16_MxE2m1MxE4m3_…`），激活 MxE4m3（fp8，32 元素一个 E8M0 scale），
**权重 MxE2m1（NVFP4，u4 tensormap，每 rank 56 个专家）**，tile 128x128x256、cluster 2x1x1、384 线程、smem 215–228 KB，
fc1 融了 siTuGlu 与 fCp，fc2 出 bf16；网格按 `ceil(tokens*16/128)+56` 的 dynamic batch 展开，参数是一个 17472 字节的
pack（tensormap + 路由表指针）。ABI 全在 manifest 里，用 python 把 pack 的 fields 打出来读。

做法（先量后写，不许先整合）：
1. 探针：用 CUTLASS（/opt/cutlass，examples/92_blackwell_moe_gemm、75_blackwell_grouped_gemm 的 block-scaled 变体、
   89_sm103_fp4_ultra_gemm）写一个独立的 grouped GEMM 探针，A = MxE4m3、B = MxE2m1、block-32 scale，
   tcgen05 `kind::mxf8f6f4`，按我们真实的 per-expert 行数分布（从 moe.blockcount / route_map 读一次真实路由）
   跑 fc2 的形状，报 TF；对照 trtllm-gen 的 3.0–3.3 PF。达不到 3.5 PF 以上就写下结论停手。
2. 达到了，再对 ABI：接同一份 route_map / cta_batch / total_padded，输出布局比特相同或 test16k 可解释，
   fc2 先于 fc1（fc1 还要融 siTuGlu 和 fp8 输出量化）。
3. padding 那 13.7 ms：持久核按 work item 循环（TensorRT-LLM `cuda_graph_grouped_gemm.cu`，problem size
   从 device 读），或者 tile 沿 M 变小；`moe.cta_batch` / `moe.cta_limit` 的含义看 b 的 route_tables 核。
TensorRT-LLM 的 `moe/cutlass/`（sm100 TMA warp-specialized，`moe_gemm_template_dispatch_tma_ws.h`）是现成的参考
结构，但它的 fp8 路径是 per-expert scale，不是 MX，scale 处理要看 CUTLASS 的 block-scaled builder。
