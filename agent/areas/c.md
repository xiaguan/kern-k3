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
