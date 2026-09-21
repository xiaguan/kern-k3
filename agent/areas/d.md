通信与残差链：nccl_reducescatter / nccl_allgather（合并小 allgather、dtype、宽度、拓扑、和相邻 kernel 的重叠）、
attnres_rms / res_mlp / land_add* 这些残差 + RMSNorm kernel。不碰 attention、MoE 专家、dense 投影 GEMM。

FlashInfer 里对你有用的：csrc/cake_all_gather_matmul/sm103a/（per-(peer, chunk) ready 标志、32 线程邮箱 barrier、
本 rank 的行先算的协议）、flashinfer/comm/trtllm_ar.py、csrc/cake_trtllm_moe_allreduce_fusion/、
csrc/cake_moe_finalize_allreduce_fusion/（lamport 缓冲和 cluster DSMEM 的归约机制）。

你的融合：残差 / RMSNorm 核算完 normed 直接写到四个 rank（gather_normed 的 push 并进 res_in）；rs_push 直接
写 owner 的 slot 已经是这个样子。finalize 直写 owner 归 b（finalize 是他的核），gather_normed 融进 qkvg 归 c
（qkvg 是他的调用）；他们要用你的 peer 地址 / coll_flags 机制，源码在 source/k3_collectives.cu，你不用管。

**多看 SASS。** 你的核现在都在"链路地板"的说法上停住了，但 `cuobjdump -sass build/<模块>.cubin` 把每个核的
循环体和 prologue 数一遍，能看到的不是链路：
1. `kern_k3_rs_pull`：`q = (rank + o) % nr` 里 nr 是运行时参数，编译成 10 组 MUFU.RCP / I2F / F2I / IMAD.HI 的
   软件除法，prologue 402 条（占 40%）；主循环 335 条，8 个 LD.E.128 位点只有 4 个执行，每个 hop 前一条跨 ~250 条
   的条件分支，4 个 peer load 不能背靠背发射，MLP 被切断。修法：`-DNR=4` 或模板特化（kernels.toml 的
   `defines = { HEADS = 24 }` 是先例），4 个 hop 写成无条件直线，q 变成 `(rank + o) & 3`；bf16 解包用
   `SHF.L.U32`（现在是 PRMT + IMAD 两条）。理想循环体约 110 条。舍入链（peers 升序、own 最后、逐跳 bf16）不变。
2. `kern_k3_rs_arrive`：两次 `__threadfence_system()` 各带一次 `CCTL.IVALL`（L1 全量失效），源码注释自己说
   producer 已隔着 kernel 边界，两条都可去；自旋里每轮重算 flags_peer 基址和 slot*8（2 LDC + IMAD.WIDE），
   epoch 比较是 64 位，轮询用 `LD.E.64.STRONG.SYS`。修法：不变量提到循环外、计数器 u32、relaxed 自旋 + 命中后
   一次 acquire 确认（FlashInfer csrc/cake_all_gather_matmul 的 barrier 就是这么写的）。
3. `kern_k3_allgather4_push`：peer 指针在循环里从 global 数组解引用（34 个 LDG.E.64 对 4 个 payload LDG.E.128），
   store 的地址依赖那个 load；每线程只搬 ~1 个向量却付 ~90 条 setup（LDC 92 次重读 28 个参数）。修法：入口把
   p[0..3] 读进标量或 `__grid_constant__`，去 8 路展开。
4. `kern_k3_allgather_push`（gather_normed 用的老核）：`dp[K3_MAX_RANKS]` 掉进 local memory（12 LDL + 9 STL），
   还有一次 u64 除法子程序调用。新核躲开了，老核没回头改。
链路上限按 900 GB/s 单向算（你自己测的远端 store 上限 ~625），不是 1.8 TB/s：reduce_attn 176 MB 的地板是
~200 µs，现在 340，所以第 1 条的上限约 −8 ms/rank，四条合计 −10 到 −15。每改一条：cuobjdump 数指令
（LDL/STL 必须为 0）、`test16k` 比特相同、再 bench。先做第 1 条。

第 1 条的具体做法：不要在源码里写死 4，而是 `template <int NR>` 或 `#ifndef NR` 的 constexpr，kernels.toml 里
`defines = { NR = 4 }` 编成新模块（如 `k3_reducescatter_pull_nr4`），rs_pull / rs_arrive / allgather4_push 一起改：
count、每线程向量数、slot 偏移这些同样是固定形状，能一起变常量的就一起变（`static_assert` 卡住不匹配的形状），
kernel 里 `if (nranks != NR) return;` 保底。目标是 rs_pull 主循环里没有一条 MUFU / I2F / F2I，四个 peer 的
LD.E.128 背靠背发射。
