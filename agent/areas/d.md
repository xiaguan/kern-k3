通信与残差链：nccl_reducescatter / nccl_allgather（合并小 allgather、dtype、宽度、拓扑、和相邻 kernel 的重叠）、
attnres_rms / res_mlp / land_add* 这些残差 + RMSNorm kernel。不碰 attention、MoE 专家、dense 投影 GEMM。

FlashInfer 里对你有用的：csrc/cake_all_gather_matmul/sm103a/（per-(peer, chunk) ready 标志、32 线程邮箱 barrier、
本 rank 的行先算的协议）、flashinfer/comm/trtllm_ar.py、csrc/cake_trtllm_moe_allreduce_fusion/、
csrc/cake_moe_finalize_allreduce_fusion/（lamport 缓冲和 cluster DSMEM 的归约机制）。

下一刀的两个候选，都在你的分区里、不改数学：
1. finalize 直接写 owner：现在 l*.finalize 把 [tokens, 3584] 的加权和写到本地，l*.scatter_moe 再整个读一遍
   推给 owner。写一个自己的模块（不改 b 的 k3_moe_prefill）：每个线程把加权和直接写进 owner rank 的 rs
   slot（rs_stage_peer 那套），landing 按固定顺序加四个 slot，舍入链和现在的 rs_sum 一致。省一次 117 MB 的
   写读，约 -0.15 ms/层。
2. gather_normed 融进 qkvg：把一次 M=16k 的 cuBLASLt 调用拆成四次 M=4096，第 r 次的 A 直接指向 rank r 的
   normed（peer 地址，TMA 走 NVLink），前面一个小核等对方的 ready flag。GEMM 消耗 A 只有 170 GB/s，链路
   藏得住；要实测 cuBLASLt 读 peer 内存的速率。这条改 qkvg 的调用，c 不会碰，你做。
