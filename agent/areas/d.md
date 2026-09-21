通信与残差链：nccl_reducescatter / nccl_allgather（合并小 allgather、dtype、宽度、拓扑、和相邻 kernel 的重叠）、
attnres_rms / res_mlp / land_add* 这些残差 + RMSNorm kernel。不碰 attention、MoE 专家、dense 投影 GEMM。

FlashInfer 里对你有用的：csrc/cake_all_gather_matmul/sm103a/（per-(peer, chunk) ready 标志、32 线程邮箱 barrier、
本 rank 的行先算的协议）、flashinfer/comm/trtllm_ar.py、csrc/cake_trtllm_moe_allreduce_fusion/、
csrc/cake_moe_finalize_allreduce_fusion/（lamport 缓冲和 cluster DSMEM 的归约机制）。

你的融合：残差 / RMSNorm 核算完 normed 直接写到四个 rank（gather_normed 的 push 并进 res_in）；rs_push 直接
写 owner 的 slot 已经是这个样子。finalize 直写 owner 归 b（finalize 是他的核），gather_normed 融进 qkvg 归 c
（qkvg 是他的调用）；他们要用你的 peer 地址 / coll_flags 机制，源码在 source/k3_collectives.cu，你不用管。
