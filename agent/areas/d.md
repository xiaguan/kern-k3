通信与残差链：nccl_reducescatter / nccl_allgather（合并小 allgather、dtype、宽度、拓扑、和相邻 kernel 的重叠）、
attnres_rms / res_mlp / land_add* 这些残差 + RMSNorm kernel。不碰 attention、MoE 专家、dense 投影 GEMM。
