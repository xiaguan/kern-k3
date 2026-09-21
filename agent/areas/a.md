attention / KDA 段：span_gather、flash_kda（KDA 递推）、kda_out_gate、mla_fmha、o_proj，以及它们之间的
中间 buffer、融合边界、launch 几何、量化。不碰残差 / RMSNorm、通信 op、MoE、shared expert、dense 投影 GEMM（qkvg 除外的部分归 c）。
