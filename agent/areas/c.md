dense 投影与 shared expert：qkvg、wsh、sh_down、wfu、lat_up、lat_norm、situ、add2 这些 gemm_* 与 landing kernel，
包括换成 fp8 / 量化 GEMM、改输出 dtype、融合 landing。不碰 attention kernel、MoE 专家、通信 op、残差 / RMSNorm。
