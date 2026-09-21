目标：最小化 K3 TP4、16384 tokens、空 KV cache 的完整 prefill CUDA graph 延迟。

当前默认 manifest：manifests/k3-tp4-prefill-16k.json，687 ms（四个 rank 中最慢的 graph p50）。
同一 checkpoint、同一机型上 SGLang 是 678 ms。模型、权重、4 卡配置固定。

允许改的：manifest 里 runtime 能校验的一切——kernel 源码与 cubin、换用别的 kernel、
中间 buffer 的 dtype、融合边界、op 序列、launch 几何。
不允许的：改权重、runtime、评测入口、workload、参考文件、语料、阈值；删减模型计算；
针对固定输入硬编码输出。

每个候选只差一处。候选由脚本从**当前**默认 manifest 生成（参考 scripts/gen_l0_bf16.py），
提交前 diff 两份 manifest 的 modules / calls / buffers / ops，只能有这一处不同。
基于旧默认 manifest 生成的候选会混进别的改动，判定结果就说不清是谁的。

每轮完成一次优化尝试：

1. bench16k manifests/k3-tp4-prefill-16k.json，看 mix，选靶。
   当前：gemm_f32 28.8%、nccl_reducescatter_bf16 14.9%、moe_fc1 12.2%、flash_kda 8.6%、
   gemm_bf16 5.9%、moe_fc2 5.7%。
2. 写下假设，生成独立的候选 JSON 和 cubin。
3. bench16k 候选。收益接近测量噪声就交替复测 baseline 和候选；没有收益则放弃。
4. judge16k CANDIDATE.json REPORT.json：候选对录好的参考分布逐位置判定。
   探索期加 --prompts 8（7 秒），提交前跑全部 48 段（41 秒）。
   只有 PASS 算通过；INCONCLUSIVE 不算；FAIL 即使翻转全是近平局也不算，记录下来，不改阈值。
5. 性能改善且 judge16k PASS 后，覆盖默认 manifest，更新对应 source、cubin 和 kernels.toml，
   python3 scripts/check.py build，提交 commit。

已知事实，不必再试：l0 的 gate/up GEMM 输出改 bf16 是数值 no-op 且无收益——kern_k3_land_situ
本来就先把 f32 partial 落成 bf16 再算激活。同类"输出 f32 但下游先落成 bf16 再用"的 GEMM 改 bf16
输出都是免费的，能省的只有写带宽；真正会改变数值的是喂给 land_add2 / 残差链的 f32 partial。

性能以四个 rank 中最慢的完整 prefill graph p50 为准。保留原始测量结果，不用微基准或 eager
时间代替完整 graph 的收益。baseline 和候选必须在同一节点、相同环境下测量。

评测入口：

  bench16k MANIFEST REPORT.json
  judge16k CANDIDATE.json REPORT.json [--prompts N]
  test16k REFERENCE.json CANDIDATE.json REPORT.json     （A/B 逐 span 归因，可选）

固定的参考：/opt/kern-eval/reference.json（初始默认 manifest）和 /opt/kern-eval/reference.parquet
（它在语料上的分布）。不要修改评测入口、workload、runtime、模型权重或这两个参考。
保留 reference 所需的旧 cubin；新 cubin 的 SHA256 必须与 manifest 和元数据一致。

每轮记录：
- 优化假设和改动。
- bench 前后延迟、收益百分比和命令耗时。
- judge 结果和报告位置。
- 未采用方案的原因。

实验存放在 /home/worker/bench_results/，按日期建立目录，包含 README.md、scripts/、results/。
后续轮次先阅读已有记录。

commit message 必须包含 bench 和 judge 的结果。使用仓库配置的 Git 身份，git commit -s，不自动 push。
报告和提交中不要包含私人主机名、内部地址或私人绝对路径。

只使用分配给本任务的 GPU。资源被其他任务占用时停止 GPU 测试。
环境异常或测试失败时记录原因，不发布未经验证的收益结论。

每轮完成一次可解释的优化尝试，由 Humanize 启动下一轮。不得启动子 agent。
