目标：最小化 K3 TP4、16384 tokens、空 KV cache 的完整 prefill CUDA graph 延迟。

使用当前默认 manifest：manifests/k3-tp4-prefill-16k.json。
模型、权重和 4 卡配置固定。

每轮完成一次优化尝试：

1. 运行 kern bench，分析具体 call 的耗时，选择优化目标。
2. 优化 kernel 或调用方式，生成独立的候选 JSON 和 cubin。
3. 对候选运行 kern bench。若收益接近测量噪声，交替复测 baseline 和候选；没有收益则放弃。
4. 确认性能收益后，运行 kern test，分别对照本轮 baseline 和固定的初始 reference。
5. 性能改善且测试通过后，覆盖默认 manifest，更新对应 source、cubin 和元数据，提交 commit。

性能以四个 rank 中最慢的完整 prefill graph p50 为准。
保留原始测量结果，不用微基准或 eager 时间代替完整 graph 的收益。
baseline 和候选必须在同一节点、相同环境下测量。

精度以 kern test 的既有阈值为准，不要求 bitwise 相等。
允许改变中间 dtype、归约顺序、线程布局和融合方式。
不得放宽测试阈值、修改参考结果、删减模型计算，或针对固定输入硬编码输出。

评测入口：

  bench16k MANIFEST REPORT.json
  test16k REFERENCE.json CANDIDATE.json REPORT.json

固定的初始 reference：
  /opt/kern-eval/reference.json

不要修改评测入口、workload、runtime、模型权重或初始 reference。
保留 reference 所需的旧 cubin；新 cubin 的 SHA256 必须与 manifest 和元数据一致。
提交前运行：

  python3 scripts/check.py build

每轮记录：
- 优化假设和改动。
- bench 前后延迟、收益百分比和命令耗时。
- test 结果和报告位置。
- 未采用方案的原因。

实验存放在 /home/worker/bench_results/，按日期建立目录，
包含 README.md、scripts/、results/。后续轮次先阅读已有记录。

commit message 必须包含 kern bench 和 kern test 的结果。
使用仓库配置的 Git 身份，git commit -s，不自动 push。
报告和提交中不要包含私人主机名、内部地址或私人绝对路径。

只使用分配给本任务的 GPU。资源被其他任务占用时停止 GPU 测试。
环境异常或测试失败时记录原因，不发布未经验证的收益结论。

每轮完成一次可解释的优化尝试，由 Humanize 启动下一轮。
不得启动子 agent。
