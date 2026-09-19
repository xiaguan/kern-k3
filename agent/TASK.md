目标：降低 K3 TP4、16384 tokens、空 KV cache 的完整 prefill CUDA graph 延迟。

模型固定为当前 93 层、224 experts 的 checkpoint，使用 4 张 GPU。默认 manifest 是 manifests/k3-tp4-prefill-16k.json。不要把它当成全量 896-expert 模型。

每轮完成一次有依据的优化尝试：
1. 阅读已有结果，先运行 bench16k 当前默认 manifest，按具体 call 的耗时选择 ROI。
2. 修改 kernel source 或调用方式，生成独立 candidate JSON 和 cubin。保留旧 cubin，使 reference 仍可运行。允许改变中间 dtype，但必须通过原有精度检查。
3. bench16k candidate；比较四个 rank 中最大的 graph p50，保留 p10/p90 和原始结果。收益接近噪声时交替复测 baseline/candidate。变慢就放弃，无需跑 test。
4. 有可重复收益后，用 test16k 同时对照本轮 baseline 和 /opt/kern-eval/reference.json。不能只与上一轮比较而累积精度漂移。两次 test 均需正常退出且报告通过。
5. 只有性能改善和测试通过，才覆盖默认 manifest，更新 source、cubins、kernels.toml，运行 python3 scripts/check.py build，提交 git commit -s。commit message 写明 bench 前后毫秒、收益百分比、test 结果及报告相对路径。没有收益则保持默认 manifest 不变。

两个评测入口：
  bench16k MANIFEST REPORT.json
  test16k REFERENCE.json CANDIDATE.json REPORT.json

编译器和 runtime 已在镜像中固定。修改 source 后需要编译新 cubin，并同步真实 SHA256 到 manifest/元数据；build.py 的 hash 不匹配是错误，不能忽略。新模块需要 source ground truth 或明确上游来源，以及调用参数示例（manifest 中的实际 call 即可）。

不要修改评测入口、workload、runtime、模型权重、reference、测试阈值，不要删除算子或针对固定输入硬编码结果。不能用 eager 时间代替 graph 时间。遇到其他任务占 GPU、测试失败或环境异常，先记录原因，不发布收益结论。不得启动子 agent。

实验放在 /home/worker/bench_results/ 下按日期建立目录，包含 README.md、scripts/、results/。每轮记录假设、改动、bench/test 的 wall time、结果和下一步，下一轮先读这些记录。报告或提交中不得出现私人主机名、内部地址或包含用户名的绝对路径。

git 身份使用 JinYan Su <751080330@qq.com>，提交带同身份 Signed-off-by。每轮只做一个可解释的尝试并报告结果；由 Humanize 决定下一轮。不要自动 push。
