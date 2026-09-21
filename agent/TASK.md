目标：把 K3 TP4、16384 tokens、空 KV cache 的完整 prefill CUDA graph 延迟压到最低。

默认 manifest：manifests/k3-tp4-prefill-16k.json。性能以四个 rank 中最慢的完整 prefill graph p50 为准。
模型、权重、4 卡配置固定。

什么都可以改，只要 runtime 校验得过、judge16k 判 PASS：kernel 源码与 cubin、换 kernel、算子融合、
量化、中间 buffer 的 dtype、op 序列、launch 几何。不允许：改权重、runtime、评测入口、workload、
参考文件、语料、阈值；删减模型计算；针对固定输入硬编码输出。

每个候选只差一处。候选由脚本从**当前**默认 manifest 生成（参考 scripts/gen_lat_bf16.py），
提交前 diff 两份 manifest 的 modules / calls / buffers / ops，只能有这一处不同。

每轮完成一次优化尝试：

1. 先读 /home/worker/bench_results/ 下已有记录，不重复试过的路。bench16k 默认 manifest，看 mix，选靶。
2. 写下假设，生成独立的候选 JSON 和 cubin。
3. bench16k 候选。收益接近噪声（约 1 ms）就交替复测 baseline 和候选；没有收益则放弃。
4. judge16k CANDIDATE.json REPORT.json：探索期加 --prompts 8，提交前跑全部 48 段。
   只有 PASS 算通过；INCONCLUSIVE 不算；FAIL 记录下来，不改阈值。
5. 性能改善且 judge16k PASS 后，覆盖默认 manifest，更新对应 source、cubin 和 kernels.toml，
   python3 scripts/check.py build，git commit -s，不 push。未采纳的候选不进仓库，只留在实验目录，
   仓库回到干净状态；回滚只撤自己的改动。

没有人在线回答问题。遇到拿不准的按上面的规则自己定；评测入口本身像坏了就记录下来换个靶，别修它。

评测入口（都在 PATH 上，flock 串行；GPU 只有一套，两个 GPU 任务不要并行）：

  bench16k MANIFEST OUT.json                  约 33 s：权重热加载约 18 s + 12 样本约 15 s。
                                              stdout 末尾有 graph p50 和 op mix。
  judge16k CANDIDATE.json OUT.json [--prompts N]   全 48 段 41 s；--prompts 8 约 3 s + 加载。
  test16k REF.json CANDIDATE.json OUT.json    A/B 逐 span 对照（比特相同 / KL），归因用。
  kern verify MANIFEST                         不用 GPU；runtime 要求 modules 里每个模块都被某个 op 启动。

bench 报告 JSON：scenarios[i] 每 rank 一个，graph.stats.p50 是 graph 时间（µs），取最慢 rank；
scenarios[i].calls[] 每个 call 有 label / op / in_program.stats.p50，按 op 聚合得 mix，按 label 找大头。
manifest：modules（cubin 名 + sha256）、buffers、ops（params + impl.launches）、
programs.prefill.calls（label 形如 l<层>.<名>）。
内核：源码在 source/，python3 scripts/build.py 用镜像里的 nvcc（CUDA 13.0，sm_103a）编到 build/，
kernels.toml 记每个模块的 sha256 / source / defines；python3 scripts/check.py build 校验一致。
长命令用 bash 工具的后台 job 跑，同时看别的。

参考：/opt/kern-eval/reference.json（初始 manifest）和 /opt/kern-eval/reference.parquet
（默认 manifest 在语料上的分布，在本镜像内录制）。已知：l0 的 gate/up GEMM 输出改 bf16 没有收益。

每轮记录到 /home/worker/bench_results/<日期>-<主题>/（README.md、scripts/、results/）：
假设和改动、bench 前后延迟和收益、judge 结果和报告位置、未采用方案的原因。
commit message 写 bench 和 judge 结果。报告和提交里不要有私人主机名、内部地址或私人绝对路径。

只使用分配给本任务的 GPU。环境异常或测试失败时记录原因，不发布未经验证的收益结论。
每轮完成一次可解释的优化尝试，由 Humanize 启动下一轮。不得启动子 agent。
