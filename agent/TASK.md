目标：把 K3 TP4、16384 tokens、空 KV cache 的完整 prefill CUDA graph 延迟压到最低。

默认 manifest：manifests/k3-tp4-prefill-16k.json。性能以四个 rank 中最慢的完整 prefill graph p50 为准。
模型、权重、4 卡配置固定。

什么都可以改，只要 runtime 校验得过、judge16k 判 PASS：kernel 源码与 cubin、换 kernel、算子融合、
量化、中间 buffer 的 dtype、op 序列、launch 几何。不允许：改权重、runtime、评测入口、workload、
参考文件、语料、阈值；删减模型计算；针对固定输入硬编码输出。

四个 agent 并行，各管一个分区（见下方"你的分区"），只动自己分区里的 op 和 kernel。
你在自己的 git worktree 和分支 agent/<名> 上工作，commit -s 到自己的分支，不 push、不 rebase、不碰 main。
四个分支最后由人用 agent/compose.py 组到 main 上。

每个候选只差一处。候选由脚本从**当前**默认 manifest 生成（参考 scripts/gen_lat_bf16.py、
scripts/gen_situ_geom.py），脚本进仓库，因为组合到 main 时会用它在新的默认上重生成。
提交前 diff 两份 manifest 的 modules / calls / buffers / ops，只能有这一处不同。

generator 的约定（compose 靠它，不守约定的 commit 进不了 main）：`python3 scripts/gen_<名>.py`
不带参数就能跑，读默认 manifest，写 manifests/<名>.json（或原地覆盖默认），幂等（在已经含这一改动的
manifest 上再跑一遍不出错、不再改），需要的 cubin 已经在 build/ 里。一个 commit 只新增一个 generator；
可以修改自己之前的 generator 让它们成链（compose 先重放改过的，再跑新增的）。

每轮完成一次优化尝试：

1. 先读记录目录下的 STATE.md（当前默认 p50、mix、已采纳、已排除）和已有轮次，不重复试过的路。
   没有 STATE.md 就 bench16k 默认 manifest 建一个。
2. 写下假设，生成独立的候选 JSON 和 cubin。
3. bench16k 候选。收益接近噪声（约 1 ms）就交替复测 baseline 和候选；没有收益则放弃。
4. judge16k CANDIDATE.json REPORT.json：探索期加 --prompts 8，提交前跑全部 48 段。
   只有 PASS 算通过；INCONCLUSIVE 不算；FAIL 记录下来，不改参考、不改规则。
   参考里有两个 producer：默认 manifest，和只把 reduce-scatter 的 bf16 求和顺序换掉的同一个模型。
   两者的分歧就是这个模型自己的数值噪声（KDA 递归把 1 ulp 放大成个别位置的翻转：两者之间
   6.7% 的位置 argmax 不同、KL 中位数约 1e-2），judge 拿它当 band：候选对最近的一个 producer
   翻转率、KL p50 / p99 不超过 band 的 2 倍就 PASS；某个翻转的 margin 超过 band 的 2 倍
   当场 FAIL。所以合法的舍入/求和顺序变化能过，处处变差或真错的核过不了。
   候选若不与默认比特相同（test16k 有 span 不同），commit message 里必须有一行
   `numerics-changing: <哪一处舍入或求和顺序变了，为什么等价>`，judge 的数字一并写上。
5. 性能改善且 judge16k PASS 后，覆盖默认 manifest，更新对应 source、cubin 和 kernels.toml，
   python3 scripts/check.py build，git commit -s。commit 只含 source/ 下的源码、那一个 generator、
   kernels.toml 条目和默认 manifest；README 段落、对照 harness、探针留在记录目录（compose 不收）。
   重编过的模块要起新名字（如 `flash_kda_vllm_d128_v2`）：每次 nvcc 的 sha 都不同，同名会把旧 cubin
   顶掉。未采纳的候选不进仓库，只留在记录目录，工作树回到干净状态；回滚只撤自己的改动。
6. 更新 STATE.md。

没有人在线回答问题。拿不准的按上面的规则自己定；评测入口本身像坏了就记录下来换个靶，别修它。

GPU 是四个 agent 共用的一套。**任何用 GPU 的命令都必须经过 withgpu**：bench16k / judge16k / test16k
自带锁，其他的（探针、ncu、nvcc 之外的任何 GPU 程序）写成 withgpu <命令>。绕过锁会污染别人的测量。
等锁时别空转，去做不用 GPU 的事。

评测入口（都在 PATH 上）：

  bench16k MANIFEST OUT.json                  约 33 s：权重热加载约 18 s + 12 样本约 15 s。
  mix16k REPORT.json [REPORT2.json]           最慢 rank p50、按 op / label 的时间，给两份就打差值。
  judge16k CANDIDATE.json OUT.json [--prompts N]   全 48 段 41 s；--prompts 8 约 3 s + 加载。
  test16k REF.json CANDIDATE.json OUT.json    A/B 逐 span 对照（比特相同 / KL），归因用。
  kern verify MANIFEST                         不用 GPU；runtime 要求 modules 里每个模块都被某个 op 启动。
  withgpu CMD...                               拿 GPU 锁再跑。

bench 报告 JSON：scenarios[i] 每 rank 一个，graph.stats.p50 是 graph 时间（µs），取最慢 rank；
scenarios[i].calls[] 每个 call 有 label / op / in_program.stats.p50。
manifest：modules（cubin 名 + sha256）、buffers、ops（params + impl.launches）、
programs.prefill.calls（label 形如 l<层>.<名>）。
内核：源码在 source/，python3 scripts/build.py 用镜像里的 nvcc（CUDA 13.0，sm_103a）编到 build/，
kernels.toml 记每个模块的 sha256 / source / defines；python3 scripts/check.py build 校验一致。
FlashKDA（source/flash-kda-vllm，CUTLASS 在 /opt/cutlass）：
  CUTLASS_INCLUDE=/opt/cutlass/include bash source/flash-kda-vllm/build.sh build/<新模块名>.cubin
不改源码重编出来的 SASS 和钉住的 cubin 完全一致，但 sha 不同（nvcc 给匿名命名空间加了进程号），
所以先用 test16k 证明重编版比特相同，再改算法；kernels.toml 里用 build = 记这个脚本。
没有源码的模块：moe_fc1 / moe_fc2（trtllm-gen 的 batched GEMM cubin）和 mla_fmha（TRT-LLM fmha cubin），
它们只能整体替换（换成别的 cubin 或自己写的核），改不了内部。dense GEMM 走 cuBLASLt，同理。
看 SASS：cuobjdump -sass build/<模块>.cubin（或 nvdisasm）。sm_103a 的指令表在 isa/sm103a.json
（社区逆向的 Blackwell ISA 库，见 isa/README.md）：每条指令形式的流水线、延迟、吞吐、stall 规则和
编码。判断一个核是被哪条流水线、哪段依赖链卡住，或者核对编译器生成的指令是不是预期的时候用它；
文件 38 MB，用 python 按 base_op 查，别整个读进上下文。
长命令用 bash 工具的后台 job 跑，同时看别的。

参考：/opt/kern-eval/reference.json（初始 manifest）和 /opt/kern-eval/reference.parquet
（默认 manifest 在语料上的分布，在本镜像内录制）。

每轮记录到 记录目录/<日期>-<主题>/（README.md、scripts/、results/）：
假设和改动、bench 前后延迟和收益、judge 结果和报告位置、未采用方案的原因。
commit message 写 bench 和 judge 结果。报告和提交里不要有私人主机名、内部地址或私人绝对路径。

环境异常或测试失败时记录原因，不发布未经验证的收益结论。
每轮完成一次可解释的优化尝试，由 Humanize 提示下一轮。不得启动子 agent。
