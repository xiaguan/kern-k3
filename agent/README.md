# Agent loop

Humanize drives four DeepSeek Harness agents (`deepseek-v4-flash`, effort `max`) in one GPU
container, through the SDK it ships with (the runtime is bundled; no `dsh` CLI). The only login
is a DeepSeek API key, read from an env file at container creation (`DEEPSEEK_API_KEY`,
optionally `DEEPSEEK_BASE_URL`); it never enters the image or the repository.

Each agent works in its own worktree on branch `agent/<name>` (`agent/worktrees.sh`), owns one
area (`agent/areas/<name>.md`), keeps one conversation across rounds (`agent/flow.py`) and logs it
under its record directory. The GPUs are shared through one lock: `bench16k`, `judge16k` and
`test16k` take it, anything else goes through `withgpu`. Branches are composed onto main by
`agent/compose.py COMMIT...`, which regenerates each candidate from its `scripts/gen_*.py` against
main's current default, re-benches, re-judges and commits what still passes.

```sh
export BASE_IMAGE=<ubuntu24-cuda13-image>
export KERN_BINARY=<kern release binary>
export NCCL_LIB_DIR=<dir with libnccl.so 2.30.7>
export WEIGHTS=<K3 pruned checkpoint>
export TOKENIZER=<tokenizer.json>
export DEEPSEEK_ENV_FILE=<file with DEEPSEEK_API_KEY=...>   # mode 600, outside the repo
agent/prepare.sh
docker exec kern-k3-opt bash agent/worktrees.sh /home/worker/wt
docker exec kern-k3-opt /opt/humanize/bin/python agent/run.py --check
docker exec kern-k3-opt /opt/humanize/bin/python agent/run.py --rounds 1
```

The image fixes the initial reference manifest, the reference distribution (`reference/k3-pruned.parquet`,
recorded inside this image: a recording from another CUDA toolkit fails the default itself), the
runtime, CUDA and NCCL. Keep it for the whole campaign.

```sh
bench16k MANIFEST OUT.json
mix16k REPORT.json [REPORT2.json]
judge16k CANDIDATE.json OUT.json [--prompts 8]
test16k REFERENCE.json CANDIDATE.json OUT.json
withgpu CMD...
```

Bench uses 12 samples, seed 24301, 16384 tokens, empty KV cache; compare graph p50 on the
slowest rank. Judge reads the candidate's distributions over `reference/corpus.json` against the
recorded reference at 816 positions (41 s; `--prompts 8` for 136 in 3 s), the gate a candidate
must PASS. The reference carries two producers, the default and its reduce-scatter in the other
summation order; a candidate passes within twice their mutual disagreement (`reference/README.md`). Test is the A/B span replay against a reference manifest, for attribution.
