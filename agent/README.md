# Humanize optimization

Humanize and Claude Code run directly inside one GPU container. Mount the host Claude Code binary read-only and reuse its Claude subscription login. No API key is required. Model: `claude-fable-5-1`, effort: `high`.

Read [TASK.md](TASK.md) before launching. Each round makes one optimization attempt. No automatic push. Prepared with Humanize revision `413d02e44d0cc0514b9f5bd3fcefea156b047a49` (`hmz 0.1.0`) and Claude Code `2.1.276`.

```sh
export BASE_IMAGE=<ubuntu24-cuda13-image-id>
export KERN_BINARY=<kern-binary>
export NCCL_LIB_DIR=<directory-containing-libnccl.so>
export WEIGHTS=<checkpoint-directory>
export TOKENIZER=<tokenizer.json>
export CLAUDE_AUTH_FILE=<existing-claude-credentials.json>
bash agent/prepare.sh
```

The host must have `claude` on PATH. Login is mounted only at container creation, then copied to a private runtime directory so token refresh can write there. It is never added to the image or repository.

The image fixes the initial reference, runtime, CUDA and NCCL. Keep it for the whole campaign. Existing local cubins populate the hash-verified registry cache, avoiding HF downloads. After adding a registry-referenced cubin, run `python3 agent/cache.py` again; local module references use `build/` directly.

```sh
docker exec kern-k3-opt claude auth status
docker exec kern-k3-opt /opt/humanize/bin/python agent/run.py --check
docker exec kern-k3-opt /opt/humanize/bin/python agent/run.py --rounds 1
```

`--check` validates configuration without making a model request. Increase `--rounds` explicitly for more attempts, or use `--rounds 0` to run indefinitely. Unlimited runs retry failed rounds after 30 seconds. Check GPU availability before evaluation; do not run alongside other GPU jobs.

Inside the container:

```sh
bench16k manifests/k3-tp4-prefill-16k.json /home/worker/bench_results/<experiment>/results/baseline.json
test16k manifests/k3-tp4-prefill-16k.json manifests/<candidate>.json /home/worker/bench_results/<experiment>/results/test.json
```

Both commands serialize evaluations and preserve kern exit status. Bench uses 12 samples, seed 24301, 16384 tokens, empty KV cache. Test uses 16k prefill plus one decode step. Compare graph p50 on the slowest rank in this image. This test checks the specified workload; it does not prove correctness for every input.

Historical Codex preparation: [smoke results](../results/humanize-smoke.json). Baseline graph p50 749.790 ms; bench wall time 47.749 s; the existing BF16 candidate test passed in 43.294 s. Runtime is the previously validated `9d1230f-dirty` binary, not a fresh master build.

For a detached unlimited run (set `RUN_DIR` to the mounted experiment directory):

```sh
docker exec -d kern-k3-opt sh -c 'exec flock -n -F /tmp/k3-opt-loop.lock /opt/humanize/bin/python -u agent/run.py --rounds 0 > "$1/results/optimization.log" 2>&1' sh "$RUN_DIR"
```

Follow `results/optimization.log`. Stop the container with `docker stop kern-k3-opt` to stop the loop and its subprocesses.
