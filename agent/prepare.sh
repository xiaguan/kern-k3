#!/usr/bin/env bash
set -euo pipefail
: "${BASE_IMAGE:?}" "${KERN_BINARY:?}" "${NCCL_LIB_DIR:?}" "${WEIGHTS:?}" "${TOKENIZER:?}" "${CLAUDE_AUTH_FILE:?}"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPUs have active compute processes; wait for an idle allocation.' >&2
  exit 1
fi
claude_binary="$(readlink -f "$(command -v claude)")"
repo="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$repo/scripts/check.py" "$repo/build"
experiment="${EXPERIMENT_DIR:-$HOME/bench_results/$(date -u +%F)-k3-humanize}"
image="${IMAGE:-kern-k3-eval:local}"
container="${CONTAINER:-kern-k3-opt}"
mkdir -p "$experiment"/{scripts,results}
if [[ ! -f "$experiment/README.md" ]]; then
  printf '%s\n' '# K3 Humanize experiment' '' 'Pinned evaluation environment; reports in results/.' > "$experiment/README.md"
fi
context="$experiment/scripts/image"
mkdir -p "$context/nccl"
cp "$KERN_BINARY" "$context/kern"
cp -a "$NCCL_LIB_DIR"/libnccl.so* "$context/nccl/"
cp "$repo/manifests/k3-tp4-prefill-16k.json" "$context/reference.json"
cp "$repo/agent/"{Dockerfile,workload.toml,bench16k,test16k} "$context/"
docker build --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$image" "$context"
docker run -d --name "$container" --gpus all --ipc=host --ulimit memlock=-1:-1 \
  --user "$(id -u):$(id -g)" --env HOME=/home/worker \
  --mount "type=bind,src=$claude_binary,dst=/usr/local/bin/claude,readonly" \
  --mount "type=bind,src=$CLAUDE_AUTH_FILE,dst=/run/claude-auth.json,readonly" \
  --mount "type=bind,src=$repo,dst=/workspace" \
  --mount "type=bind,src=$WEIGHTS,dst=/weights,readonly" \
  --mount "type=bind,src=$TOKENIZER,dst=/tokenizer.json,readonly" \
  --mount "type=bind,src=$experiment,dst=/home/worker/bench_results/$(basename "$experiment")" \
  "$image"
docker exec "$container" sh -c 'mkdir -p "$HOME/.claude" && cp /run/claude-auth.json "$HOME/.claude/.credentials.json" && chmod 600 "$HOME/.claude/.credentials.json"'
docker exec "$container" git config --global user.name 'JinYan Su'
docker exec "$container" git config --global user.email '751080330@qq.com'
docker image inspect "$image" --format '{{.Id}}' > "$experiment/results/image-id.txt"
docker exec "$container" /opt/kern-eval/kern --version > "$experiment/results/runtime-version.txt"
docker exec "$container" python3 agent/cache.py
