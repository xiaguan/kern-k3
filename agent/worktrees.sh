#!/usr/bin/env bash
# One worktree per agent on branch agent/<name>: continued where the branch is, or started from main.
# Kernels the branch added are rebuilt from its kernels.toml. Run inside the container, in the repo.
set -euo pipefail
root="${1:-/home/worker/wt}"
git worktree prune
for name in a b c d; do
  [[ -d "$root/$name" ]] && continue
  if git show-ref -q --verify "refs/heads/agent/$name"; then
    git worktree add -q "$root/$name" "agent/$name"
  else
    git worktree add -q -b "agent/$name" "$root/$name" main
  fi
  mkdir -p "$root/$name/build" && cp build/*.cubin "$root/$name/build/"
  (cd "$root/$name" && python3 scripts/build.py >/dev/null && python3 scripts/check.py build | tail -1)
done
git worktree list
