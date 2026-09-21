#!/usr/bin/env bash
# One worktree and branch per agent, with the cubins of the main checkout in place. Run inside the container, in the repo.
set -euo pipefail
root="${1:-/home/worker/wt}"
git worktree prune
for name in a b c d; do
  [[ -d "$root/$name" ]] && continue
  git worktree add -q -B "agent/$name" "$root/$name" main
  mkdir -p "$root/$name/build" && cp build/*.cubin "$root/$name/build/"
  (cd "$root/$name" && python3 scripts/check.py build | tail -1)
done
git worktree list
