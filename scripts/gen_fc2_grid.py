#!/usr/bin/env python3
"""Candidate: `moe_fc2`'s launch grid is the kernel's own work-list bound, not the global route space.

`l*.fc2` (`moe_fc2`, the trtllm-gen dynamic-batch MXFP8 x MXFP4 batched GEMM) is
launched `grid (28, ceil_div(tokens*16, 128) + 56, 1)` = (28, 2104, 1) for a
16384-token prefill: that is the *global* route-block space (tokens x 16 routed
rows / 128 rows per M tile) plus one block per expert, and every rank launches
all of it although its own work is a quarter of it.  The kernel is
dynamic-batch and does not loop: CTA row y takes work item y of the device-side
list and rows past the item count exit without doing anything, so the rows
between each rank's live item count (~1000) and 2104 are pure dispatch.

`moe_fc2` reads the *same* list as `moe_fc1` (`moe.cta_batch` / `moe.cta_limit`,
both are inputs of either op), so agent b's `test16k --prefill 16384` boundary
for fc1 applies here unchanged: 512 -> 446/446 spans differ, 640 -> 116/116,
768 -> 28/28, 896 -> 2/2 with bit-identical logits, 1024 -> bit-identical
everywhere.  The bound adopted here is b's expression

    grid.Y = 56 + ceil_div(5 * ceil_div(tokens*16, 128), 8)     # 1336 at 16384

a >=30% margin over that boundary, with `tokens` kept in it so shorter prefills
scale down and the decode program (`tokens = 1`) keeps its original 57.  It is
the same expression `moe_fc1` carries on main (commit `ee53b15`, -1.05 ms).

`test16k` cannot repeat the comparison for this op: recording `moe.fc2_out`
(52-80 MB per rank per layer, 92 of them) dies with CUDA_ERROR_OUT_OF_MEMORY on
rank 3 at 16384, 12288 and 8192 alike, so the work-list length is read off the
device instead with `kern run --probe-dir ... --probe-labels
moe.cta_limit,moe.num_non_exiting,moe.total_padded` (record dir
`2026-09-22-r12/`), and `judge16k` (48 prompts) confirms the numerics moved only
inside the band.

**Round 11 (`8c6ff75`) set this grid to the literal 128 and dropped the
`tokens` term.**  That number came from a judge16k boundary (56..72) that
judge16k cannot measure: the judge corpus' prompts are at most ~3k tokens, where
a rank's live item count is a few dozen.  At the graded 16384 tokens a grid of
128 drops most of the tiles -- faster and wrong.  Do not resurrect it.

Derived from the current default manifest; re-running on a default that already
carries the change is a no-op.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ref = ROOT / "manifests/k3-tp4-prefill-16k.json"
out = ROOT / "manifests/k3-tp4-prefill-16k-fc2-grid.json"

# 56 + ceil_div(5 * ceil_div(tokens*16, 128), 8); kept as an expression so the `tokens` term stays.
GRID_Y = {"add": [{"ceil_div": [{"mul": [{"ceil_div": [{"mul": ["tokens", 16]}, 128]}, 5]}, 8]}, 56]}

m = json.loads(ref.read_text())
launch = m["ops"]["moe_fc2"]["impl"]["launches"][0]
grid = list(launch["grid"])
grid[1] = GRID_Y
launch["grid"] = grid
out.write_text(json.dumps(m, indent=1) + "\n")
print(out.name)
