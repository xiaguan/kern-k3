#!/usr/bin/env python3
"""Candidate: the K3 MoE latent RMSNorm runs with 512-thread blocks.

`l*.lat_norm` (`rms`, `kern_k3_rms` in `k3_land`) is launched as one
**1024**-thread block per row.  A 3584-wide bf16 row is 448 sixteen-byte
vectors, so 576 of those 1024 threads are always idle and the row is covered by
14 of 32 warps; with `__launch_bounds__(1024, 1)` the block also caps at one
resident block per SM, so B = 4096 (16k-token TP4 prefill) is 27 serial waves of
a latency-bound block (one load per thread, two `__syncthreads` and a warp
butterfly).  Measured 49.4 us per call in the graph, 58.7 MB moved at 1.19 TB/s
(15% of peak) -- the only op in this area that is at neither the FLOP wall nor
the bandwidth wall.

The kernel is written against `blockDim.x` (vector slot loop bound by
`RMS_REGS * blockDim.x`, `block_sum` over `nwarps` warp partials), so the launch
geometry is free.  At **512** threads every thread still gets at most one of the
448 vectors, so the vector-to-thread assignment and the reduction tree are
unchanged (warp `w` keeps vectors `[32w, 32w+32)`, and lanes `>= nwarps` are
zero, so the butterfly pairs the same partials in the same lanes), while two
blocks are resident per SM.  Standalone on the pinned cubin (`scripts` of the
round record, driver API, 200 launches): 35.76 -> 18.03 us per call,
1642 -> 3256 GB/s.  Block sizes below 448 threads take two vectors per thread,
change the warp partials and move ~10-25 of 14.68M outputs by 1-2 bf16 ulp --
which `judge16k` rejects (a flipped low-margin token diverges the greedy decode),
so 512 is the fastest bit-exact size that also pays off (448 is 3% faster
standalone but 0.05 ms over the whole graph).

Numerics are unchanged: verified over the whole 4096 x 3584 row set, 0 of
14,680,064 elements differ from the 1024-thread launch, and `judge16k` on all 48
prompts reports KL <= 2.5e-12 (48/48 prefill and 768/768 decode argmax agree).

Derived from the current default manifest; the whole difference is the block
dimension of the one `rms` launch (the grid, module, entry, arguments and every
other op are untouched).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ref = ROOT / "manifests/k3-tp4-prefill-16k.json"
out = ROOT / "manifests/k3-tp4-prefill-16k-rms-geom.json"

m = json.loads(ref.read_text())
launch = m["ops"]["rms"]["impl"]["launches"][0]
assert launch["module"] == "k3_land" and launch["entry"] == "kern_k3_rms", launch
assert launch["block"] in ([1024, 1, 1], [512, 1, 1]), launch["block"]
launch["block"] = [512, 1, 1]
out.write_text(json.dumps(m, indent=1) + "\n")
print(out.name)
