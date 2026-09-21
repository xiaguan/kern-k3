#!/usr/bin/env python3
"""Candidate: `moe_finalize` compacts a token's live rows and walks two tokens per block.

`l*.finalize` (92 calls) combines the top-16 expert rows of every token:

    out[t] = sum over the local k of wts[t][k] * fc2[exp2perm[16 t + k]]

On a rank 12 of the 16 slots are another rank's (56 of 224 experts), so the
kernel's column loop walks 16 predicated slots per output vector and loads
~4 rows.  The module `k3_moe_prefill_v2` keeps `kern_k3_moe_finalize`'s
arithmetic exactly -- one f32 multiply-add per element, rows visited in
ascending k, one bf16 rounding per output element -- but compacts the live
(p, w) list of two tokens into shared memory once per block and then walks
only those rows, two 16-element (32 B) vectors per item.  Verified
bit-identical to the current kernel on a 16384x3584 synthetic stand-in
(58.7M outputs, 0 mismatches) and against a host f32 reference.

Measured standalone: 110.2 -> 101.9 us per call (-7.5%) on the same traffic
(470 MB read + 117 MB write, 5.34 -> 5.78 TB/s).

One change, two consequences of it: the module the MoE ops launch, and this
op's launch geometry (grid ceil(tokens/2) x block 128 instead of tokens x 256).

Derived from the current default manifest (pass another path as the first
argument if the default has moved).  The second argument is the output path.
"""
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "manifests/k3-tp4-prefill-16k.json")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "manifests/k3-tp4-prefill-16k-finalize-geom.json")

OLD = "k3_moe_prefill"
NEW = "k3_moe_prefill_v2"
SHA = "dc3cc8805dbbb0c198d9d06001923f542cd9058875ca745e8a767541985e4531"

m = json.loads(SRC.read_text())
kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
assert kernels.get(NEW, {}).get("sha256", SHA) == SHA, NEW
if NEW in m["modules"]:
    raise SystemExit(f"{SRC.name} already runs {NEW}")
assert m["modules"][OLD]["sha256"] == kernels[OLD]["sha256"]

# 1. the MoE ops launch the v2 module
m["modules"][NEW] = {"source": f"{NEW}.cubin", "sha256": SHA}
del m["modules"][OLD]
rep = 0
for op in m["ops"].values():
    for launch in op["impl"].get("launches", []):
        if launch.get("module") == OLD:
            launch["module"] = NEW
            rep += 1
assert rep == 7, rep

# 2. the finalize's own launch geometry
launch = m["ops"]["moe_finalize"]["impl"]["launches"][0]
assert launch["entry"] == "kern_k3_moe_finalize", launch["entry"]
assert launch["block"] == [256, 1, 1], launch["block"]
assert launch["grid"] == [{"mul": [{"ceil_div": ["tokens", 4]}, 4]}, 1, 1], launch["grid"]
launch["block"] = [128, 1, 1]
launch["grid"] = [{"ceil_div": ["tokens", 2]}, 1, 1]
n = sum(1 for c in m["programs"]["prefill"]["calls"] if c["op"] == "moe_finalize")
assert n == 92, n

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(m, indent=1) + "\n")
print(OUT)
