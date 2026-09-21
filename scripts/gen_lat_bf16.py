#!/usr/bin/env python3
"""Candidate: the MoE latent down-projection lands bf16 directly.

`l*.lat_down` currently runs as `gemm_f32` writing the f32 `latent_partial`,
and `l*.latent` (`land_n3584`, `kern_k3_land`) is a pure landing that rounds
that f32 to bf16 into `latent`.  `kern_k3_land` is documented as
`o[b,i] = bf16(p[b*ldc + off + i])` -- one landing, no arithmetic -- and the
only reader of `latent` is `moe_quant` (bf16).  So the GEMM can write bf16
straight into `latent` and the 92 landing calls disappear.

Derived from the current default manifest; the whole difference is that one
GEMM output dtype + the landing calls it makes redundant.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ref = ROOT / "manifests/k3-tp4-prefill-16k.json"
out = ROOT / "manifests/k3-tp4-prefill-16k-lat-bf16.json"

m = json.loads(ref.read_text())
calls = m["programs"]["prefill"]["calls"]
kept = []
moved = 0
for c in calls:
    if c["op"] == "land_n3584":
        continue
    if c["op"] == "gemm_f32" and c["label"].endswith(".lat_down"):
        c["op"] = "gemm_bf16"
        # third arg is the output buffer: write `latent` (bf16) instead
        c["args"][2] = {"buf": "latent"}
        moved += 1
    kept.append(c)
m["programs"]["prefill"]["calls"] = kept
assert moved == 92, moved
del m["buffers"]["latent_partial"]
del m["ops"]["land_n3584"]
out.write_text(json.dumps(m, indent=1) + "\n")
print(out.name)
