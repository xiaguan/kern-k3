#!/usr/bin/env python3
"""Candidate: the fp8 operand quantize runs vectorized (module `k3_quant_fp8_pair_v2`).

`quant_fp8_tensor` (the two passes that turn a bf16 tensor into e4m3 + a per-tensor scale) is 29.4 ms
per rank, 0.43 ms per call at the qkvg shape -- 1400 GB/s for 235 MB read twice and 117 MB written.
The v1 kernel moves 2 bytes per thread per step and converts element by element through
`__nv_fp8_e4m3(float)`.  v2 is the same arithmetic in a memory-shaped loop: 16 B loads / 8 B stores
and the amax over the raw |bf16| bit patterns (an unsigned max -- for bf16 the magnitude ordering is
the unsigned ordering of the low 15 bits), 3500 GB/s on the same probe.

Bit-identical to v1: the maximum is exact and order-independent, the scale is the same `amax/448`,
and each element is rounded once by the same round-to-nearest-even e4m3 conversion.  Nothing else in
the manifest changes -- same op, same params, same launch geometry, only the module and its two entry
points.

Idempotent: on a manifest that already points at `k3_quant_fp8_pair_v2` it re-pins the module sha.
"""
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "manifests/k3-tp4-prefill-16k.json"
OLD, NEW = "k3_quant_fp8_pair", "k3_quant_fp8_pair_v2"
ENTRIES = {"kern_quant_fp8_amax": "kern_quant_fp8_amax_v2",
           "kern_quant_fp8_apply": "kern_quant_fp8_apply_v2"}

kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
m = json.loads(DEFAULT.read_text())
m["modules"][NEW] = {"source": f"{NEW}.cubin", "sha256": kernels[NEW]["sha256"]}
m["modules"].pop(OLD, None)
for launch in m["ops"]["quant_fp8_tensor"]["impl"]["launches"]:
    launch["module"] = NEW
    launch["entry"] = ENTRIES.get(launch["entry"], launch["entry"])
DEFAULT.write_text(json.dumps(m, indent=1) + "\n")
print(DEFAULT.name)
