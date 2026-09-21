#!/usr/bin/env python3
"""Candidate: the routed and shared partials land bf16 into `land_add2_v2`.

`l*.lat_up` (routed_expert_up_proj) and `l*.sh_down` (shared down_proj) run as
`gemm_f32` into the f32 `routed_partial` / `shared_partial2` (117 MB each per
layer), and `l*.hidden` (`kern_k3_land_add2`) then reads both and rounds each
through bf16 before the sum: K1c lands `bf16(p1)`, `bf16(p2)` first and adds
them to `f32(prefix2)`, one final round.  So 351 MB per layer -- 234 MB of it
f32 partials that are rounded to bf16 the moment they are read -- is written and
read only to be narrowed, and the kernel runs at 5.7 TB/s of the ~6.1 TB/s this
GPU reaches on a pure stream (measured with the pinned cubin: 61.4 us per call
in the graph).

`kern_k3_land_add2_bf16` (source/k3_land_bf16.cu) is that kernel with bf16
partials, so the two round trips in K1c are no-ops and `hidden` is bit-identical
whenever `bf16(cublas_bf16_tn_f32(A, B)) == cublaslt_bf16_tn(A, B)` for these
three GEMMs (checked with `test16k` on the real `hidden`).  The sum is per
element with no cross-thread reduction, so the launch geometry is free and the
one below is the bandwidth best of the sweep in the round record.

Layer 0 has no routed MoE: `l0.w_dn` writes `routed_partial` and `l0.hidden`
runs add2 with two = 0, so that GEMM moves too.

Derived from the current default manifest; the whole difference is the dtype of
the two partial buffers, the three GEMMs that write them and the one kernel that
reads them (the shape of the adopted `lat_down` landing, gen_lat_bf16.py).
"""
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
ref = ROOT / "manifests/k3-tp4-prefill-16k.json"
out = ROOT / "manifests/k3-tp4-prefill-16k-add2-bf16.json"

# kern_k3_land_add2_bf16: 896 vectors of 8 bf16 per row, GY * 128 threads
GY, BLK = 1, 448

m = json.loads(ref.read_text())
kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
m["modules"]["k3_land_bf16"] = {"source": "k3_land_bf16.cubin",
                               "sha256": kernels["k3_land_bf16"]["sha256"]}

for name in ("routed_partial", "shared_partial2"):
    assert m["buffers"][name]["dtype"] in ("f32", "bf16"), (name, m["buffers"][name])
    m["buffers"][name]["dtype"] = "bf16"

moved = 0
for c in m["programs"]["prefill"]["calls"]:
    suf = c["label"].split(".", 1)[-1]
    if c["op"] == "gemm_f32" and suf in ("lat_up", "sh_down", "w_dn"):
        c["op"] = "gemm_bf16"
        moved += 1
assert moved in (0, 185), moved   # 0 when the default already lands bf16

op = m["ops"]["land_add2_v2"]
assert op["params"] in (["in buffer<f32>", "in buffer<f32>", "in buffer<bf16>",
                          "out buffer<bf16>", "i32", "i32"],
                        ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                         "out buffer<bf16>", "i32", "i32"]), op["params"]
op["params"] = ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                "out buffer<bf16>", "i32", "i32"]
launch = op["impl"]["launches"][0]
assert (launch["module"], launch["entry"]) in (
    ("k3_residual_v2", "kern_k3_land_add2"), ("k3_land_bf16", "kern_k3_land_add2_bf16")), launch
launch["module"] = "k3_land_bf16"
launch["entry"] = "kern_k3_land_add2_bf16"
launch["block"] = [BLK, 1, 1]
launch["grid"] = [{"ceil_div": ["tokens", 4]}, GY, 1]

out.write_text(json.dumps(m, indent=1) + "\n")
print(out.name)
