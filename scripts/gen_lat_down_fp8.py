#!/usr/bin/env python3
"""Candidate: `l*.lat_down` joins `l*.wsh` on the runtime's `extern:cublaslt_fp8_tn`.

`lat_down` is the MoE path's latent projection (M = tokens/4, N = 3584, K = 7168, 92 calls,
9.9 ms per rank at 16k in bf16).  Its operand is the same `normed` the shared expert reads, and
`scripts/gen_wsh_fp8.py` already quantizes that tensor once per layer into `mlp_normed_fp8`
right in front of the first call of the layer that reads it -- which is `l*.router`, ahead of
`l*.lat_down` -- so this step reuses that operand and scale as they are, adds no quantization
of its own, and only re-pins the weight: `layers.L.w_lat_down` is quantized once in `load` into
`layers.L.w_lat_down_fp8` plus a one-element f32 scale.

numerics-changing: the weight is rounded once to e4m3 (relative error <= 2^-4 of its amax) and
the product is accumulated in f32.  The activation rounding is the one `gen_wsh_fp8` already
carries; no new rounding is introduced on `normed`.

Idempotent: on a manifest that already has `l*.lat_down` on `gemm_fp8_shared` it rewrites the
same file.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "manifests/k3-tp4-prefill-16k.json"
OP = "gemm_fp8_shared"

m = json.loads(DEFAULT.read_text())
assert OP in m["ops"] and "mlp_normed_fp8" in m["buffers"], "run scripts/gen_wsh_fp8.py first"

calls = m["programs"]["prefill"]["calls"]
lat = [c for c in calls if c["label"].endswith(".lat_down")]
assert len(lat) == 92, len(lat)

load = m["programs"]["load"]["calls"]
if all(c["op"] == OP for c in lat):
    DEFAULT.write_text(json.dumps(m, indent=1) + "\n")
    print(DEFAULT.name, "(already applied)")
    raise SystemExit

for c in lat:
    a = c["args"]
    assert c["op"] == "gemm_bf16" and a[0] == {"buf": "normed"}, a
    w, n, k = a[1]["buf"], a[4]["i32"], a[5]["i32"]
    m["buffers"][f"{w}_fp8"] = {"dtype": "u8", "shape": [n, k], "kind": "carry"}
    m["buffers"][f"{w}_scale"] = {"dtype": "f32", "shape": [1], "kind": "carry"}
    load.append({"label": f"load.k3_quant_fp8_pair.{w}", "op": "quant_fp8_tensor",
                 "args": [{"buf": w}, {"buf": f"{w}_fp8"}, {"buf": f"{w}_scale"}, {"buf": "quant_partials"},
                          {"i32": n * k}]})

new = []
for c in calls:
    if c["label"].endswith(".lat_down"):
        w = c["args"][1]["buf"]
        new.append({"label": c["label"], "op": OP,
                    "args": [{"buf": "mlp_normed_fp8"}, {"buf": f"{w}_fp8"}, c["args"][2],
                             {"buf": "mlp_normed_scale"}, {"buf": f"{w}_scale"},
                             c["args"][3], c["args"][4], c["args"][5], c["args"][6]]})
    else:
        new.append(c)
m["programs"]["prefill"]["calls"] = new
DEFAULT.write_text(json.dumps(m, indent=1) + "\n")
print(DEFAULT.name, len(lat), "lat_down calls")
