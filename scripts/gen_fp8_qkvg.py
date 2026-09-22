#!/usr/bin/env python3
"""Candidate: `l*.qkvg` runs as an fp8 GEMM through the runtime's `extern:cublaslt_fp8_tn`.

`l*.qkvg` is the graph's largest dense GEMM family (M = tokens, N = 12288, K = 7168, 69 calls,
101 ms per rank at 16k in bf16).  cuBLASLt runs the same shape in e4m3 at 4.5 PF: 0.62 ms per call
against 1.44.  The weight (`layers.L.wbig`, bf16) is quantized once in `load` into a `wbig_fp8`
carry buffer plus a one-element f32 scale (per-tensor amax / 448); the activation (`normed_all`)
is quantized the same way in `prefill` right before each GEMM.  Both scales go to cuBLASLt as
device pointers (`A/B_SCALE_POINTER`), so nothing about the numbers reaches the host.

numerics-changing: both operands are rounded once to e4m3 (relative error <= 2^-4 of the tensor's
amax) and accumulated in f32 exactly as the bf16 product was.

Idempotent: on a manifest that already carries `gemm_fp8_qkvg` it re-pins the quant module's sha.
"""
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "manifests/k3-tp4-prefill-16k.json"
QUANT = "k3_quant_fp8_pair"
NQ = 2048
N, K = 12288, 7168

m = json.loads(DEFAULT.read_text())
sha = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"][QUANT]["sha256"]
m["modules"][QUANT] = {"source": f"{QUANT}.cubin", "sha256": sha}
if "gemm_fp8_qkvg" in m["ops"]:
    DEFAULT.write_text(json.dumps(m, indent=1) + "\n")
    print(DEFAULT.name, "(already applied)")
    raise SystemExit

m["ops"]["quant_fp8_tensor"] = {
    "params": ["in buffer<bf16>", "out buffer<u8>", "out buffer<f32>", "out buffer<f32>", "i32"],
    "impl": {"launches": [
        {"module": QUANT, "entry": "kern_quant_fp8_amax",
         "params": ["in buffer<bf16>", "i32", "out buffer<f32>"],
         "block": [256, 1, 1], "grid": [NQ, 1, 1],
         "args": [{"param": 0}, {"param": 4}, {"param": 3}]},
        {"module": QUANT, "entry": "kern_quant_fp8_apply",
         "params": ["in buffer<bf16>", "out buffer<u8>", "i32", "in buffer<f32>", "i32", "out buffer<f32>"],
         "block": [256, 1, 1], "grid": [NQ, 1, 1],
         "args": [{"param": 0}, {"param": 1}, {"param": 4}, {"param": 3}, {"i32": NQ}, {"param": 2}]},
    ]},
}
m["ops"]["gemm_fp8_qkvg"] = {
    "params": ["in buffer<u8>", "in buffer<u8>", "out buffer<bf16>", "in buffer<f32>", "in buffer<f32>",
               "i32", "i32", "i32", "i32"],
    "impl": {"launches": [{"entry": "extern:cublaslt_fp8_tn"}]},
}
m["buffers"]["normed_fp8"] = {"dtype": "u8", "shape": [16384, K], "kind": "workspace"}
m["buffers"]["normed_fp8_scale"] = {"dtype": "f32", "shape": [1], "kind": "workspace"}
m["buffers"]["quant_partials"] = {"dtype": "f32", "shape": [NQ], "kind": "workspace"}

calls = m["programs"]["prefill"]["calls"]
qkvg = [c for c in calls if c["op"] == "gemm_bf16" and c["label"].endswith(".qkvg")]
assert len(qkvg) == 69, len(qkvg)
weights = []
for c in qkvg:
    a = c["args"]
    assert a[0] == {"buf": "normed_all"} and a[3] == {"var": "tokens"}, a
    assert [x["i32"] for x in a[4:]] == [N, K, N], a
    weights.append(a[1]["buf"])

load = m["programs"]["load"]["calls"]
for w in weights:
    m["buffers"][f"{w}_fp8"] = {"dtype": "u8", "shape": [N, K], "kind": "carry"}
    m["buffers"][f"{w}_scale"] = {"dtype": "f32", "shape": [1], "kind": "carry"}
    load.append({"label": f"load.{QUANT}.{w}", "op": "quant_fp8_tensor",
                 "args": [{"buf": w}, {"buf": f"{w}_fp8"}, {"buf": f"{w}_scale"}, {"buf": "quant_partials"},
                          {"i32": N * K}]})

new = []
for c in calls:
    if c["op"] == "gemm_bf16" and c["label"].endswith(".qkvg"):
        lay, w = c["label"].split(".")[0], c["args"][1]["buf"]
        new.append({"label": f"{lay}.qkvg_quant", "op": "quant_fp8_tensor",
                    "args": [{"buf": "normed_all"}, {"buf": "normed_fp8"}, {"buf": "normed_fp8_scale"},
                             {"buf": "quant_partials"}, {"expr": {"mul": ["tokens", K]}}]})
        new.append({"label": c["label"], "op": "gemm_fp8_qkvg",
                    "args": [{"buf": "normed_fp8"}, {"buf": f"{w}_fp8"}, {"buf": "kda_partial"},
                             {"buf": "normed_fp8_scale"}, {"buf": f"{w}_scale"},
                             {"var": "tokens"}, {"i32": N}, {"i32": K}, {"i32": N}]})
    else:
        new.append(c)
m["programs"]["prefill"]["calls"] = new
DEFAULT.write_text(json.dumps(m, indent=1) + "\n")
print(DEFAULT.name)
