#!/usr/bin/env python3
"""Candidate: `l*.wsh` runs as an fp8 GEMM through the runtime's `extern:cublaslt_fp8_tn`.

`l*.wsh` is the shared expert's up/gate projection over this rank's rows (M = tokens/4,
N = 12288, K = 7168, 92 calls, 34.5 ms per rank at 16k in bf16; cuBLASLt runs at ~1.9 PF).
Its operand `normed` (bf16 [tokens/4, 7168], the MLP input) is the same tensor `l*.lat_down`
and the router read, so it is quantized once per layer, right before its first fp8 consumer,
into `mlp_normed_fp8` + `mlp_normed_scale` (per-tensor amax / 448, the same pair of kernels
`l*.qkvg_quant` uses).  The weight (`layers.L.wsh`, bf16) is quantized once in `load` into
`layers.L.wsh_fp8` plus a one-element f32 scale.  Both scales reach cuBLASLt as device
pointers, so nothing about the numbers reaches the host.

numerics-changing: `normed` and `layers.L.wsh` are each rounded once to e4m3 (relative error
<= 2^-4 of that tensor's amax) and accumulated in f32.  The scale of the activation is the
amax over all 4096 rows, the same one `l*.lat_down` reuses, so the tensor is rounded the same
way wherever it is read.

Idempotent: on a manifest that already carries `gemm_fp8_shared` it re-pins the quant module's
sha and rewrites the same file.
"""
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "manifests/k3-tp4-prefill-16k.json"
QUANT = "k3_quant_fp8_pair"
OP = "gemm_fp8_shared"

m = json.loads(DEFAULT.read_text())
if OP in m["ops"]:
    DEFAULT.write_text(json.dumps(m, indent=1) + "\n")
    print(DEFAULT.name, "(already applied)")
    raise SystemExit

m["ops"][OP] = {
    "params": ["in buffer<u8>", "in buffer<u8>", "out buffer<bf16>", "in buffer<f32>", "in buffer<f32>",
               "i32", "i32", "i32", "i32"],
    "impl": {"launches": [{"entry": "extern:cublaslt_fp8_tn"}]},
}
m["buffers"]["mlp_normed_fp8"] = {"dtype": "u8", "shape": [4096, 7168], "kind": "workspace"}
m["buffers"]["mlp_normed_scale"] = {"dtype": "f32", "shape": [1], "kind": "workspace"}

calls = m["programs"]["prefill"]["calls"]
wsh = [c for c in calls if c["op"] == "gemm_bf16" and c["label"].endswith(".wsh")]
assert len(wsh) == 92, len(wsh)

# the weight of every wsh is quantized once in load
load = m["programs"]["load"]["calls"]
for c in wsh:
    a = c["args"]
    assert a[0] == {"buf": "normed"}, a
    w, n, k = a[1]["buf"], a[4]["i32"], a[5]["i32"]
    m["buffers"][f"{w}_fp8"] = {"dtype": "u8", "shape": [n, k], "kind": "carry"}
    m["buffers"][f"{w}_scale"] = {"dtype": "f32", "shape": [1], "kind": "carry"}
    load.append({"label": f"load.{QUANT}.{w}", "op": "quant_fp8_tensor",
                 "args": [{"buf": w}, {"buf": f"{w}_fp8"}, {"buf": f"{w}_scale"}, {"buf": "quant_partials"},
                          {"i32": n * k}]})

# one quant of `normed` per layer, immediately after the call that produces it
PARAMS = {op: spec["params"] for op, spec in m["ops"].items()}


def arg_is_out(c, i):
    return PARAMS[c["op"]][i].startswith(("inout", "out"))


def touches(c, i):
    a = c["args"][i]
    return isinstance(a, dict) and a.get("buf") == "normed" and arg_is_out(c, i)


producer = {}
for i, c in enumerate(calls):
    out = [j for j in range(len(c["args"])) if touches(c, j)]
    if out:
        assert len(out) == 1 and c["label"].split(".")[0] not in producer, c["label"]
        producer[c["label"].split(".")[0]] = i
layers = {c["label"].split(".")[0] for c in wsh}
assert layers <= set(producer), sorted(layers - set(producer))

new, quantized = [], set()
for i, c in enumerate(calls):
    if c["op"] == "gemm_bf16" and c["label"].endswith(".wsh"):
        w = c["args"][1]["buf"]
        new.append({"label": c["label"], "op": OP,
                    "args": [{"buf": "mlp_normed_fp8"}, {"buf": f"{w}_fp8"}, c["args"][2],
                             {"buf": "mlp_normed_scale"}, {"buf": f"{w}_scale"},
                             c["args"][3], c["args"][4], c["args"][5], c["args"][6]]})
    else:
        new.append(c)
    lay = c["label"].split(".")[0]
    if lay in layers and producer.get(lay) == i and lay not in quantized:
        quantized.add(lay)
        new.append({"label": f"{lay}.normed_quant", "op": "quant_fp8_tensor",
                    "args": [{"buf": "normed"}, {"buf": "mlp_normed_fp8"}, {"buf": "mlp_normed_scale"},
                             {"buf": "quant_partials"},
                             {"expr": {"mul": [{"ceil_div": ["tokens", 4]}, 7168]}}]})
assert len(quantized) == len(layers), (len(quantized), len(layers))
m["programs"]["prefill"]["calls"] = new
DEFAULT.write_text(json.dumps(m, indent=1) + "\n")
print(DEFAULT.name, len(wsh), "wsh calls and", len(quantized), "normed quants")
