#!/usr/bin/env python3
"""Candidate: `l*.sh_down` runs as an fp8 GEMM through the runtime's `extern:cublaslt_fp8_tn`.

`sh_down` is the shared expert's down projection (M = tokens/4, N = 7168, K = 6144, 92 calls,
17.3 ms per rank at 16k in bf16).  Its operand `shared_act` (bf16 [tokens/4, 6144], the SiTU
output of the shared expert) is produced by `l*.shared_situ` and read by nothing else, so it is
quantized once per layer, right after that producer, into `shared_act_fp8` +
`shared_act_scale` (per-tensor amax / 448, the same pair of kernels `l*.qkvg_quant` uses).  The
weight (`layers.L.sh_down`, bf16) is quantized once in `load` into `layers.L.sh_down_fp8` plus a
one-element f32 scale.  Both scales reach cuBLASLt as device pointers, so nothing about the
numbers reaches the host.

numerics-changing: `shared_act` and `layers.L.sh_down` are each rounded once to e4m3 (relative
error <= 2^-4 of that tensor's amax) and the product is accumulated in f32 exactly as the bf16
product was.

Idempotent: on a manifest that already has `l*.sh_down` on `gemm_fp8_shared` it rewrites the same
file.
"""
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "manifests/k3-tp4-prefill-16k.json"
QUANT = "k3_quant_fp8_pair"
OP = "gemm_fp8_shared"
SRC, DST, K = "shared_act", "shared_act_fp8", 6144

m = json.loads(DEFAULT.read_text())
assert OP in m["ops"], "run scripts/gen_wsh_fp8.py first"

calls = m["programs"]["prefill"]["calls"]
down = [c for c in calls if c["label"].endswith(".sh_down")]
assert len(down) == 92, len(down)
if all(c["op"] == OP for c in down):
    DEFAULT.write_text(json.dumps(m, indent=1) + "\n")
    print(DEFAULT.name, "(already applied)")
    raise SystemExit

m["buffers"][DST] = {"dtype": "u8", "shape": [4096, K], "kind": "workspace"}
m["buffers"]["shared_act_scale"] = {"dtype": "f32", "shape": [1], "kind": "workspace"}

load = m["programs"]["load"]["calls"]
for c in down:
    a = c["args"]
    assert c["op"] == "gemm_bf16" and a[0] == {"buf": SRC}, a
    w, n, k = a[1]["buf"], a[4]["i32"], a[5]["i32"]
    assert k == K, k
    m["buffers"][f"{w}_fp8"] = {"dtype": "u8", "shape": [n, k], "kind": "carry"}
    m["buffers"][f"{w}_scale"] = {"dtype": "f32", "shape": [1], "kind": "carry"}
    load.append({"label": f"load.{QUANT}.{w}", "op": "quant_fp8_tensor",
                 "args": [{"buf": w}, {"buf": f"{w}_fp8"}, {"buf": f"{w}_scale"}, {"buf": "quant_partials"},
                          {"i32": n * k}]})

# the quant sits immediately after the call that produces `shared_act`
PARAMS = {op: spec["params"] for op, spec in m["ops"].items()}


def writes(c, buf):
    return any(isinstance(a, dict) and a.get("buf") == buf and PARAMS[c["op"]][i].startswith(("inout", "out"))
               for i, a in enumerate(c["args"]))


producer = {}
for i, c in enumerate(calls):
    if writes(c, SRC):
        assert c["label"].split(".")[0] not in producer, c["label"]
        producer[c["label"].split(".")[0]] = i
layers = {c["label"].split(".")[0] for c in down}
assert layers <= set(producer), sorted(layers - set(producer))

new, quantized = [], set()
for i, c in enumerate(calls):
    if c["op"] == "gemm_bf16" and c["label"].endswith(".sh_down"):
        w = c["args"][1]["buf"]
        new.append({"label": c["label"], "op": OP,
                    "args": [{"buf": DST}, {"buf": f"{w}_fp8"}, c["args"][2],
                             {"buf": "shared_act_scale"}, {"buf": f"{w}_scale"},
                             c["args"][3], c["args"][4], c["args"][5], c["args"][6]]})
    else:
        new.append(c)
    lay = c["label"].split(".")[0]
    if lay in layers and producer.get(lay) == i and lay not in quantized:
        quantized.add(lay)
        new.append({"label": f"{lay}.shared_act_quant", "op": "quant_fp8_tensor",
                    "args": [{"buf": SRC}, {"buf": DST}, {"buf": "shared_act_scale"},
                             {"buf": "quant_partials"},
                             {"expr": {"mul": [{"ceil_div": ["tokens", 4]}, K]}}]})
assert len(quantized) == len(layers), (len(quantized), len(layers))
m["programs"]["prefill"]["calls"] = new
DEFAULT.write_text(json.dumps(m, indent=1) + "\n")
print(DEFAULT.name, len(down), "sh_down calls and", len(quantized), "shared_act quants")
