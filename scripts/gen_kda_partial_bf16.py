#!/usr/bin/env python3
"""Candidate: the KDA q/k/v/gate projection lands bf16 directly.

`l*.qkvg` runs `gemm_f32` (cuBLAS `cublas_bf16_tn_f32`) writing the f32
`kda_partial` [tokens, 12288] (805 MB per call, 69 calls).  Both of its readers
round every element through bf16 first:

  * `span_gather` (`k9_land4` in k3_span_gather_v2.cu): x_i = f32(bf16(partial[i, c]))
    before the causal conv, and
  * `kda_out_gate` (`gg = f32(bf16(gate_partial[b, 3*INNER + ...]))`) before the sigmoid.

Writing the partial as bf16 therefore moves that rounding into the GEMM epilogue
and removes half of the buffer's write and of both reads, with identical values
entering the conv and the gate.

One change, three consequences of it: the buffer dtype, the two reader kernels
(v3 = v2 with the f32 load replaced by a bf16 unpack; same entry, geometry, math
and landing points), and the GEMM op of the 69 producer calls.

Derived from the previous default manifest (`manifests/k3-tp4-prefill-16k.json`
before this change; pass that path as the first argument when the default has
already moved).  The second argument is the output path.
"""
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "manifests/k3-tp4-prefill-16k.json")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "manifests/k3-tp4-prefill-16k-kda-bf16.json")

m = json.loads(SRC.read_text())
kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]

# The two v3 cubins are the v2 sources with the partial load switched to bf16.
# These hashes are what `nvcc -cubin -arch=sm_103a -DHEADS=24` produces from
# them in the evaluation image (rebuilt and compared byte for byte); they enter
# kernels.toml only when the candidate is adopted.
SHA = {
    "k3_span_gather_v3+HEADS=24": "d5e4aa72745cd207eec897a783ae3affc25e0334372caf443b6e01f9fe41102f",
    "k3_kda_out_gate_v3+HEADS=24": "9e1de0c40f58bd748e604efcabee8d8c535b9a60a0e28adb2ce942dcacdbf069",
}
for name, sha in SHA.items():
    assert kernels.get(name, {}).get("sha256", sha) == sha, name

# 1. the producer lands bf16, its readers take the bf16 buffer
m["buffers"]["kda_partial"]["dtype"] = "bf16"
n = 0
for c in m["programs"]["prefill"]["calls"]:
    if c["op"] == "gemm_f32" and c["label"].endswith(".qkvg"):
        c["op"] = "gemm_bf16"
        n += 1
assert n == 69, n

# 2. the span gather reads the bf16 partial (k3_span_gather_v3)
m["modules"]["k3_span_gather_v3+HEADS=24"] = {
    "source": "k3_span_gather_v3+HEADS=24.cubin", "sha256": SHA["k3_span_gather_v3+HEADS=24"]}
assert m["ops"]["span_gather_v2"]["params"][0] == "in buffer<f32>"
m["ops"]["span_gather_v2"]["params"][0] = "in buffer<bf16>"
launch = m["ops"]["span_gather_v2"]["impl"]["launches"][0]
assert launch["params"][0] == "in buffer<f32>" and launch["module"] == "k3_span_gather_v2+HEADS=24"
launch["params"][0] = "in buffer<bf16>"
launch["module"] = "k3_span_gather_v3+HEADS=24"
del m["modules"]["k3_span_gather_v2+HEADS=24"]

# 3. the output gate reads the same buffer's band 3 (k3_kda_out_gate_v3)
m["modules"]["k3_kda_out_gate_v3+HEADS=24"] = {
    "source": "k3_kda_out_gate_v3+HEADS=24.cubin", "sha256": SHA["k3_kda_out_gate_v3+HEADS=24"]}
assert m["ops"]["kda_out_gate_v2"]["params"][1] == "in buffer<f32>"
m["ops"]["kda_out_gate_v2"]["params"][1] = "in buffer<bf16>"
gl = m["ops"]["kda_out_gate_v2"]["impl"]["launches"][0]
assert gl["module"] == "k3_kda_out_gate_v2+HEADS=24"
gl["module"] = "k3_kda_out_gate_v3+HEADS=24"
del m["modules"]["k3_kda_out_gate_v2+HEADS=24"]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(m, indent=1) + "\n")
print(OUT)
