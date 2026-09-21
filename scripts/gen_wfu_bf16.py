#!/usr/bin/env python3
"""Candidate: the MoE-fused MLA up-projection lands bf16, and mla_prep reads it.

`l*.wfu` (24 calls, one per MLA layer) is `gemm_f32`: the bf16 GEMM writes the
f32 `mla_fused_partial` (16,384 x 5,184 x 4 B = 340 MB per call, the largest
single f32 tensor in the graph).  Its only reader is `l*.mla_prep`, whose two
kernels load four f32 per unit and immediately round each one to bf16
(`landf`) before the rms sum, the q_norm landing, the latent row and the gate
band -- the source documents "every f32 partial column is rounded to bf16
before it is used".

So the GEMM can land bf16 (its own f32 accumulator rounded once, the same
landing `landf` does) and `mla_prep` can load bf16 directly: 340 MB -> 170 MB
per call for the write and for the read.  The GEMM itself gains only a little
(an f32 C costs 1-4% standalone: 0.596 vs 0.588 ms on this shape); the read
halving is the point.

`source/k3_mla_prep_bf16.cu` is `source/k3_mla_prep_v2.cu` -- agent a's kernel,
unmodified on disk -- with `partial` typed `const bf16*`, the four-element
loads turned into 8-byte uint2 loads and `landf` the identity.  Geometry,
reduction order and every landing point are unchanged, so the values are the
same as v2's whenever the GEMM's bf16 landing equals `bf16(f32(partial))`, i.e.
whenever the two GEMM paths round the same f32 accumulator.

Interface for agent a: the op `mla_prep` changes its first param from
`in buffer<f32>` to `in buffer<bf16>` (both launches in `impl.launches`) and
points at the new module; the buffer `mla_fused_partial` becomes bf16; the 24
`l*.wfu` calls become `gemm_bf16`.  Nothing else in the op changes.

Derived from the current default manifest; re-running on a default that already
carries the change is a no-op.
"""
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ref = ROOT / "manifests/k3-tp4-prefill-16k.json"
out = ROOT / "manifests/k3-tp4-prefill-16k-wfu-bf16.json"

NAME = "k3_mla_prep_bf16+INNER=3072+MLA_FUSED=5184"
OLD = "k3_mla_prep_v2+INNER=3072+MLA_FUSED=5184"

m = json.loads(ref.read_text())
k = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"][NAME]
m["modules"][NAME] = {"source": f"{NAME}.cubin", "sha256": k["sha256"]}
m["modules"].pop(OLD, None)

m["buffers"]["mla_fused_partial"]["dtype"] = "bf16"
wfu = [c for c in m["programs"]["prefill"]["calls"]
       if c["label"].split(".", 1)[-1] == "wfu" and c["op"] in ("gemm_f32", "gemm_bf16")]
assert len(wfu) == 24, len(wfu)
for c in wfu:
    c["op"] = "gemm_bf16"

op = m["ops"]["mla_prep"]
op["params"][0] = "in buffer<bf16>"
for launch in op["impl"]["launches"]:
    launch["module"] = NAME
    if launch["params"][0].endswith("<f32>"):
        launch["params"][0] = "in buffer<bf16>"
out.write_text(json.dumps(m, indent=1) + "\n")
print(out.name)
