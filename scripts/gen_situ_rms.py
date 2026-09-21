#!/usr/bin/env python3
"""Candidate: the shared-expert SiTU landing and the latent RMSNorm in one launch.

`l*.lat_norm` (`rms`, 512-thread blocks) costs 28.6 us per call in the graph for
58.7 MB: 2.06 TB/s, one 16 B load per thread, two block-wide `__syncthreads`
per row.  It is the only op of the dense/shared area that is at neither the FLOP
wall (every GEMM there runs at 1.8-2.0 PFLOPS, cuBLASLt's rate on this node) nor
the DRAM wall (`land_situ` 5.3-5.6 TB/s): it is latency-bound.  `l*.shared_situ`
(`land_situ_n6144`, 45.2 us eagerly for 251.7 MB) runs 555 us later in the same
layer and reads nothing the rms touches, but a CUDA graph serialises the two
launches, so the pair costs 71.8 us per layer.

`source/k3_land_fused.cu` (module `k3_land_fused`, entry
`kern_k3_situ_rms_bf16`) does both in one launch of 3 * ceil_div(tokens, 16)
512-thread blocks: two blocks in three do two situ rows each, one in three does
four rms rows; an SM holds both kinds, so the latency-bound rms stream runs
under the situ stream instead of after it (52.8 us for the same 209.7 MB, 19 us
less than the pair).  `l*.wsh` lands bf16 (`gemm_f32` -> `gemm_bf16`): the
pinned `kern_k3_land_situ` reads f32 and lands both operands to bf16 before the
activation anyway, so that landing is a no-op and the values are unchanged.

Numerics are bit-identical: the rms part keeps warp `w` on vectors
[32w, 32w + 32), the idle warps contribute 0 and every row sees the same
`block_sum` tree, the same `rsqrtf(sum/h + 1e-5f)` and the same
`__hmul2(__floats2bfloat162_rn(x*rs), gamma)` landing as `kern_k3_rms` at block
512; the situ part keeps the pinned expression element for element.  The
difference is the dtype of `shared_partial`, the op of the 92 `wsh` calls, the
new fused op replacing `shared_situ` + `lat_norm`, and `lat_up` (which reads
the rms output) moving after it.  Derived from the current default manifest;
re-running on a default that already carries the change is a no-op.
"""
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
ref = ROOT / "manifests/k3-tp4-prefill-16k.json"
out = ROOT / "manifests/k3-tp4-prefill-16k-situ-rms.json"

RMS_R = 4  # rms rows per block in kern_k3_situ_rms_bf16

m = json.loads(ref.read_text())
kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
m["modules"]["k3_land_fused"] = {"source": "k3_land_fused.cubin",
                                "sha256": kernels["k3_land_fused"]["sha256"]}

buf = m["buffers"]["shared_partial"]
assert buf["dtype"] in ("f32", "bf16"), buf
buf["dtype"] = "bf16"

op = {
    "params": ["in buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>",
               "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
    "impl": {"launches": [{
        "module": "k3_land_fused",
        "entry": "kern_k3_situ_rms_bf16",
        "params": ["in buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>",
                   "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32",
                   "i32"],
        "block": [512, 1, 1],
        "grid": [{"mul": [{"ceil_div": ["tokens", 4 * RMS_R]}, 3]}, 1, 1],
        "args": [{"param": 0}, {"param": 1}, {"param": 2}, {"param": 3},
                 {"param": 4}, {"i32": 6144}, {"param": 5}, {"i32": 3584},
                 {"param": 6}],
    }]},
}
if "land_situ_rms" in m["ops"]:
    assert m["ops"]["land_situ_rms"] == op, "land_situ_rms differs from this generator"
else:
    m["ops"]["land_situ_rms"] = op

moved = fused = norm = 0
gbuf = {}
calls = []
for c in m["programs"]["prefill"]["calls"]:
    suf = c["label"].split(".", 1)[-1] if c["label"].startswith("l") else c["label"]
    if suf == "wsh" and c["op"] == "gemm_f32":
        c["op"] = "gemm_bf16"
        moved += 1
        calls.append(c)
    elif suf == "lat_norm":
        norm += 1  # folded into the fused call of the same layer
        gbuf[c["label"].rsplit(".", 1)[0]] = c["args"][1]["buf"]
    elif suf == "shared_situ":
        lay = c["label"].rsplit(".", 1)[0]
        lat_up = [x for x in m["programs"]["prefill"]["calls"]
                  if x["label"] == lay + ".lat_up"]
        assert len(lat_up) == 1, c["label"]
        fused += 1
        gam = gbuf.get(lay, "layers." + lay[1:] + ".gamma_lat")
        assert gam in m["buffers"], gam
        calls.append({"label": c["label"], "op": "land_situ_rms", "args": [
            {"buf": "shared_partial"}, {"buf": "shared_act"},
            {"buf": "routed_latent"}, {"buf": gam},
            {"buf": "routed_latent_norm"},
            {"expr": {"ceil_div": ["tokens", 4]}},
            {"expr": {"ceil_div": ["tokens", 4]}}]})
        calls.append(dict(lat_up[0]))  # lat_up reads what the fused call wrote
    elif suf == "lat_up":
        pass  # emitted right after the fused call
    else:
        calls.append(c)

assert fused == 92, fused
assert moved in (0, 92) and norm in (0, 92) and moved == norm, (moved, norm)
m["programs"]["prefill"]["calls"] = calls

# `rms` and `land_situ_n6144` have no caller any more; the runtime wants every
# op called.
for dead in ("rms", "land_situ_n6144"):
    if dead in m["ops"] and not any(c["op"] == dead for c in calls):
        del m["ops"][dead]

out.write_text(json.dumps(m, indent=1) + "\n")
print(out.name, f"wsh->bf16 {moved}, fused {fused}, lat_norm folded {norm}")
