#!/usr/bin/env python3
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
reference = ROOT / "manifests/k3-tp4-prefill-16k.json"
candidate = ROOT / "manifests/k3-tp4-prefill-16k-l0-bf16.json"
m = json.loads(reference.read_text())
k = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]["k3_situ_bf16"]
m["modules"]["k3_situ_bf16"] = {
    "source": "k3_situ_bf16.cubin", "sha256": k["sha256"],
}
m["buffers"]["dense_partial"]["dtype"] = "bf16"
call = next(c for c in m["programs"]["prefill"]["calls"] if c["label"] == "l0.wgu")
call["op"] = "gemm_bf16"
op = m["ops"]["land_situ_n33792"]
op["params"][0] = "in buffer<bf16>"
launch = op["impl"]["launches"][0]
launch["params"][0] = "in buffer<bf16>"
launch["module"] = "k3_situ_bf16"
launch["entry"] = "kern_k3_situ_bf16"
candidate.write_text(json.dumps(m, indent=1) + "\n")
print(candidate.name)
