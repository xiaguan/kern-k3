#!/usr/bin/env python3
"""Candidate: the fused SiTU + latent-RMSNorm landing runs one row per block.

`l*.shared_situ` (`land_situ_rms`, `kern_k3_situ_rms_bf16`) runs 92 times per
rank; in the graph it is 54.9 us per call for 209.7 MB.  It is not at a
bandwidth wall: the SiTU half is SFU/MUFU bound (25.2M elements x 4 MUFU =
100.6M MUFU in ~39 us) and the rms half is latency bound, so the kernel's floor
is the SiTU half alone.  The v1 launch packs both jobs into 3 blocks out of
every 3 (`3 * ceil_div(tokens, 16)`, RMS_R = 4 rms rows and 2 situ rows per
block); with 1024 rms rows and 2 rows per situ block the two streams are coarse
and the launch has only 3072 blocks.

`source/k3_land_fused_v2.cu` (module `k3_land_fused_v2`, same entry) keeps the
arithmetic of v1 element for element -- the situ part is elementwise, the rms
part keeps warp w on vectors [32w, 32w+32), the idle warps contributing 0, the
same block_sum butterfly, the same rsqrtf(sum/h + 1e-5f) and the same
`__hmul2(__floats2bfloat162_rn(x*rs), gamma)` landing -- and changes only the
job layout: of every RATIO + 1 blocks one does RMS_R = 1 rms row and RATIO = 1
does SITU_R = 1 situ row, so the launch is `2 * ceil_div(tokens, 4)` = 8192
blocks of 512 threads, one row each.  Measured standalone, bit-checked against
`kern_k3_land_situ` + `kern_k3_rms` (0 of 25,165,824 situ and 0 of 14,680,064
rms outputs differ): v1 55.10 us, this 45.79 us for the same data.  The
remaining ~5 us is the rms half, which the v1 layout also pays.

The diff against the default is one op: `land_situ_rms` moves to the new module
and its grid becomes 2 * ceil_div(tokens, 4).  Derived from the current default
manifest; re-running on a default that already carries the change is a no-op.
"""
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ref = ROOT / "manifests/k3-tp4-prefill-16k.json"
out = ROOT / "manifests/k3-tp4-prefill-16k-land-geom.json"

NAME = "k3_land_fused_v2"
OLD = "k3_land_fused"
RATIO = 1  # situ blocks per rms block; RMS_R = 1 rms row and SITU_R = 1 situ row per block

m = json.loads(ref.read_text())
k = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"][NAME]
m["modules"][NAME] = {"source": f"{NAME}.cubin", "sha256": k["sha256"]}
m["modules"].pop(OLD, None)
launch = m["ops"]["land_situ_rms"]["impl"]["launches"][0]
launch["module"] = NAME
launch["grid"] = [{"mul": [{"ceil_div": ["tokens", 4]}, RATIO + 1]}, 1, 1]
out.write_text(json.dumps(m, indent=1) + "\n")
print(out.name)
