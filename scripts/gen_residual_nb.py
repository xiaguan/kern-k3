#!/usr/bin/env python3
"""Candidate: the residual + RMSNorm kernel is compiled once per candidate count.

`l*.res_mlp` (`land_add_attnres_rms_bf16_v2`, kernel `kern_k3_land_add_attnres_rms`
in source/k3_residual_v2.cu) reads its candidate count `nb` out of the launch
arguments, so nvcc sizes `V8 x[KNB_MAX]`, `sq/dp[KNB_MAX+1]` and `p[KNB_MAX+1]`
for nb = 8 whatever the call asks for, and every candidate loop keeps its
`if (c < nb)` guard.  `nb` is not a run-time property of this workload: twelve
layers use each value 1..7, nine use 8, and the call site knows which.

`source/k3_residual_nb.cu` writes the same kernel out once per candidate count
with `const int nb = k` in place of the parameter, so the inliner folds the
guards away and the arrays shrink to what is used.  cuobjdump, per entry:

    nb  registers  instructions   blocks/SM at block 128
     1       88         1848        5   (was 4, at 124 registers)
     2       86           --        5
     4      102         3000        5
     8      108         4456        4

This generator makes one op per count and points every call at the one its own
`nb` argument asks for.  The op interface, the launch geometry and the call
arguments are unchanged, so nothing else in the program moves.

Idempotent: re-running it on an already converted manifest only re-checks.

Round 16 builds on this step and carries it: `scripts/gen_residual_smem.py` (the
shared-memory row slice, same entries, same module name, new sha) is replayed
after this one and repeats this split itself if the default does not have it yet,
so that a commit whose default is still main's run-time-`nb` op comes out whole.
This line is why the commit touches this file.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULE = "k3_residual_nb+LAND_BF16=1"
OLD_OP = "land_add_attnres_rms_bf16_v2"
OP = OLD_OP + "_nb{}"
ENTRY = "kern_k3_land_add_attnres_rms_nb{}"
OLD_ENTRY = "kern_k3_land_add_attnres_rms"
NBS = range(1, 9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or args.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    m = json.loads(args.manifest.read_text())
    calls = m["programs"]["prefill"]["calls"]
    converted = all(OP.format(k) in m["ops"] for k in NBS)
    assert OLD_OP in m["ops"] or converted, "the residual landing op is not in this manifest"

    # every call site has to be one of the shapes an entry was built for
    n = 0
    for c in calls:
        if c["op"] == OLD_OP or c["op"].startswith(OLD_OP + "_nb"):
            nb = c["args"][-3]["i32"]
            assert nb in NBS, f"nb {nb} has no entry"
            n += 1
    assert n, "no residual landings to specialize"

    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}
    if not converted:
        base = m["ops"][OLD_OP]
        launch = base["impl"]["launches"][0]
        assert launch["entry"] == OLD_ENTRY, launch["entry"]
        for k in NBS:
            op = json.loads(json.dumps(base))
            op["impl"]["launches"][0]["module"] = MODULE
            op["impl"]["launches"][0]["entry"] = ENTRY.format(k)
            m["ops"][OP.format(k)] = op
    for c in calls:
        if c["op"] == OLD_OP or c["op"].startswith(OLD_OP + "_nb"):
            c["op"] = OP.format(c["args"][-3]["i32"])

    called = {c["op"] for prog in m["programs"].values() for c in prog["calls"]}
    for op in [o for o in m["ops"] if o not in called]:
        del m["ops"][op]
    used_mod = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    assert "k3_residual_v2+LAND_BF16=1" not in used_mod, "the generic module should be free now"
    for mod in [x for x in m["modules"] if x not in used_mod]:
        del m["modules"][mod]
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"{n} residual landings spread over {len(list(NBS))} entries of {MODULE}, "
          f"{len(m['ops'])} ops, {len(m['modules'])} modules -> {out}")


if __name__ == "__main__":
    main()
