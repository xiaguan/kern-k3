#!/usr/bin/env python3
"""Candidate: the fused residual kernel lets the next launch carry its release.

`kern_k3_attnres_rms_push` (source/k3_residual_push.cu) writes this rank's slice
of `normed_all` into all four ranks' copies and then executed
`__threadfence_system()` in **every** thread -- 524288 of them per call, each one
a system-scope fence plus a `CCTL.IVALL` that throws away the SM's L1.  Measured,
that fence costs 40 us per call: `l*.res_in` is 314.9 us with it and 274.9 us
without, and 274.9 is what the same kernel costs when it has no candidates to
read at all, i.e. its store floor.

The fence is ordering stores that are ordered already.  The peers read this
rank's slice once the barrier of the *next* call passes, and that barrier is a
launch of its own -- `l*.normed_sync` -> `kern_k3_rs_arrive1`, one block since
round 7 -- which executes `__threadfence_system()` before it bumps this rank's
arrival.  A kernel boundary flushes the stores of a finished kernel to the
device's coherence point, which is the L2 a peer's NVLink read is serviced from,
and the fence that runs afterwards is a release for everything that
happens-before it, the finished kernel's stores included.  So the ordering the
protocol needs is still there, once per call instead of 524288 times.

`source/k3_residual_push_v3.cu` is that kernel with the fence removed; the
arithmetic, the launch geometry and everything else are untouched, so the output
is bit-identical.

Idempotent: re-running it on an already converted manifest only re-checks.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULE = "k3_residual_push_v3"
ENTRY = "kern_k3_attnres_rms_push"
OPS = ("attnres_rms_push_v3", "attnres_rms_push_first_v3")
OLD = "k3_residual_push"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or args.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    m = json.loads(args.manifest.read_text())
    assert all(o in m["ops"] for o in OPS), "the fused residual ops are not in this manifest"
    for o in OPS:
        l = m["ops"][o]["impl"]["launches"][0]
        assert l["entry"] == ENTRY, l["entry"]
        assert l["module"] in (OLD, MODULE), l["module"]
    # the barrier of the next call is the one that carries the release
    assert m["ops"]["k3_allgather_push"]["impl"]["launches"][0]["module"] == "k3_rs_arrive1", \
        "run scripts/gen_arrive1.py first (the arrival kernel carries the fence)"
    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}
    for o in OPS:
        m["ops"][o]["impl"]["launches"][0]["module"] = MODULE
    used = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    if OLD not in used:
        m["modules"].pop(OLD, None)
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"fused residual on {MODULE} (no per-thread fence), {len(m['ops'])} ops, "
          f"{len(m['modules'])} modules -> {out}")


if __name__ == "__main__":
    main()
