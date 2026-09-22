#!/usr/bin/env python3
"""Candidate: the two peer-facing pushes take the cheap path.

Both of this partition's pushes write sixteen-byte vectors into other ranks'
memory over NVLink, and both paid for it twice:

  * `l*.gather_small` (`kern_k3_allgather4_push`) published its arrival -- the
    bump every peer waits for -- with a `__threadfence_system()` in **each of the
    block's 1024 threads**, i.e. one `CCTL.IVALL` per thread per call.  The CTA
    barrier that follows the stores already orders them (a CTA barrier is
    release-acquire at CTA scope and PTX cumulativity carries it), so one
    `red.add.release.sys` per block publishes exactly the same ordering.
    `source/k3_allgather4_v3.cu` is that kernel; main still runs
    `k3_allgather4`, which has neither this nor round 11's other half.
  * `l*.res_in` (`kern_k3_attnres_rms_push`) writes this rank's slice of
    `normed_all` into all four ranks' copies.  The three peer copies are lines in
    *our* L2 that exist only to be forwarded, and they were written with the
    default policy, so they evicted whatever the next kernels wanted.

Measured on main's default (main's manifest as the baseline, the two modules
repointed into it as the candidate), interleaved: default 598.868 / 598.777 /
598.919 ms against 598.177 / 598.244 ms -> **-0.62 ms**, with the ops themselves
`l*.gather_small` 103.1 -> 95.2 us per call and `l*.res_in` 278.1 -> 273.5.

Idempotent: re-running it only re-checks.
"""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
AG4 = "k3_allgather4_v3"
AG4_OLD = ("k3_allgather4", "k3_allgather4_v2")
PUSH = "k3_residual_push_v3"
PUSH_OPS = ("attnres_rms_push_v3", "attnres_rms_push_first_v3")
MARKERS = {AG4: ("red.add.release.sys",), PUSH: ("stv_peer", "__stcs")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or a.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    for name, markers in MARKERS.items():
        body = (ROOT / kernels[name]["source"]).read_text()
        for marker in markers:
            assert marker in body, f"{kernels[name]['source']} does not carry {marker}"
    m = json.loads(a.manifest.read_text())

    l = m["ops"]["k3_allgather4_push"]["impl"]["launches"][0]
    assert l["entry"] == "kern_k3_allgather4_push", l["entry"]
    assert l["module"] in AG4_OLD + (AG4,), l["module"]
    l["module"] = AG4

    for op in PUSH_OPS:
        l = m["ops"][op]["impl"]["launches"][0]
        assert l["entry"] == "kern_k3_attnres_rms_push", l["entry"]
        assert l["module"] in ("k3_residual_push", PUSH), l["module"]
        l["module"] = PUSH

    for name in MARKERS:
        m["modules"][name] = {"source": f"{name}.cubin", "sha256": kernels[name]["sha256"]}
    used = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    for name in [n for n in m["modules"] if n not in used]:
        del m["modules"][name]
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"peer path: {AG4} for gather_small, {PUSH} with streaming peer stores "
          f"({len(m['ops'])} ops, {len(m['modules'])} modules) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
