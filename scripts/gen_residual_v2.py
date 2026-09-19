#!/usr/bin/env python3
"""Derive the residual-v2 candidate manifest from the v1 (reference) manifest.

Moves the K1 residual family (attnres_rms, attnres_rms_first,
land_add_attnres_rms_bf16, land_add2) from k3_residual (1024-thread blocks) to
k3_residual_v2 (128-thread blocks, same ABI, same grid, same arguments).  The
new ops carry a `_v2` suffix because `kern test` treats a call whose op name
differs as a changed call: with `--v1-layers A-B` the calls of layers A..B (and
`out.res` when B is the last layer) keep the reference's ops and names, which
gives an intermediate manifest that differs from the reference only on the
other layers, and from the full candidate only on layers A..B.  The runner
keeps every changed span's outputs on the device and runs out of memory with
all 187 spans at once, so the change is validated in two such hops.
Unreferenced ops and modules are dropped; the v1 modules stay in kernels.toml
and build/ for the reference manifest.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
BLOCK = [128, 1, 1]
MOVES = {
    "attnres_rms": ("k3_residual", "k3_residual_v2"),
    "attnres_rms_first": ("k3_residual", "k3_residual_v2"),
    "land_add2": ("k3_residual", "k3_residual_v2"),
    "land_add_attnres_rms_bf16": ("k3_residual+LAND_BF16=1", "k3_residual_v2+LAND_BF16=1"),
}
SUFFIX = "_v2"


def layer_of(label, last):
    head = label.split(".")[0]
    if head.startswith("l") and head[1:].isdigit():
        return int(head[1:])
    return last if label.startswith("out.") else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path,
                        help="a v1 manifest, e.g. the initial reference or `git show <pre-2026-09-19>:manifests/k3-tp4-prefill-16k.json`")
    parser.add_argument("--out", default=ROOT / "manifests/k3-tp4-prefill-16k.json", type=Path)
    parser.add_argument("--v1-layers", metavar="A-B", help="keep layers A..B (and 'out.res' if B is the last layer) "
                        "on the v1 kernels: an intermediate manifest so `kern test` can validate the change in hops "
                        "(the runner keeps every changed span's outputs on the device and runs out of memory with "
                        "all 187 spans at once)")
    args = parser.parse_args()
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    m = json.loads(args.reference.read_text())
    ref = json.loads(json.dumps(m))
    old_modules = set()
    for op, (old, new) in MOVES.items():
        v2 = json.loads(json.dumps(m["ops"][op]))
        launch = v2["impl"]["launches"][0]
        assert launch["module"] == old, (op, launch["module"])
        assert launch["block"] == [1024, 1, 1], (op, launch["block"])
        launch["module"] = new
        launch["block"] = list(BLOCK)
        m["ops"][op + SUFFIX] = v2
        old_modules.add(old)
        m["modules"][new] = {"source": f"{new}.cubin", "sha256": kernels[new]["sha256"]}
    calls = m["programs"]["prefill"]["calls"]
    last = max(layer_of(c["label"], -1) for c in calls)
    keep_v1 = range(0)
    if args.v1_layers:
        lo, hi = (int(x) for x in args.v1_layers.split("-"))
        keep_v1 = range(lo, hi + 1)
    moved = 0
    for call in calls:
        if call["op"] in MOVES and layer_of(call["label"], last) not in keep_v1:
            call["op"] += SUFFIX
            moved += 1
    called = {c["op"] for prog in m["programs"].values() for c in prog["calls"]}
    for op in [op for op in m["ops"] if op not in called]:
        del m["ops"][op]
    still_used = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    for old in old_modules:
        if old not in still_used:
            del m["modules"][old]
    print(f"{moved} calls moved to {SUFFIX} ops, {len(m['ops'])} ops, {len(m['modules'])} modules")
    args.out.write_text(json.dumps(m, indent=1) + "\n")
    print(args.out.name)


if __name__ == "__main__":
    main()
