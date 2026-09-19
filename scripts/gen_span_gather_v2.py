#!/usr/bin/env python3
"""Derive the span_gather-v2 candidate manifest from a manifest that uses v1.

Moves the K9 span kernel (`span_gather`, 69 calls, one per KDA layer) from
k3_span_gather+HEADS=24 (grid (6, 4, ceil(tokens/8)), block 128) to
k3_span_gather_v2+HEADS=24 (grid (6, 4, ceil(tokens/16)), block 128; same ABI
and arguments; the v2 kernel computes the SiLU with the fast-math ex2/rcp
forms, gives each block 16 rows and loads them in batches of four rows).
The new op is `span_gather_v2` because `kern test` counts a call whose op
name differs as a changed call; `--v1-layers A-B` keeps the calls of layers
A..B on the v1 op, for intermediate manifests that split a validation into
hops (see gen_residual_v2.py).  Unreferenced ops and modules are dropped.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OP = "span_gather"
OLD, NEW = "k3_span_gather+HEADS=24", "k3_span_gather_v2+HEADS=24"
SUFFIX = "_v2"
INNER, VEC, ROWS, THREADS = 24 * 128, 4, 16, 128


def layer_of(label):
    head = label.split(".")[0]
    return int(head[1:]) if head.startswith("l") and head[1:].isdigit() else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path, help="a manifest whose span_gather calls use v1")
    parser.add_argument("--out", default=ROOT / "manifests/k3-tp4-prefill-16k.json", type=Path)
    parser.add_argument("--v1-layers", metavar="A-B", help="keep layers A..B on the v1 op (hop manifests)")
    args = parser.parse_args()
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    m = json.loads(args.reference.read_text())
    v2 = json.loads(json.dumps(m["ops"][OP]))
    launch = v2["impl"]["launches"][0]
    assert launch["module"] == OLD and launch["block"] == [THREADS, 1, 1], launch
    assert launch["grid"] == [INNER // (THREADS * 4), 4, {"ceil_div": ["tokens", 8]}], launch
    launch["module"] = NEW
    launch["grid"] = [INNER // (THREADS * VEC), 4, {"ceil_div": ["tokens", ROWS]}]
    m["ops"][OP + SUFFIX] = v2
    m["modules"][NEW] = {"source": f"{NEW}.cubin", "sha256": kernels[NEW]["sha256"]}
    keep_v1 = range(0)
    if args.v1_layers:
        lo, hi = (int(x) for x in args.v1_layers.split("-"))
        keep_v1 = range(lo, hi + 1)
    moved = 0
    for call in m["programs"]["prefill"]["calls"]:
        if call["op"] == OP and layer_of(call["label"]) not in keep_v1:
            call["op"] += SUFFIX
            moved += 1
    called = {c["op"] for prog in m["programs"].values() for c in prog["calls"]}
    for op in [op for op in m["ops"] if op not in called]:
        del m["ops"][op]
    still_used = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    if OLD not in still_used:
        del m["modules"][OLD]
    print(f"{moved} calls moved to {OP}{SUFFIX}, {len(m['ops'])} ops, {len(m['modules'])} modules")
    args.out.write_text(json.dumps(m, indent=1) + "\n")
    print(args.out.name)


if __name__ == "__main__":
    main()
