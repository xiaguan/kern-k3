#!/usr/bin/env python3
"""Candidate: the arrival polls its peers from one thread each.

`kern_k3_rs_arrive1` (source/k3_rs_arrive1.cu) is the arrival of every
collective in this partition that has one -- the reduce-scatters' (185 calls) and
the payload-less `normed_sync` barrier behind the fused residual kernel (94).
One block, one warp: thread 0 bumps this rank's counter and then reads the four
peers' counters **one after the other**, each of which is a remote round trip to
a different rank, and block 0 of the old 896-block form had the same shape.
Measured in round 6, a barrier of this kind whose wait is *already satisfied*
still costs 17 us per call; four serialized remote reads are 4-6 us of that.

The wait is per peer, so it can be issued per peer: thread q polls peer q, and
`__syncwarp()` (the block is one warp) is the whole barrier.  Semantics are
unchanged -- every counter still has to reach the epoch before the kernel is
done -- and nothing about the arithmetic, the slots or the epoch moves.

Idempotent: re-running it on an already converted manifest only re-checks.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULE = "k3_rs_arrive_v2"
ENTRY = "kern_k3_rs_arrive1"
OLD = "k3_rs_arrive1"
# the two ops that launch the arrival
OPS = ("k3_reducescatter_bf16", "k3_allgather_push")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or args.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    m = json.loads(args.manifest.read_text())
    hits = 0
    for name in OPS:
        assert name in m["ops"], name
        for l in m["ops"][name]["impl"]["launches"]:
            if l["entry"] != ENTRY:
                continue
            assert l["module"] in (OLD, MODULE), l["module"]
            assert l["block"] == [32, 1, 1] and l["grid"] == [1, 1, 1], (l["block"], l["grid"])
            hits += 1
    assert hits == len(OPS), f"{hits} arrival launches, expected {len(OPS)}"
    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}
    for name in OPS:
        for l in m["ops"][name]["impl"]["launches"]:
            if l["entry"] == ENTRY:
                l["module"] = MODULE
    used = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    if OLD not in used:
        m["modules"].pop(OLD, None)
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"arrival of {len(OPS)} ops on {MODULE} (one polling thread per peer), "
          f"{len(m['ops'])} ops, {len(m['modules'])} modules -> {out}")


if __name__ == "__main__":
    main()
