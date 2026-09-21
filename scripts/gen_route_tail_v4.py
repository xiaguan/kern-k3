#!/usr/bin/env python3
"""Candidate: the routing tail runs the restructured tables and a wider scatter block.

Two kernels of `source/k3_moe_prefill_v4.cu`, both measured bit-identical against the
shipped ones before they were timed (the task book's probe rule):

* `moe_route_tables` -> the restructured kernel.  Round 19 decomposed its 11.2 us into
  2.0 launch + 0.8 count pass + 0.35 cta fill + ~8 spread over a single-thread 56-step
  scan with a software integer division, a per-expert write loop and the padding stores.
  The new kernel sums the counts first and folds the expert offset into one write loop
  (no read-modify-write of `blockoff`) and makes the tile a compile-time constant
  (`-DTILE_SHIFT=7`), so the scan divides by a shift: 11.18 -> 6.80 us, all six output
  buffers byte-identical (`round20-route-fold/scripts/probe_fold.cu`).
* `moe_route_scatter` -> block 256 -> 512 threads with `-DSCATTER_WARPS=16`, which is
  the same staged work spread over twice the warps (ROUTE_GROUPS/16 = 8 rounds instead
  of 16).  Round 13's ncu measured this kernel at 2.02 active warps per scheduler with
  85.6% of cycles having no eligible warp: it is empty, not busy.

usage: gen_route_tail_v4.py [OUT.json] [IN.json]; with no arguments it rewrites the
default manifest in place.  Idempotent.  Replay contract: no arguments rewrites the
default in place, is idempotent, and this branch's generators may be replayed in any
order.
"""
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "manifests/k3-tp4-prefill-16k.json").is_file() else Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "manifests/k3-tp4-prefill-16k.json"
OLD_MOD = "k3_moe_prefill_v3+EP_STRIDED=1"
NEW_MOD = "k3_moe_prefill_v4+EP_RANKS=4+EP_STRIDED=1+TILE_SHIFT=7+SCATTER_WARPS=16"
TABLES, SCATTER = "moe_route_tables", "moe_route_scatter"


def without_target(doc):
    d = json.loads(json.dumps(doc))
    d["ops"][TABLES]["impl"]["launches"][0]["module"] = OLD_MOD
    sc = d["ops"][SCATTER]["impl"]["launches"][0]
    sc["module"] = OLD_MOD
    sc["block"] = [256, 1, 1]
    d["modules"].pop(NEW_MOD, None)
    return d


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT
    m = json.loads(src.read_text())
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    assert kernels[NEW_MOD]["source"] == "source/k3_moe_prefill_v4.cu", kernels[NEW_MOD]
    assert kernels[NEW_MOD]["defines"] == {"EP_RANKS": 4, "EP_STRIDED": 1, "TILE_SHIFT": 7,
                                          "SCATTER_WARPS": 16}, kernels[NEW_MOD]
    if m["ops"][TABLES]["impl"]["launches"][0]["module"] == NEW_MOD:
        print(f"{out}: the routing tail is already on {NEW_MOD}; nothing to do")
        return
    before = json.loads(json.dumps(m))
    tab = m["ops"][TABLES]["impl"]["launches"][0]
    sca = m["ops"][SCATTER]["impl"]["launches"][0]
    assert tab["module"] == OLD_MOD and sca["module"] == OLD_MOD, (tab["module"], sca["module"])
    assert sca["block"] == [256, 1, 1], sca["block"]
    m["modules"][NEW_MOD] = {"source": f"{NEW_MOD}.cubin", "sha256": kernels[NEW_MOD]["sha256"]}
    tab["module"] = NEW_MOD
    sca["module"] = NEW_MOD
    sca["block"] = [512, 1, 1]
    live = {l.get("module") for op in m["ops"].values() for l in op["impl"].get("launches", [])}
    assert set(m["modules"]) <= live, sorted(set(m["modules"]) - live)
    assert without_target(before) == without_target(m), "more than the routing tail changed"
    out.write_text(json.dumps(m, indent=1))
    print(f"{out}: {TABLES} and {SCATTER} -> {NEW_MOD}, scatter block 512")


main()
