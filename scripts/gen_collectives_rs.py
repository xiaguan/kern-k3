#!/usr/bin/env python3
"""Candidate: the TP reduce-scatter runs as a peer-pointer push + local sum.

`l*.reduce_attn` is `nccl_reducescatter_bf16(attn_part, attn_own, rows*7168)` and
`l*.scatter_moe` is the same on `moe_partial` -> `routed_latent` (rows*3584): the
o_proj / MoE down-projection partials of all four ranks are summed and each rank
keeps its own quarter of the rows.  Together they are 96 ms per rank, at 323 and
177 GB/s of egress per rank, where the link bound is ~900.

`kern_k3_rs_push` moves chunk c of the local partial into peer c's staging slot
`rank`; `kern_k3_rs_sum` then adds this rank's own chunk `rank` to the three
staged ones in f32 and lands bf16.  Each rank moves exactly (nranks-1)/nranks of
the partial and receives the same, with one barrier per call instead of the
nranks-1 sequential steps of a ring.  The two launches are the two launches of
one op, so the call site keeps its shape.

The barrier is the one adopted for the allgather: every block fences its stores
and bumps its own rank's `coll_flags[slot]`, block 0 waits until each peer's
counter reaches `coll_flags[0]`, and that epoch accumulates `gridDim.x` once per
rank per program call.  That only works if *every* collective of a call launches
the same number of blocks, so all of them -- the allgather, both reduce-scatters
and the epoch bump -- share one grid expression over `tokens`.

Idempotent: re-running it on an already converted manifest only re-checks the
pieces it adds.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULE = "k3_collectives"
VPT = 4              # vectors of 16 B per thread in the kernels
THREADS = 1024
STAGE = "rs_stage"
FLAGS = "coll_flags"
# one grid for every collective of a call (see the module docstring)
GRID = {"ceil_div": [{"mul": [{"ceil_div": ["tokens", 4]}, 7168]}, 8 * VPT * THREADS]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    p.add_argument("--out", type=Path)
    p.add_argument("--order", type=int, default=6,
                   help="rs_sum accumulation order: ownpos*2 + descend (see the kernel)")
    args = p.parse_args()
    out = args.out or args.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    m = json.loads(args.manifest.read_text())
    calls = m["programs"]["prefill"]["calls"]
    tp = m["topology"]["groups"]["tp"]
    assert tp == 4, tp

    legacy = [c for c in calls if c["op"] == "nccl_reducescatter_bf16"]
    done = [c for c in calls if c["op"] == "k3_reducescatter_bf16"]
    n = len(legacy) + len(done)
    assert n, "no reduce-scatters to convert"
    assert "k3_epoch_bump" in m["ops"], "run scripts/gen_collectives_p2p.py first (epoch bump)"
    assert m["ops"]["k3_epoch_bump"]["impl"]["launches"][0]["grid"] == [GRID, 1, 1], \
        "the epoch bump and the collectives must share one grid expression"

    # ---- slots: the collectives already in the manifest keep theirs; new ones continue
    # the allgather script owns slots 1..n; the reduce-scatters take the next
    # block, renumbered from the call order every time so a replay is a no-op
    ag_slots = [c["args"][-1]["i32"] for c in calls if c["op"] == "k3_allgather_push"]
    assert len(set(ag_slots)) == len(ag_slots), "duplicate allgather slot"
    slot = max(ag_slots) if ag_slots else 0

    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}
    m["ops"]["k3_reducescatter_bf16"] = {
        "params": ["in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>",
                   "in buffer<u64>", "out buffer<u64>", "in buffer<u64>", "i64", "i32", "i32"],
        "impl": {"launches": [
            {"module": MODULE, "entry": "kern_k3_rs_push",
             "params": ["in buffer<bf16>", "out buffer<bf16>", "in buffer<u64>",
                        "out buffer<u64>", "in buffer<u64>", "i64", "i32", "i32", "i32"],
             "args": [{"param": 0}, {"param": 2}, {"param": 3}, {"param": 4}, {"param": 5},
                      {"param": 6}, {"param": 7}, {"rank": "tp"}, {"i32": tp}],
             "block": [THREADS, 1, 1], "grid": [GRID, 1, 1]},
            {"module": MODULE, "entry": "kern_k3_rs_sum",
             "params": ["in buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>",
                        "i64", "i32", "i32", "i32"],
             "args": [{"param": 0}, {"param": 1}, {"param": 2}, {"param": 6},
                      {"rank": "tp"}, {"i32": tp}, {"param": 8}],
             "block": [THREADS, 1, 1], "grid": [GRID, 1, 1]},
        ]},
    }
    m["buffers"][STAGE] = {"dtype": "bf16", "shape": [16384, 7168], "kind": "workspace",
                           "export": True}
    m["buffers"][STAGE + "_peer"] = {"dtype": "u64", "shape": [tp], "kind": "peer",
                                     "of": STAGE, "group": "tp"}

    for c in calls:
        if c["op"] == "nccl_reducescatter_bf16":
            slot += 1
            src, dst, count = c["args"]
            assert src["buf"] in ("attn_part", "moe_partial"), src
            assert "expr" in count, count
            c["op"] = "k3_reducescatter_bf16"
            c["args"] = [src, dst, {"buf": STAGE}, {"buf": STAGE + "_peer"},
                         {"buf": FLAGS}, {"buf": FLAGS + "_peer"}, count, {"i32": slot},
                         {"i32": args.order}]
        elif c["op"] == "k3_reducescatter_bf16":
            slot += 1
            c["args"][-2] = {"i32": slot}
            c["args"][-1] = {"i32": args.order}

    # ---- counters: grow only, so either generator can be replayed in any order
    size = max(m["buffers"][FLAGS]["shape"][0], slot + 1)
    m["buffers"][FLAGS]["shape"] = [size]
    for prog in m["programs"].values():
        for c in prog["calls"]:
            if c["op"] == "k3_flags_init":
                c["args"][1] = {"i32": size}

    called = {c["op"] for prog in m["programs"].values() for c in prog["calls"]}
    for op in [o for o in m["ops"] if o not in called]:
        del m["ops"][op]
    used_mod = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    for mod in [x for x in m["modules"] if x not in used_mod]:
        del m["modules"][mod]
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"{n} reduce-scatters on {MODULE}, slots 1..{size - 1}, "
          f"{len(m['ops'])} ops, {len(m['modules'])} modules -> {out}")


if __name__ == "__main__":
    main()
