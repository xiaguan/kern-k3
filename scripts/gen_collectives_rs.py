#!/usr/bin/env python3
"""Candidate: the TP reduce-scatter runs as a peer-pointer pull + inline ring sum.

`l*.reduce_attn` is `nccl_reducescatter_bf16(attn_part, attn_own, rows*7168)` and
`l*.scatter_moe` is the same on `moe_partial` -> `routed_latent` (rows*3584): the
o_proj / MoE down-projection partials of all four ranks are summed and each rank
keeps its own quarter of the rows.  Together they are ~80 ms per rank, the two
largest items of the partition.

The first version of this generator pushed: `kern_k3_rs_push` moved this rank's
three quarters of the partial into the peers' staging slots and `kern_k3_rs_sum`
read that staging back afterwards, so the link traffic and the ~150 MB of local
read/modify/write were serialised behind one barrier and two launches.

This version pulls instead.  `kern_k3_rs_arrive` is the barrier (it moves no
data), and `kern_k3_rs_pull` reads the four partials of the chunk this rank keeps
-- three of them straight out of the peers' `src` over NVLink -- and lands the
same ring chain in the same kernel.  The link volume is unchanged (rank r reads
three quarters of the partial and serves its own three quarters to the peers),
the local staging buffer disappears, and the local traffic overlaps the link
traffic instead of following it.

`src` (attn_part / moe_partial) therefore gets exported and a peer address
buffer, `rs_stage` and `rs_stage_peer` are dropped, and both launches keep the
one grid expression every collective of a call shares -- the arrival counters
are one `coll_flags` table and the epoch scheme of source/k3_collectives.cu only
holds while all of them launch the same grid over `tokens`.

Idempotent: re-running it on an already converted manifest only re-checks the
pieces it adds.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULE = "k3_reducescatter_pull"
ARRIVE = "kern_k3_rs_arrive"
PULL = "kern_k3_rs_pull"
THREADS = 1024
FLAGS = "coll_flags"
SRC = ("attn_part", "moe_partial")
# one grid for every collective of a call (see the module docstring)
GRID = {"ceil_div": [{"mul": [{"ceil_div": ["tokens", 4]}, 7168]}, 8 * 4 * THREADS]}
OP = "k3_reducescatter_bf16"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    p.add_argument("--out", type=Path)
    p.add_argument("--order", type=int, default=6,
                   help="rs chain order: ownpos*2 + descend (see kern_k3_rs_pull)")
    args = p.parse_args()
    out = args.out or args.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    m = json.loads(args.manifest.read_text())
    calls = m["programs"]["prefill"]["calls"]
    tp = m["topology"]["groups"]["tp"]
    assert tp == 4, tp

    legacy = [c for c in calls if c["op"] == "nccl_reducescatter_bf16"]
    done = [c for c in calls if c["op"] == OP]
    n = len(legacy) + len(done)
    assert n, "no reduce-scatters to convert"
    assert "k3_epoch_bump" in m["ops"], "run scripts/gen_collectives_p2p.py first (epoch bump)"
    assert m["ops"]["k3_epoch_bump"]["impl"]["launches"][0]["grid"] == [GRID, 1, 1], \
        "the epoch bump and the collectives must share one grid expression"

    # ---- slots: the allgather script owns slots 1..n; the reduce-scatters take the
    # next block, renumbered from the call order every time so a replay is a no-op
    ag_slots = [c["args"][-1]["i32"] for c in calls if c["op"] == "k3_allgather_push"]
    assert len(set(ag_slots)) == len(ag_slots), "duplicate allgather slot"
    slot = max(ag_slots) if ag_slots else 0

    # ---- the op: src, dst, src_peer, flags, flags_peer, count, slot, order
    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}
    m["ops"][OP] = {
        "params": ["in buffer<bf16>", "out buffer<bf16>", "in buffer<u64>",
                   "out buffer<u64>", "in buffer<u64>", "i64", "i32", "i32"],
        "impl": {"launches": [
            {"module": MODULE, "entry": entry,
             "params": ["in buffer<bf16>", "out buffer<bf16>", "in buffer<u64>",
                        "out buffer<u64>", "in buffer<u64>", "i64", "i32", "i32",
                        "i32", "i32"],
             "args": [{"param": i} for i in range(8)] + [{"rank": "tp"}, {"i32": tp}],
             "block": [THREADS, 1, 1], "grid": [GRID, 1, 1]}
            for entry in (ARRIVE, PULL)]}}

    # ---- the pull reads the peers' src straight out of their workspace
    for buf in SRC:
        assert buf in m["buffers"], buf
        m["buffers"][buf]["export"] = True
        m["buffers"][buf + "_peer"] = {"dtype": "u64", "shape": [tp], "kind": "peer",
                                       "of": buf, "group": "tp"}
    for stale in ("rs_stage", "rs_stage_peer"):
        m["buffers"].pop(stale, None)

    # three shapes reach this loop: the reference's `nccl_reducescatter_bf16(src,
    # dst, count)`, the push form of this generator's first version (9 args, the
    # staging pair in the middle) and its own pull form (8 args).  All of them
    # carry src, dst and count; the rest is rebuilt from scratch every replay.
    for c in calls:
        if c["op"] not in ("nccl_reducescatter_bf16", OP):
            continue
        a = c["args"]
        assert len(a) in (3, 8, 9), a
        src, dst = a[0], a[1]
        counts = [x for x in a if "expr" in x]
        assert len(counts) == 1, a
        count = counts[0]
        assert src["buf"] in SRC, src
        slot += 1
        c["op"] = OP
        c["args"] = [src, dst, {"buf": src["buf"] + "_peer"},
                     {"buf": FLAGS}, {"buf": FLAGS + "_peer"}, count,
                     {"i32": slot}, {"i32": args.order}]

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
    print(f"{n} reduce-scatters pull on {MODULE}, slots 1..{size - 1}, "
          f"{len(m['ops'])} ops, {len(m['modules'])} modules -> {out}")


if __name__ == "__main__":
    main()
