#!/usr/bin/env python3
"""Candidate: the residual kernel that computes `normed` writes it to all four ranks.

`l*.res_in` (`attnres_rms_v2` / `attnres_rms_first_v2`, kernel
`kern_k3_attnres_rms`) produces this rank's 4096 rows of 7168 bf16 `normed`, and
the `l*.gather_normed` call right behind it (`k3_allgather_push`) reads those
58.7 MB back and stores them into all four ranks' `normed_all`: 176 MB of NVLink
egress per layer.  Measured per call per rank they are 97 us + 283 us, and each
is far from what limits the other -- the residual kernel is HBM-bound at
~2.4 TB/s, the push is link-bound at ~620 GB/s of egress.

This generator folds the push into the kernel that already holds the values:
`kern_k3_attnres_rms_push` (source/k3_residual_push.cu, module
`k3_residual_push`) is `kern_k3_attnres_rms` with the last store redirected, so
each thread lands its normed vector in this rank's slice of `normed_all` and in
the three peers' slices.  The arithmetic is untouched -- same candidate loop,
same fixed-order reductions, same single bf16 landing.

What is left of the allgather is its barrier.  An allgather needs every peer's
slice to have landed before the consumer runs, and a block that spins inside the
residual kernel could never retire, so the blocks that still have to arrive
could never be scheduled (source/k3_collectives.cu has the long version).  The
`l*.gather_normed` call therefore keeps its shape -- same op, same buffers, same
`coll_flags` slot, same grid, which is the one expression every collective of a
call shares and which the arrival counters depend on -- but its payload drops to
zero bytes and its block to 32 threads: it is stream-ordered after the residual
kernel, so its arrival still means "this rank's slice is written and
system-visible".  Its `src` argument becomes `normed_all` for the same reason
the fused kernel writes it: the call no longer reads anything, and naming the
buffer that the write actually landed in keeps the manifest's dataflow honest.

Idempotent: re-running it on an already converted manifest only re-checks the
pieces it adds.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULE = "k3_residual_push"
ENTRY = "kern_k3_attnres_rms_push"
OP = "attnres_rms_push_v3"
OP_FIRST = "attnres_rms_push_first_v3"
OLD = ("attnres_rms_v2", "attnres_rms_first_v2")
PUSH = "k3_allgather_push"
FLAGS = "coll_flags"
THREADS = 128
BARRIER_THREADS = 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or args.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    m = json.loads(args.manifest.read_text())
    calls = m["programs"]["prefill"]["calls"]
    tp = m["topology"]["groups"]["tp"]
    assert tp == 4, tp

    assert PUSH in m["ops"], "run scripts/gen_collectives_p2p.py first (the normed barrier)"
    for buf in ("normed", "normed_all", "normed_all_peer", FLAGS):
        assert buf in m["buffers"], buf
    launch = m["ops"][PUSH]["impl"]["launches"][0]
    grid = launch["grid"]

    # ---- the fused op: one row per block, grid (B, 1, 1), the residual geometry.
    # The kernel signature is prefix, blocks, sw, gamma, normed_all, dst_peer,
    # nb, snapshot, B, rank, nranks.
    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}
    sig = ["in buffer<bf16>", "inout buffer<bf16>", "in buffer<f32>", "in buffer<bf16>",
           "out buffer<bf16>", "in buffer<u64>", "i32", "i32", "i32", "i32", "i32"]
    m["ops"][OP] = {
        "params": sig[:9],
        "impl": {"launches": [
            {"module": MODULE, "entry": ENTRY, "params": sig,
             "args": [{"param": i} for i in range(9)] + [{"rank": "tp"}, {"i32": tp}],
             "block": [THREADS, 1, 1], "grid": [{"ceil_div": ["tokens", 4]}, 1, 1]}]}}
    # l0.res_in is the one call that snapshots the prefix into `blocks` and has
    # no candidate to read, so its interface says `blocks` is written -- the same
    # reason the original split attnres_rms_first_v2 off attnres_rms_v2.
    first = [sig[0], "out buffer<bf16>", sig[2], sig[3], sig[4], sig[5], sig[8]]
    # the launch's parameter list has to agree with the interface it is reached
    # through: this entry is declared with `blocks` as an out buffer
    first_sig = [sig[0], "out buffer<bf16>"] + sig[2:]
    m["ops"][OP_FIRST] = {
        "params": first,
        "impl": {"launches": [
            {"module": MODULE, "entry": ENTRY, "params": first_sig,
             "args": [{"param": i} for i in range(6)] + [{"i32": 0}, {"i32": 1},
                                                         {"param": 6}, {"rank": "tp"},
                                                         {"i32": tp}],
             "block": [THREADS, 1, 1], "grid": [{"ceil_div": ["tokens", 4]}, 1, 1]}]}}

    # ---- the residual calls move to it; `normed` stops being their output
    n = 0
    for c in calls:
        if c["op"] not in OLD and c["op"] not in (OP, OP_FIRST):
            continue
        if c["op"] in OLD:
            a = c["args"]
            head = [a[0], a[1], a[2], a[3], {"buf": "normed_all"}, {"buf": "normed_all_peer"}]
            if c["op"] == "attnres_rms_v2":
                assert len(a) == 8, a
                c["op"], c["args"] = OP, head + [a[5], a[6], a[7]]
            else:
                assert len(a) == 6, a
                c["op"], c["args"] = OP_FIRST, head + [a[5]]
        else:
            assert len(c["args"]) in (7, 9), c["args"]
        n += 1
    assert n, "no residual calls to convert"

    # ---- and their barrier loses its payload: same call, same slot, same grid
    b = 0
    for c in calls:
        if c["op"] != PUSH:
            continue
        a = c["args"]
        assert len(a) == 7, a
        c["label"] = c["label"].split(".")[0] + ".normed_sync"
        c["args"][0] = {"buf": "normed_all"}      # what the write actually landed in
        c["args"][-2] = {"i64": 0}                # ... but nothing goes through it
        b += 1
    assert b, "no normed barriers to convert"
    assert b == n, f"{n} residual writes but {b} barriers"
    launch["block"] = [BARRIER_THREADS, 1, 1]

    called = {c["op"] for prog in m["programs"].values() for c in prog["calls"]}
    for op in [o for o in m["ops"] if o not in called]:
        del m["ops"][op]
    used_mod = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    for mod in [x for x in m["modules"] if x not in used_mod]:
        del m["modules"][mod]
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"{n} residual writes land in normed_all via {MODULE}, {b} payload-less barriers "
          f"(grid {json.dumps(grid)}), {len(m['ops'])} ops, {len(m['modules'])} modules -> {out}")


if __name__ == "__main__":
    main()
