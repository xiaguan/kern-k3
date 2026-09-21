#!/usr/bin/env python3
"""Candidate: the TP allgather of `normed` runs as a peer-pointer push kernel.

`l*.gather_normed` is `nccl_allgather_bf16(normed, normed_all, rows*7168)`: every
rank contributes its own 4096 rows of `normed` and every rank ends up holding all
16384.  `kern_allgather_push` does the same thing without NCCL: the runtime maps
each rank's copy of an `export`ed buffer and hands a `peer` buffer (u64[tp]) of
their device addresses to the kernel, so each rank writes its own slice straight
into every peer's `normed_all`.  A rank cannot run past the call while a peer's
slice is in flight, so every block publishes its arrival in a per-collective
counter (`coll_flags`) and the kernel waits for all peers' counters before it
completes; the counters are zeroed once per program replay.

Nothing else changes: same buffers, same call sites, same dtype and element
count.  Idempotent: re-running it on an already converted manifest only
re-checks the pieces it adds.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULE = "k3_collectives"
SRC, DST = "normed", "normed_all"
VPT = 4              # vectors of 16 B per thread, must match k3_collectives.cu
THREADS = 1024


def grid_expr(count):
    """ceil(count / 8 / VPT / THREADS), written over the `tokens` var."""
    return {"ceil_div": [count, 8 * VPT * THREADS]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or args.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    m = json.loads(args.manifest.read_text())
    sha = kernels[MODULE]["sha256"]
    calls = m["programs"]["prefill"]["calls"]

    pushes = [c for c in calls if c["op"] == "nccl_allgather_bf16" and c["args"][0].get("buf") == SRC]
    legacy = [c for c in calls if c["op"] == "k3_allgather_push"]
    n = len(pushes) + len(legacy)
    assert n, "no normed allgathers to convert"

    tp = m["topology"]["groups"]["tp"]
    assert tp == 4, tp
    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": sha}
    m["ops"]["k3_epoch_bump"] = {
        "params": ["out buffer<u64>"],
        "impl": {"launches": [{"module": MODULE, "entry": "kern_k3_epoch_bump",
                               "block": [1024, 1, 1],
                               "grid": [grid_expr({"mul": [{"ceil_div": ["tokens", 4]}, 7168]}), 1, 1]}]},
    }
    m["ops"]["k3_flags_init"] = {
        "params": ["out buffer<u64>", "i32"],
        "impl": {"launches": [{"module": MODULE, "entry": "kern_k3_flags_init",
                               "block": [256, 1, 1], "grid": [1, 1, 1]}]},
    }
    m["ops"]["k3_allgather_push"] = {
        "params": ["in buffer<bf16>", "out buffer<bf16>", "in buffer<u64>",
                   "out buffer<u64>", "in buffer<u64>", "i64", "i32"],
        "impl": {"launches": [{
            "module": MODULE, "entry": "kern_k3_allgather_push",
            "params": ["in buffer<bf16>", "out buffer<bf16>", "in buffer<u64>",
                       "out buffer<u64>", "in buffer<u64>", "i64", "i32", "i32", "i32"],
            "args": [{"param": i} for i in range(7)] + [{"rank": "tp"}, {"i32": tp}],
            "block": [THREADS, 1, 1],
            "grid": [grid_expr({"mul": [{"ceil_div": ["tokens", 4]}, 7168]}), 1, 1],
        }]},
    }
    m["buffers"][DST]["export"] = True
    # grow only: scripts/gen_collectives_rs.py adds slots to the same table, and
    # either script may be replayed on a manifest the other has already touched
    nflag = max(m["buffers"].get("coll_flags", {}).get("shape", [0])[0], n + 1)
    m["buffers"]["coll_flags"] = {"dtype": "u64", "shape": [nflag], "kind": "workspace",
                                  "export": True}
    m["buffers"]["coll_flags_peer"] = {"dtype": "u64", "shape": [tp], "kind": "peer",
                                       "of": "coll_flags", "group": "tp"}
    m["buffers"][DST + "_peer"] = {"dtype": "u64", "shape": [tp], "kind": "peer",
                                   "of": DST, "group": "tp"}

    slot = 0
    for c in calls:
        if c["op"] == "nccl_allgather_bf16" and c["args"][0].get("buf") == SRC:
            assert c["args"][1].get("buf") == DST, c
            slot += 1
            c["op"] = "k3_allgather_push"
            c["args"] = [c["args"][0], c["args"][1], {"buf": DST + "_peer"},
                         {"buf": "coll_flags"}, {"buf": "coll_flags_peer"},
                         c["args"][2], {"i32": slot}]
        elif c["op"] == "k3_allgather_push":
            slot += 1
            c["args"][6] = {"i32": slot}

    init = {"label": "coll_init", "op": "k3_flags_init",
            "args": [{"buf": "coll_flags"}, {"i32": nflag}]}
    load_calls = m["programs"]["load"]["calls"]        # in place, so re-running is a no-op
    load_calls[:] = [c for c in load_calls if c["op"] != "k3_flags_init"]
    load_calls.insert(0, init)
    bump = {"label": "coll_epoch", "op": "k3_epoch_bump", "args": [{"buf": "coll_flags"}]}
    for prog in m["programs"].values():   # in place: `calls` stays bound to prefill
        prog["calls"][:] = [c for c in prog["calls"] if c["op"] != "k3_epoch_bump"]
    calls.insert(0, bump)
    assert calls is m["programs"]["prefill"]["calls"] and calls[0]["op"] == "k3_epoch_bump"

    called = {c["op"] for prog in m["programs"].values() for c in prog["calls"]}
    for op in [o for o in m["ops"] if o not in called]:
        del m["ops"][op]
    used = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    for mod in [x for x in m["modules"] if x not in used]:
        del m["modules"][mod]
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"{n} allgathers on {MODULE}, flags[0..{n}], {len(m['ops'])} ops, "
          f"{len(m['modules'])} modules -> {out}")


if __name__ == "__main__":
    main()
