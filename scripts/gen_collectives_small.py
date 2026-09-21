#!/usr/bin/env python3
"""Candidate: the four small TP allgathers of a layer run as one push kernel.

`l*.gather_latent_q`, `l*.gather_latent_sf`, `l*.gather_topk_idx` and
`l*.gather_topk_weight` are four `nccl_allgather_{u8,u8,i32,f32}` calls of 15.7 MB
of payload per rank in total, ~190 us per layer per rank, because each call pays
~25 us of fixed cost on top of its own bandwidth.  They are the same collective
as `normed`, so they move to the peer-pointer scheme of
scripts/gen_collectives_p2p.py: `kern_k3_allgather4_push`
(source/k3_allgather4.cu) writes this rank's slice of all four buffers into every
rank's copy of the gathered buffer with one barrier per call instead of four.
The four payloads are byte streams (every count is a multiple of 16), so one
launch covers all of them; the `*_all` layouts are unchanged.

The launch keeps the grid expression every other collective of a call uses --
the arrival counters are the same `coll_flags` table and the epoch scheme of
source/k3_collectives.cu only holds while all of them launch the same grid over
`tokens` -- and takes a new slot after the ones the other two generators own.
Idempotent: re-running it on an already converted manifest only re-checks the
pieces it adds.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULE = "k3_allgather4"
ENTRY = "kern_k3_allgather4_push"
FLAGS = "coll_flags"
THREADS = 1024
# one grid for every collective of a call (see source/k3_collectives.cu)
GRID = {"ceil_div": [{"mul": [{"ceil_div": ["tokens", 4]}, 7168]}, 8 * 4 * THREADS]}

# name -> (element bytes, elements per row) of the four payloads, in the order
# the prefill program calls them
PART = [("latent_q", 1, 3584), ("latent_sf", 1, 112),
        ("topk_idx", 4, 16), ("topk_weight", 4, 16)]
DTYPE = {"latent_q": "u8", "latent_sf": "u8", "topk_idx": "i32", "topk_weight": "f32"}
LEGACY = {"latent_q": "nccl_allgather_u8", "latent_sf": "nccl_allgather_u8",
          "topk_idx": "nccl_allgather_i32", "topk_weight": "nccl_allgather_f32"}


def rows():
    return {"ceil_div": ["tokens", 4]}


def op_def():
    params, args = [], []
    for name, esize, per_row in PART:
        dt = DTYPE[name]
        params += [f"in buffer<{dt}>", f"out buffer<{dt}>", "in buffer<u64>", "i64"]
        args += [{"param": len(params) - 4}, {"param": len(params) - 3},
                 {"param": len(params) - 2}, {"param": len(params) - 1}]
    params += ["out buffer<u64>", "in buffer<u64>", "i32"]
    args += [{"param": len(params) - 3}, {"param": len(params) - 2}, {"param": len(params) - 1}]
    launch_args = args + [{"rank": "tp"}, {"i32": 4}]   # rank and nranks are launches' own
    return {"params": params,
            "impl": {"launches": [{"module": MODULE, "entry": ENTRY,
                                   "params": params + ["i32", "i32"],
                                   "args": launch_args, "block": [THREADS, 1, 1],
                                   "grid": [GRID, 1, 1]}]}}


def call_args(layer, slot):
    out = []
    for name, esize, per_row in PART:
        out += [{"buf": name}, {"buf": name + "_all"}, {"buf": name + "_all_peer"},
                {"expr": {"mul": [rows(), per_row * esize]}}]
    out += [{"buf": FLAGS}, {"buf": FLAGS + "_peer"}, {"i32": slot}]
    return out


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
    assert "k3_epoch_bump" in m["ops"], "run scripts/gen_collectives_p2p.py first (epoch bump)"
    assert "k3_reducescatter_bf16" in m["ops"], "run scripts/gen_collectives_rs.py first (slots)"
    assert m["ops"]["k3_epoch_bump"]["impl"]["launches"][0]["grid"] == [GRID, 1, 1], \
        "the epoch bump and the collectives must share one grid expression"

    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}
    m["ops"]["k3_allgather4_push"] = op_def()
    for name, _, _ in PART:
        m["buffers"][name + "_all"]["export"] = True
        m["buffers"][name + "_all_peer"] = {"dtype": "u64", "shape": [tp], "kind": "peer",
                                            "of": name + "_all", "group": "tp"}

    # ---- slots: the allgather script owns 1..n and the reduce-scatters the next
    # block; the four-way gathers take the ones after them.  Assigned from the
    # call order once, kept afterwards (a replay recomputes the same numbers).
    taken = [c["args"][-1 if c["op"] != "k3_reducescatter_bf16" else -2]["i32"] for c in calls
             if c["op"] in ("k3_allgather_push", "k3_reducescatter_bf16", "k3_allgather4_push")]
    slot = max(taken) if taken else 0
    n_new = 0
    i = 0
    while i < len(calls):
        c = calls[i]
        name = None
        if c["op"] == "k3_allgather4_push":
            n_new += 1
            i += 1
            continue
        for cand, _, _ in PART:
            if c["op"] == LEGACY[cand] and c["args"][0].get("buf") == cand:
                name = cand
        if name is None:
            i += 1
            continue
        # the four calls of one layer are consecutive in the program
        quad = [(PART[0])]
        block = []
        for cand, _, _ in PART:
            j = i + len(block)
            assert j < len(calls), "truncated gather group"
            cj = calls[j]
            assert cj["op"] == LEGACY[cand], cj
            assert cj["args"][0].get("buf") == cand, cj
            assert cj["args"][1].get("buf") == cand + "_all", cj
            block.append(cj)
        slot += 1
        layer = c["label"].split(".")[0]
        calls[i:i + 4] = [{"label": f"{layer}.gather_small", "op": "k3_allgather4_push",
                           "args": call_args(layer, slot)}]
        n_new += 1
        i += 1
    assert n_new, "no four-way gather groups to convert"
    assert len(set(taken)) == len(taken), "duplicate collective slot"

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
    used = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    for mod in [x for x in m["modules"] if x not in used]:
        del m["modules"][mod]
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"{n_new} four-way gathers on {MODULE}, slots ..{slot}, flags {size}, "
          f"{len(m['ops'])} ops, {len(m['modules'])} modules -> {out}")


if __name__ == "__main__":
    main()
