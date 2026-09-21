#!/usr/bin/env python3
"""Candidate: a collective's arrival is one block, not the whole grid.

Two calls in this partition exist only to arrive and wait.  `kern_k3_rs_arrive`
(the arrival half of the reduce-scatter pull) and `kern_k3_allgather_push` used
as a payload-less barrier behind the fused residual kernel (`l*.normed_sync`,
since round 5) both fence, bump this rank's `coll_flags[slot]` once per block,
and have block 0 wait for every peer.  Both are launched with the collective
grid -- 896 blocks, 1024 threads for one and 32 for the other -- because
`coll_flags[0]` is bumped by `gridDim.x` once per rank per program call and every
collective of a call has to contribute exactly that many arrivals.

Nothing about that needs 896 blocks.  `kern_k3_rs_arrive1`
(source/k3_rs_arrive1.cu) is one block that adds the whole grid's worth of
arrivals in a single atomic, which leaves the counters exactly where they were:
the epoch target still rises by `gridDim.x` per call, and this rank's slot still
reaches it only after this kernel, which is stream-ordered after the producer.
What it drops is the ramp of the launch, 895 of the 896 atomicAdds to one cache
line, and 896 copies of the two `__threadfence_system()` (each a full L1
invalidate) -- and for the normed barrier also the 456 instructions of push
setup it computes for a call that moves no bytes, including its 12 LDL / 9 STL
of spilled peer pointers (agent/areas/d.md item 4).

Both calls keep their op name, their slot, their `coll_flags` accounting and the
grid expression they share with every other collective; the only new thing is an
`units` argument in front of the slot, so the slot stays the last argument the
other two generators read their slot block out of.

Measured before this: removing the arrival launch from the reduce-scatter op
entirely takes the graph from 620.5 to 613.0 ms over 187 calls (~40 us per call,
of which ~15 us is the rank skew any barrier has to pay).

Idempotent: re-running it on an already converted manifest only re-checks.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULE = "k3_rs_arrive1"
ENTRY = "kern_k3_rs_arrive1"
THREADS = 32
# module entry name -> the launch it replaces
KPARAMS = ["in buffer<bf16>", "out buffer<bf16>", "in buffer<u64>", "out buffer<u64>",
           "in buffer<u64>", "i64", "i32", "i32", "i32", "i32", "i32"]
RS = "k3_reducescatter_bf16"
RS_ARRIVE = ("kern_k3_rs_arrive", "kern_k3_rs_arrive1")
RS_PULL = "kern_k3_rs_pull"
PUSH = "k3_allgather_push"
PUSH_ENTRY = ("kern_k3_allgather_push", "kern_k3_rs_arrive1")
ORDER = 6


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
    assert RS in m["ops"] and PUSH in m["ops"], "run the collectives generators first"

    # the epoch target rises by exactly the epoch bump's grid, once per call
    units = m["ops"]["k3_epoch_bump"]["impl"]["launches"][0]["grid"][0]
    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}

    # ---- the reduce-scatter's arrival
    launches = {l["entry"]: l for l in m["ops"][RS]["impl"]["launches"]}
    assert RS_PULL in launches, list(launches)
    old = [l for e, l in launches.items() if e in RS_ARRIVE]
    assert len(old) == 1, list(launches)
    old[0].update({"module": MODULE, "entry": ENTRY, "params": KPARAMS,
                   "args": [{"param": i} for i in range(9)] + [{"rank": "tp"}, {"i32": tp}],
                   "block": [THREADS, 1, 1], "grid": [1, 1, 1]})
    # the op gains `units` in front of the slot, so the pull launch is re-argued
    m["ops"][RS]["params"] = ["in buffer<bf16>", "out buffer<bf16>", "in buffer<u64>",
                              "out buffer<u64>", "in buffer<u64>", "i64", "i32", "i32", "i32"]
    pull = launches[RS_PULL]
    assert pull["module"].endswith("_nr4"), pull["module"]
    pull["args"] = [{"param": i} for i in range(6)] + [{"param": 7}, {"param": 8},
                                                       {"rank": "tp"}, {"i32": tp}]

    # ---- the payload-less normed barrier: same kernel, same accounting
    played = m["ops"][PUSH]["impl"]["launches"][0]
    assert played["entry"] in PUSH_ENTRY, played["entry"]
    if played["entry"] == "kern_k3_allgather_push":
        # it shares the epoch's grid, which is what makes its arrivals countable
        assert played["grid"] == m["ops"]["k3_epoch_bump"]["impl"]["launches"][0]["grid"]
    played.update({"module": MODULE, "entry": ENTRY, "params": KPARAMS,
                   "args": [{"param": i} for i in range(5)] + [{"i64": 0}, {"param": 5},
                                                               {"param": 6}, {"i32": ORDER},
                                                               {"rank": "tp"}, {"i32": tp}],
                   "block": [THREADS, 1, 1], "grid": [1, 1, 1]})
    # its sixth parameter used to be the payload length; it is the arrival count
    m["ops"][PUSH]["params"][5] = "i32"

    # ---- and the calls: `units` where the arrival count used to be derived
    n = {RS: 0, PUSH: 0}
    for c in calls:
        if c["op"] == RS:
            a = c["args"]
            assert len(a) in (8, 9), a
            c["args"] = a[:6] + [{"expr": units}] + (a[6:] if len(a) == 8 else a[7:])
            n[RS] += 1
        elif c["op"] == PUSH:
            a = c["args"]
            assert len(a) == 7, a
            assert a[-1].get("i32") is not None, a
            c["args"] = a[:5] + [{"expr": units}, a[-1]]
            n[PUSH] += 1
    assert all(n.values()), n

    used_mod = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    assert "k3_reducescatter_pull" not in used_mod, "the old arrive module should be free now"
    assert "k3_collectives" in used_mod, "the epoch bump and the flags init still live there"
    for mod in [x for x in m["modules"] if x not in used_mod]:
        del m["modules"][mod]
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"{n[RS]} reduce-scatter arrivals + {n[PUSH]} normed barriers on {MODULE} "
          f"(one block, units = {json.dumps(units)}), {len(m['ops'])} ops, "
          f"{len(m['modules'])} modules -> {out}")


if __name__ == "__main__":
    main()
