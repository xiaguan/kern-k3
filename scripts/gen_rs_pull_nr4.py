#!/usr/bin/env python3
"""Candidate: the reduce-scatter pull is compiled for the shape it actually runs.

`kern_k3_rs_pull` read `nranks` and the ring `order` back out of its launch
arguments, so the chain was built at run time: `q = (rank + o) % nr` came out of
nvcc as ten groups of MUFU.RCP / I2F / F2I / IMAD.HI, the kernel was 944
instructions of which 402 were prologue, and each of the four hops sat behind a
conditional branch of ~250 instructions that kept the four peer loads from being
issued back to back.  The shape is not a run-time property: the TP group is four
ranks and every call site asks for order 6.

`source/k3_reducescatter_pull_nr4.cu` compiles it in (`defines = { NR = 4,
ORDER = 6 }` in kernels.toml, the same mechanism `k3_kda_out_gate+HEADS=24` uses),
so the chain is four unconditional straight lines, each hop's rank is
`(rank + o) & (NR - 1)` with `o` a constant, and a bf16 pair unpacks with a shift
and a mask instead of PRMT + IMAD.  Measured with cuobjdump: 944 -> 224
instructions, no MUFU / I2F / F2I left, no LDL/STL, the four LDG.E.128 of a hop
adjacent.  The kernel keeps its signature and returns early if a caller of
another shape reaches it; the rounding chain is untouched -- peers ascending, own
partial last, one bf16 rounding per hop, which is what ncclReduceScatter's ring
does.

Only the pull launch changes module: `kern_k3_rs_arrive` stays where it is, so
this candidate is one change in one place.

Idempotent: re-running it on an already converted manifest only re-checks.
"""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OP = "k3_reducescatter_bf16"
ENTRY = "kern_k3_rs_pull"
MODULE = "k3_reducescatter_pull_nr4"
OLD = "k3_reducescatter_pull"
NR = 4
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
    assert tp == NR, tp
    assert OP in m["ops"], "run scripts/gen_collectives_rs.py first (the reduce-scatter)"
    launches = m["ops"][OP]["impl"]["launches"]
    pull = [l for l in launches if l["entry"] == ENTRY]
    assert len(pull) == 1, launches
    assert pull[0]["module"] in (OLD, MODULE), pull[0]["module"]

    # every call site has to be the shape the cubin was built for
    n = 0
    for c in calls:
        if c["op"] != OP:
            continue
        assert len(c["args"]) == 8, c["args"]
        assert c["args"][-1]["i32"] == ORDER, f"call carries order {c['args'][-1]}, cubin is built for {ORDER}"
        n += 1
    assert n, "no reduce-scatters to specialize"

    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}
    pull[0]["module"] = MODULE

    used_mod = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    assert OLD in used_mod, "the arrive launch still needs its module"
    for mod in [x for x in m["modules"] if x not in used_mod]:
        del m["modules"][mod]
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"{n} reduce-scatter pulls on {MODULE} (NR={NR}, ORDER={ORDER}), "
          f"{len(m['ops'])} ops, {len(m['modules'])} modules -> {out}")


if __name__ == "__main__":
    main()
