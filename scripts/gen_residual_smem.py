#!/usr/bin/env python3
"""Candidate: the residual landing keeps its thread's row slice in shared memory.

`attnres_rms_row` (source/k3_residual_nb.cu) holds `V8 pv[KVPT]` -- the prefix
candidate (prefix + partial) and then the mixed row -- in registers from the
entry's first load to the last `normed` store: 28 registers at KVPT = 7, live
through pass 1, pass 2 and the RMS phase, and the reason the entries sit at
102-106 registers and four blocks per SM at nb >= 3.

It lives in shared memory now: one 16 B slot per (vector, thread) at
`s_pv[v * RES_THREADS + t]` (consecutive threads, so the warp's 128-bit accesses
are bank-clean), read and written through volatile `ld.shared.v4.u32` /
`st.shared.v4.u32` -- the same trick `ldv_nv` uses for the global candidates, so
the compiler cannot hoist or cache the value back into registers.  Nothing else
moves: same accumulation order, same reductions, same landings.

cuobjdump, per entry, registers before -> after (block 128):

    nb  1   2   3   4   5   6   7   8
       88  86 102 102 100 106 102 104   (registers)
    -> 40  48  48  54  60  66  70  68
    blocks/SM 5  5   5   5   5   4   5   4
    ->        12  10  10   9   8   7   7   7

with 15.7 KB of shared memory per block and no spill (STACK 0, LDL/STL 0).  The
result is bit-identical: `test16k` reports every span, the logits and the KV/KDA
state identical, and the values written to and read back from shared memory are
the same bits the registers held.

This generator also carries the two steps this one builds on, so that replaying
it on a default that still has the run-time-`nb` op reproduces the whole chain:

  * the per-candidate-count entries (`scripts/gen_residual_nb.py`, which is why
    that file is touched by the same commit -- compose replays it first when it
    is present, and this one then finds the work done);
  * the chunked candidate loads (`CH = 2`, round 15), which live in the source
    and in the module's `defines`.

Idempotent: re-running it, on either shape of default, only re-checks.
"""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULE = "k3_residual_nb+LAND_BF16=1"
OLD_MODULE = "k3_residual_v2+LAND_BF16=1"
OLD_OP = "land_add_attnres_rms_bf16_v2"
OLD_ENTRY = "kern_k3_land_add_attnres_rms"
OP = OLD_OP + "_nb{}"
ENTRY = "kern_k3_land_add_attnres_rms_nb{}"
NBS = range(1, 9)


def split_candidates(m, kernels):
    """Give the one run-time-`nb` landing op the eight compile-time entries.

    This is `scripts/gen_residual_nb.py` (round 8) inlined, so that a default
    that never saw that commit -- main's, where the op still points at
    `k3_residual_v2+LAND_BF16=1` -- comes out the same way as the branch's.
    """
    if all(OP.format(k) in m["ops"] for k in NBS):
        assert OLD_OP not in m["ops"], "both the generic and the split landing are present"
        return 0
    calls = m["programs"]["prefill"]["calls"]
    assert OLD_OP in m["ops"], "the residual landing op is not in this manifest"
    n = 0
    for c in calls:
        if c["op"] == OLD_OP or c["op"].startswith(OLD_OP + "_nb"):
            assert c["args"][-3]["i32"] in NBS, f"nb {c['args'][-3]['i32']} has no entry"
            n += 1
    assert n, "no residual landings to specialize"
    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}
    base = m["ops"][OLD_OP]
    launch = base["impl"]["launches"][0]
    assert launch["entry"] == OLD_ENTRY, launch["entry"]
    for k in NBS:
        op = json.loads(json.dumps(base))
        op["impl"]["launches"][0]["module"] = MODULE
        op["impl"]["launches"][0]["entry"] = ENTRY.format(k)
        m["ops"][OP.format(k)] = op
    for c in calls:
        if c["op"] == OLD_OP or c["op"].startswith(OLD_OP + "_nb"):
            c["op"] = OP.format(c["args"][-3]["i32"])
    called = {c["op"] for prog in m["programs"].values() for c in prog["calls"]}
    for op in [o for o in m["ops"] if o not in called]:
        del m["ops"][op]
    used = {l["module"] for op in m["ops"].values() for l in op["impl"]["launches"] if "module" in l}
    assert OLD_MODULE not in used, "the run-time-nb module should be free now"
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or args.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    src = ROOT / kernels[MODULE]["source"]
    body = src.read_text()
    assert "ld.shared.v4.u32" in body and "s_pv" in body, f"{src} is not the shared-memory form"
    assert "#define CH 2" in body, f"{src} is not the chunked form"
    assert kernels[MODULE].get("defines", {}) == {"LAND_BF16": 1, "CH": 2}, kernels[MODULE]["defines"]
    m = json.loads(args.manifest.read_text())
    split = split_candidates(m, kernels)
    for k, op in enumerate([OP.format(k) for k in NBS], 1):
        l = m["ops"][op]["impl"]["launches"][0]
        assert l["module"] == MODULE and l["entry"] == ENTRY.format(k), (op, l)
        assert l["block"] == [128, 1, 1], (op, l["block"])
    m["modules"][MODULE] = {"source": f"{MODULE}.cubin", "sha256": kernels[MODULE]["sha256"]}
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"residual landing: {split} calls split per candidate count, shared-memory row "
          f"slice ({len(m['ops'])} ops, {len(m['modules'])} modules) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
