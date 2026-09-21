#!/usr/bin/env python3
"""Candidate: the routed experts are placed on ranks round robin, not in blocks of 56.

The 224 routed experts are split over the four EP ranks by the `ep` group of
each MoE weight bind: rank k gets the tensor in position k of every bind entry,
so entry 2i/2i+1 (expert `i` of the rank, w3 then w1) decides the whole
placement.  The default manifest puts expert `56 k + i` on rank k, so a rank
owns a contiguous block of the id space -- but the router does not spread a
token's top-16 over the id space evenly, and the four ranks do not hold the
same tokens, so the per-rank routed row counts differ by 17% and the per-layer
maximum is 16.7% above the per-layer mean (rank 0..3 MoE totals over 92 layers
132.9/143.2/152.1/155.5 ms, per-layer max averaged 1851 us against a 1586 us
mean).

This candidate keeps every op and every buffer, and changes only which rank
owns which expert: rank k owns the global expert ids `e` with `e % 4 == k`
(local slot `e / 4`), i.e. each bind entry lists the four experts `4i+k`.  A
token's 16 chosen experts then fall ~4 to a rank instead of ~16 to one, so
every rank gets about a quarter of every token's rows and the per-layer
maximum should approach the mean.

Only `kern_k3_moe_route_count` and `kern_k3_moe_route_scatter` encode the
placement (the local-expert test on a global id); `kern_k3_moe_route_tables`,
`kern_k3_moe_finalize`, `kern_k3_moe_quant` and the two weight shuffles are
placement agnostic and are byte-identical in `k3_moe_prefill_v3+EP_STRIDED=1`
to their `k3_moe_prefill_v2` code (checked with
`cuobjdump -sass` per entry).  The candidate module is the same source file
built with `-DEP_STRIDED=1 -DEP_RANKS=4`.

No arithmetic changes: the weights, the activations, the GEMMs, the tables and
the finalize's per-token summation order are all as before.  What changes is
which rank's partial sum holds which experts, so `moe_partial` rounds
differently in the last bf16 ulp for the tokens whose local slot set moves.

Run with no arguments to rewrite the default manifest in place; pass another
path to write a copy there instead.  Idempotent: the module swap is skipped
when the manifest already runs the module and the bind permutation is a
function of the expert ids only.
"""
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "manifests/k3-tp4-prefill-16k.json"
OLD = "k3_moe_prefill_v2"
NEW = "k3_moe_prefill_v3+EP_STRIDED=1"
RANKS = 4          # `ep` group width
E = 56             # local experts per rank
PHASES = {"w13": ("w3", "w1"), "w13_sf": ("w3", "w1"), "w2": ("w2",), "w2_sf": ("w2",)}
EXPERT = re.compile(r"^(.*block_sparse_moe\.experts\.)(\d+)(\.w\d)(\..*)$")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    out = src if len(sys.argv) <= 2 else Path(sys.argv[2])
    m = json.loads(src.read_text())
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    sha = kernels[NEW]["sha256"]
    assert kernels[NEW]["source"] == "source/k3_moe_prefill_v3.cu"
    assert kernels[NEW]["defines"] == {"EP_STRIDED": 1, "EP_RANKS": RANKS}, kernels[NEW]["defines"]

    # 1. the MoE ops launch the strided module
    if OLD in m["modules"]:
        assert m["modules"][OLD]["sha256"] == kernels[OLD]["sha256"]
        del m["modules"][OLD]
    m["modules"][NEW] = {"source": f"{NEW}.cubin", "sha256": sha}
    for name, op in m["ops"].items():
        for launch in op["impl"].get("launches", []):
            if launch.get("module") in (OLD, NEW):
                launch["module"] = NEW
    live = {l.get("module") for op in m["ops"].values() for l in op["impl"].get("launches", [])}
    assert set(m["modules"]) <= live, sorted(set(m["modules"]) - live)

    # 2. the weight binds: entry 2i+p of rank k is the expert 4i+k of phase p
    moved = 0
    for key, buf in m["buffers"].items():
        mm = re.match(r"layers\.\d+\.moe\.(w13|w13_sf|w2|w2_sf)\.raw$", key)
        if not mm:
            continue
        phases = PHASES[mm.group(1)]
        by_id = {}
        for entry in buf["bind"]:
            assert entry["tensor"]["group"] == "ep"
            for t in entry["tensor"]["tensors"]:
                g = EXPERT.match(t)
                assert g, t
                by_id[(int(g.group(2)), g.group(3))] = t
        assert len(by_id) == 224 * len(phases), (key, len(by_id))
        bind = [{"tensor": {"group": "ep",
                            "tensors": [by_id[(RANKS * i + k, "." + p)] for k in range(RANKS)]}}
                for i in range(E) for p in phases]
        if bind != buf["bind"]:
            moved += 1
        buf["bind"] = bind
    assert moved in (0, 368), moved   # 92 MoE layers x 4 weight buffers

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"{out}: {moved} of 368 MoE weight binds re-laid round robin, module {NEW}")


if __name__ == "__main__":
    main()
