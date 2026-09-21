#!/usr/bin/env python3
"""Candidate: `l*.mla_prep` runs as two prefill-shaped kernels.

`mla_prep` (K4) is the MLA block's fused prep: from the f32 partial row
P = [q_a 1536 | kv_a 512 | rope 64 | gate 3072] it writes
  q_norm   = bf16(bf16(P[0:1536]) * rsqrt(mean + 1e-5) * gamma_q_a)
  slab row = kv_norm | rope          (paged kv state, slot_mapping[b])
  mla_gate = bf16(P[2112:5184])
The current kernel (module k3_mla_prep+INNER=3072+MLA_FUSED=5184) is one launch
of grid (tokens, 3), block 512: 65536 blocks at 16384 rows, ~4 B (head) or one
16 B pair (each gate block) per thread, 180 us per call for 573 MB = 3.2 TB/s
-- the furthest from the streaming rate of any kernel in this area.

v2 (`source/k3_mla_prep_v2.cu`, module k3_mla_prep_v2+INNER=3072+MLA_FUSED=5184)
splits that into two kernels of the same op:

  head  grid (ceil(tokens/16), 1, 1) block 512: one warp per row, a lane owns 16
        of the row's 528 float4 units, the sums of squares reduce inside the
        warp (no shared memory, no barrier), the row is read twice (L1).
  gate  grid (ceil(tokens/4), 1, 1) block 384: a flat landing copy, thread t
        owns 8 columns of four consecutive rows -- eight 16 B loads and four
        16 B stores in flight per thread instead of one.

Every landing point is the current kernel's, element for element (bf16 landing
before use, round-before-scale rms, __hmul2 gamma product, single landing for
rope and the gate band).  Only the order of the sums of squares changes: four
elements per thread + warp tree + per-warp partials -> 48 elements per lane +
one warp tree, worth at most one bf16 ulp on the elements sitting on a rounding
boundary.

One difference: which kernels `mla_prep` launches (one module, two launches,
the op interface and every buffer untouched).  Generated from the default
manifest of the moment.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "manifests/k3-tp4-prefill-16k.json"
DEFAULT_CUBIN = ROOT / "build" / "k3_mla_prep_v2+INNER=3072+MLA_FUSED=5184.cubin"
OLD = "k3_mla_prep+INNER=3072+MLA_FUSED=5184"
NEW = "k3_mla_prep_v2+INNER=3072+MLA_FUSED=5184"
PAGE_STRIDE = 884736
ROWS_HEAD = 4           # rows per head block (PH_ROWS, four warps per row)
ROWS_GATE = 4           # rows per gate block (PG_ROWS)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "manifests/k3-tp4-prefill-16k-mla-prep-v2.json"
    cubin = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CUBIN
    m = json.loads(src.read_text())
    assert OLD in m["modules"], OLD
    sha = hashlib.sha256(cubin.read_bytes()).hexdigest()
    m["modules"][NEW] = {"source": f"{NEW}.cubin", "sha256": sha}
    del m["modules"][OLD]

    op = m["ops"]["mla_prep"]
    assert op["params"] == ["in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>",
                            "in buffer<i64>", "inout state", "i64",
                            "out buffer<bf16>", "out buffer<bf16>", "i32"], op["params"]
    old = op["impl"]["launches"][0]
    assert old["module"] == OLD and old["entry"] == "kern_k3_mla_prep", old
    assert old["args"][0] == {"param": 0} and old["args"][6] == {"i64": PAGE_STRIDE}
    assert old["args"][5] == {"param": 5} and old["args"][9] == {"param": 8}
    assert old["grid"] == ["tokens", 3, 1] and old["block"] == [512, 1, 1], old

    head = {
        "module": NEW, "entry": "kern_k3_mla_prep_head",
        "params": ["in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>",
                   "in buffer<i64>", "inout state", "i64", "i64",
                   "out buffer<bf16>", "i32"],
        "args": [{"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                 {"param": 5}, {"i64": PAGE_STRIDE}, {"param": 6}, {"param": 8}],
        "block": [512, 1, 1], "grid": [{"ceil_div": ["tokens", ROWS_HEAD]}, 1, 1],
    }
    gate = {
        "module": NEW, "entry": "kern_k3_mla_prep_gate",
        "params": ["in buffer<f32>", "out buffer<bf16>", "i32"],
        "args": [{"param": 0}, {"param": 7}, {"param": 8}],
        "block": [384, 1, 1], "grid": [{"ceil_div": ["tokens", ROWS_GATE]}, 1, 1],
    }
    op["impl"]["launches"] = [head, gate]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(out, sha)


if __name__ == "__main__":
    main()
