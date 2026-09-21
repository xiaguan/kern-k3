#!/usr/bin/env python3
"""Candidate: K2's recurrent state update is one bf16 fused multiply-add.

K2's inner loop (Phase 6) walks the whole [D, VD] recurrent state once per
chunk and updates every element as

    s = bf16(f32(s) * g + f32(u))

which the compiler spends three instructions on per element: `cvt.f32.bf16`
for s (an IMAD in SASS), an FFMA, and an F2FP back to bf16.  64 of those per
warp per chunk is the single largest instruction block in the loop body, and
the loop body is what this kernel spends its ~1200 cycles per chunk on (48
CTAs, 1024 serial chunks, ~447 SASS instructions per warp per chunk of which
52 are HMMA).

`__hfma` on bf16 evaluates a*b+c in full precision and rounds once, so

    s = __hfma(s, bf16(g), bf16(u))

is the same computation with one instruction per element: `nvdisasm` on the
candidate shows FFMA 66 -> 2, IMAD 304 -> 241, PRMT 52 -> 20 and 32 new HFMA2,
i.e. 112 fewer instructions in the VD=64 recurrence (1888 -> 1776).

Numerics: the product and the add are still exact and rounded once, and the
state was already bf16 before and after.  Two things round differently: `g`,
the per-chunk decay the workspace carries as f32, is now the bf16 rounding of
it (<= 0.2% on the decayed part, against the 0.4% the state's own bf16 rounding
already applies every chunk), and `u` is rounded to bf16 before the add
instead of after (one ulp).  The module is renamed because the cubin's sha
changes.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = "flash_kda_vllm_d128_ws1"
NEW = "flash_kda_vllm_d128_hfma1"


def rec_entry(cubin):
    syms = subprocess.run(["cuobjdump", "-symbols", str(cubin)], capture_output=True,
                          text=True, check=True).stdout.split()
    rec = [w for w in syms if w.startswith("_Z25_flash_kda_fwd_recurrence")
           and "Li64EEvT_T0_T1_T2_T3_PSV_PvPKT14_iii" in w]
    assert len(rec) == 1, rec
    return rec[0]


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "manifests/k3-tp4-prefill-16k.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "manifests/k3-tp4-prefill-16k-kda-hfma.json"
    cubin = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "build" / f"{NEW}.cubin"
    m = json.loads(src.read_text())
    op = m["ops"]["flash_kda"]
    mods = {l["module"] for l in op["impl"]["launches"]}
    if NEW in mods:
        print(f"{src.name}: already on {NEW}, nothing to do")
        return
    assert mods == {OLD}, mods
    mod = json.loads(json.dumps(m["modules"][OLD]))
    mod["sha256"] = hashlib.sha256(cubin.read_bytes()).hexdigest()
    mod["source"] = f"{NEW}.cubin"
    m["modules"][NEW] = mod
    op["impl"]["launches"][1]["module"] = NEW
    op["impl"]["launches"][1]["entry"] = rec_entry(cubin)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(out, mod["sha256"][:16])


if __name__ == "__main__":
    main()
