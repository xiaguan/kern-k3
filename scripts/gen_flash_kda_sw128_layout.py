#!/usr/bin/env python3
"""Candidate: K2's value tile moves to the 128-byte-swizzled layout.

`flash_kda`'s K2 recurrence loads the value tile v[16, 64] and stores the output
tile out[16, 64] through tensor maps built from `VOLayout`.  The vendored kernel
uses `GMMA::Layout_K_INTER_Atom` for that layout, whose inner run is 128 *bits*
(16 B of bf16), so a tensor-map box can be at most 8 elements wide in D: one
16x64 tile costs eight 256 B TMA operations on the way in and eight on the way
out.  With `GMMA::Layout_K_SW128_Atom` the inner run is 128 bytes, the box is
the whole 64-column tile, and each direction is one 2 KB operation.

Both the load and the store are pure data movement: the same elements land in
the same registers in the same order, the MMA/LDSM/STSM instruction stream is
unchanged apart from the shared-memory addresses it reads and writes.  The
judge reproduces the default manifest's numbers exactly (8 flips of 136
positions and KL p50 0.0127 against the recorded producers, the same flips at
the same positions as the unmodified default), and `kern test` shows no span
difference from the default manifest.

The tensor-map fields in the manifest are part of the kernel ABI: box width and
swizzle mode have to follow the layout, so this generator patches
  * the module (sha256 from build/, entry name from the cubin),
  * K2's launch `shared_mem` (the swizzled tile needs 1024-byte alignment),
  * the two descriptors that use VOLayout: param 2 (v, the box's first
    dimension 8 -> 64, swizzle 0 -> 128) and param 9 (out, the same).
Everything else in the manifest is untouched.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = "flash_kda_vllm_d128"
NEW = "flash_kda_vllm_d128_sw128"
SMEM = 70656

def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "manifests/k3-tp4-prefill-16k.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "manifests/k3-tp4-prefill-16k-kda-sw128.json"
    cubin = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "build" / f"{NEW}.cubin"
    m = json.loads(src.read_text())
    op = m["ops"]["flash_kda"]
    mods = {l["module"] for l in op["impl"]["launches"]}
    if mods == {NEW}:
        print(f"{src.name}: already on {NEW}, nothing to do")
        return
    assert mods == {OLD}, mods
    assert OLD in m["modules"], OLD
    sha = hashlib.sha256(cubin.read_bytes()).hexdigest()
    syms = subprocess.run(["cuobjdump", "-symbols", str(cubin)],
                          capture_output=True, text=True, check=True).stdout
    ent = {w for w in syms.split()
           if w.startswith("_Z25_flash_kda_fwd_recurrence")
           and "Li64EEvT_T0_T1_T2_T3_PSV_PvPKT14_iii" in w}
    assert len(ent) == 1, ent
    ent = ent.pop()

    mod = json.loads(json.dumps(m["modules"][OLD]))
    mod["sha256"] = sha
    mod["source"] = f"{NEW}.cubin"
    m["modules"][NEW] = mod
    del m["modules"][OLD]

    launches = op["impl"]["launches"]
    assert len(launches) == 2 and launches[0]["block"] == [128, 1, 1], launches[0]
    assert launches[1]["block"] == [192, 1, 1] and launches[1]["grid"] == [48, 1, 1], launches[1]
    assert launches[1]["shared_mem"] == 68608, launches[1]["shared_mem"]
    for l in launches:
        l["module"] = NEW
    launches[1]["entry"] = ent
    launches[1]["shared_mem"] = SMEM
    patched = []
    for arg in launches[1]["args"]:
        if not isinstance(arg, dict) or "pack" not in arg:
            continue
        for f in arg["pack"]["fields"]:
            tm = f.get("tensormap")
            if tm and tm["param"] in (2, 9):
                assert tm["box"] == [8, 16, 1] and tm["swizzle"] == 0, tm
                tm["box"] = [64, 16, 1]
                tm["swizzle"] = 128
                patched.append(tm["param"])
    assert patched == [2, 9], patched
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(out, sha)

if __name__ == "__main__":
    main()
