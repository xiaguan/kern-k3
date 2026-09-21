#!/usr/bin/env python3
"""Candidate: K2 restores its whole chunk stage with one bulk copy.

K1 writes six per-chunk intermediates (k_decayed, q_decayed, k_restored in bf16,
g_total in f32, INV and Mqk bf16) into six separate planes; K2 restores each
with its own `cp.async.bulk` into the shared-memory stage, six 4 KB/512 B
copies per chunk and per CTA.  The six stage members of K2's `InputStorage`
are contiguous in exactly that order (offsets 2176, 6272, 10368, 14464, 14976,
15488 of a 16000-byte stage, 13824 bytes from k_decayed to the end of Mqk), so
K1 can write them as one 13824-byte block per (head, chunk) and K2 can move the
whole thing with a single bulk copy of the same bytes.

That is what `flash_kda_vllm_d128_ws1` does (source/flash-kda-vllm): the K1 and
K2 signatures take one workspace pointer instead of six, the shared-memory
members, their swizzles and every value are untouched, and the transaction
count K2's mbarrier expects is the same 13824 bytes expressed as one figure.
The manifest change is the module, its two entry names, one workspace buffer in
place of six, and one workspace argument in place of six.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = "flash_kda_vllm_d128_sw128"
NEW = "flash_kda_vllm_d128_ws1"
WS_BYTES_PER_CHUNK = 13824
OLD_WS = ["span_ws_kd", "span_ws_qd", "span_ws_kr", "span_ws_gt", "span_ws_inv", "span_ws_mqk"]


def entry_names(cubin):
    syms = subprocess.run(["cuobjdump", "-symbols", str(cubin)], capture_output=True,
                          text=True, check=True).stdout.split()
    prep = [w for w in syms if w.startswith("_Z22_flash_kda_fwd_prepare")]
    rec = [w for w in syms if w.startswith("_Z25_flash_kda_fwd_recurrence")
           and "Li64EEvT_T0_T1_T2_T3_PSV_PvPKT14_iii" in w]
    assert len(prep) == 1 and len(rec) == 1, (prep, rec)
    return prep[0], rec[0]


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "manifests/k3-tp4-prefill-16k.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "manifests/k3-tp4-prefill-16k-ws1.json"
    cubin = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "build" / f"{NEW}.cubin"
    m = json.loads(src.read_text())
    op = m["ops"]["flash_kda"]
    mods = {l["module"] for l in op["impl"]["launches"]}
    if mods == {NEW}:
        print(f"{src.name}: already on {NEW}, nothing to do")
        return
    assert mods == {OLD}, mods
    sha = hashlib.sha256(cubin.read_bytes()).hexdigest()
    prep_entry, rec_entry = entry_names(cubin)
    mod = json.loads(json.dumps(m["modules"][OLD]))
    mod["sha256"] = sha
    mod["source"] = f"{NEW}.cubin"
    m["modules"][NEW] = mod
    del m["modules"][OLD]

    # one workspace buffer in place of six: per (head, chunk) block of 13824 B
    ws_elems = 0
    for name in OLD_WS:
        buf = m["buffers"][name]
        n = 1
        for d in buf["shape"]:
            n *= d
        ws_elems += n * (4 if buf["dtype"] == "f32" else 2)
        del m["buffers"][name]
    assert ws_elems % WS_BYTES_PER_CHUNK == 0, ws_elems
    m["buffers"]["span_ws"] = {"dtype": "bf16", "shape": [ws_elems // 2], "kind": "workspace"}

    old_params = op["params"]
    assert len(old_params) == 17 and old_params[10:16] == [
        "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>",
        "out buffer<f32>", "out buffer<bf16>", "out buffer<bf16>"], old_params
    op["params"] = old_params[:10] + ["out buffer<bf16>"] + [old_params[16]]
    for li, l in enumerate(op["impl"]["launches"]):
        l["module"] = NEW
        l["entry"] = prep_entry if li == 0 else rec_entry
        # the six workspace pointer arguments collapse into one
        args = l["args"]
        assert [a for a in args[13:19]] == [{"param": p} for p in range(10, 16)], args[13:19]
        l["args"] = args[:13] + [{"param": 10}] + args[19:]
        assert len(l["args"]) == len(l["params"]) - 5, (len(l["args"]), len(l["params"]))
        is_k1 = li == 0
        d = "out" if is_k1 else "in"
        assert l["params"][13:19] == [f"{d} buffer<bf16>"] * 3 + [f"{d} buffer<f32>"] + \
            [f"{d} buffer<bf16>"] * 2, l["params"][13:19]
        l["params"] = l["params"][:13] + ["out buffer<bf16>" if is_k1 else "in buffer<bf16>"] + \
            l["params"][19:]
        # the tail scalars referenced the old interface index of `tokens` (16)
        for a in l["args"]:
            if a == {"param": 16}:
                a["param"] = 11

    calls = m["programs"]["prefill"]["calls"]
    n = 0
    for c in calls:
        if c["op"] == "flash_kda":
            a = c["args"]
            assert [x.get("buf") for x in a[10:16]] == OLD_WS, a[10:16]
            c["args"] = a[:10] + [{"buf": "span_ws"}] + a[16:]
            n += 1
    assert n == 69, n
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(out, sha, f"({n} calls, {ws_elems} workspace bytes)")


if __name__ == "__main__":
    main()
