#!/usr/bin/env python3
"""Candidate: `l*.res_mlp` reads the candidates' statistics out of `l*.res_in`.

Both kernels walk the layer's attention-residual candidates `blocks[b][0..nb-1]`
with the same tree -- 128 threads, seven 16 B vectors each, the same butterfly
and the same four warp partials -- to mix them by a softmax weight: `res_in` for
`normed` (with `sw_attn`), `res_mlp` for the MLP residual (with `sw_mlp`).  Of the
two statistics per candidate, `sq_c = sum x^2` is the *same* number for both and
`dp_c = sum x*sw` differs only in which `sw` is used.

So `source/k3_residual_push_stats.cu` computes both -- `sq_c` for its own score
(which it was computing anyway) and `dp_c` with the *MLP's* `sw` -- and stores
`(sq_c, dp_c)` per (row, candidate) in `attn_stats` (256 KB per call).  The
summation order is `res_mlp`'s own, so what it reads is bit-identical to what it
used to compute.  `source/k3_residual_nb_stats.cu` is the landing reading them:
its pass 1 keeps only the prefix candidate's statistics (those need the
reduce-scatter's output) and skips the candidate rows entirely.

The candidates are then read once per layer instead of twice.  The hand-off only
pays where the landing has enough candidates to make the skipped reads matter:
measured per call on this branch, the landing saves 2.1 us at nb = 1, 3.8 at 2,
5.1 at 3, 8.8 at 4, 16.0 at 5, 16.3 at 6, 30.0 at 7 and 48.4 at 8, while the push
pays 6.7 us for the extra `sw` vector and its own statistics everywhere -- so
`MIN_NB` keeps the small layers on the plain path, where the hand-off would lose
money.  `l0` and `out.res` stay on it too (their candidate count is 0).

Idempotent: re-running it on a converted manifest re-checks and re-pins.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
STATS = "attn_stats"
KNB_MAX = 8
MIN_NB = 4              # below this the landing saves less than the push pays
PUSH_OLD = "attnres_rms_push_v3"
PUSH_FIRST = "attnres_rms_push_first_v3"
PUSH_STATS = "attnres_rms_push_stats"
PUSH_MODULE, NB_MODULE = "k3_residual_push_stats", "k3_residual_nb_stats"
NB_OPS = [f"land_add_attnres_rms_bf16_v2_nb{k}" for k in range(1, 9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or a.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    for name, marker in ((PUSH_MODULE, "dpm"), (NB_MODULE, "nb_stats")):
        body = (ROOT / kernels[name]["source"]).read_text()
        assert marker in body, f"{kernels[name]['source']} does not carry {marker}"
    m = json.loads(a.manifest.read_text())
    calls = m["programs"]["prefill"]["calls"]
    converted = STATS in m["buffers"]

    # the two kernels the landing and the push read: one manifest for both shapes
    wired = {}
    for c in calls:
        if c["label"].startswith("l") and c["op"] in NB_OPS:
            wired[c["label"].split(".", 1)[0]] = c
    def is_wired(p2):
        """a converted landing carries the statistics buffer as its second-to-last
        argument and the count they cover as its last"""
        a = p2["args"]
        return len(a) >= 2 and isinstance(a[-2], dict) and a[-2].get("buf") == STATS

    def landing_nb(p2):
        return p2["args"][-1]["i32"] if is_wired(p2) else p2["args"][-3]["i32"]

    def push_nb(c):
        """how many candidates the push's statistics cover: its nb argument, which
        the two appended arguments move to index -5 once converted"""
        a = c["args"]
        if c["op"] == PUSH_STATS:
            return a[-5]["i32"] if len(a) >= 5 and "i32" in a[-5] else 0
        return a[-3]["i32"] if len(a) >= 9 else 0

    pairs = []
    for c in calls:
        if c["op"] in (PUSH_OLD, PUSH_STATS) and c["label"].startswith("l"):
            lab = c["label"].split(".", 1)[0]
            p2 = wired.get(lab)
            if p2 is not None and landing_nb(p2) >= MIN_NB:
                pairs.append((c, p2, push_nb(c)))

    if not converted:
        assert len(pairs) >= 50, len(pairs)
        m["buffers"][STATS] = {"dtype": "f32", "shape": [4096, KNB_MAX, 2], "kind": "workspace"}
        push = json.loads(json.dumps(m["ops"][PUSH_OLD]))
        push["params"] += ["in buffer<f32>", "out buffer<f32>"]
        l = push["impl"]["launches"][0]
        lp = l["params"]
        assert lp[-2:] == ["i32", "i32"], lp[-3:]     # rank, nranks end the kernel ABI
        l["params"] = lp + ["in buffer<f32>", "out buffer<f32>"]
        l["args"] = l["args"] + [{"param": len(push["params"]) - 2},
                                 {"param": len(push["params"]) - 1}]
        l["module"] = PUSH_MODULE
        m["ops"][PUSH_STATS] = push
        for op in [o for o in NB_OPS if int(o.rsplit("nb", 1)[1]) >= MIN_NB]:
            o = m["ops"][op]
            o["params"] += ["in buffer<f32>", "i32"]
            for l2 in o["impl"]["launches"]:
                assert l2["module"] in (NB_MODULE, "k3_residual_nb+LAND_BF16=1"), l2["module"]
                l2["module"] = NB_MODULE
        for c, p2, nb in pairs:
            sw_mlp = p2["args"][3]["buf"]
            assert sw_mlp.startswith("layers.") and sw_mlp.endswith(".sw_mlp"), sw_mlp
            c["op"] = PUSH_STATS
            c["args"] = c["args"] + [{"buf": sw_mlp}, {"buf": STATS}]
            p2["args"] = p2["args"] + [{"buf": STATS}, {"i32": nb}]

    # post-conditions, on either shape of default
    n = 0
    for c, p2, nb in pairs:
        assert c["op"] == PUSH_STATS and c["args"][-1]["buf"] == STATS, c["label"]
        assert p2["args"][-2]["buf"] == STATS and p2["args"][-1]["i32"] == nb, p2["label"]
        n += 1
    assert n >= 50, n
    for name in (PUSH_MODULE, NB_MODULE):
        m["modules"][name] = {"source": f"{name}.cubin", "sha256": kernels[name]["sha256"]}
    used = {l["module"] for o in m["ops"].values() for l in o["impl"]["launches"] if "module" in l}
    for name in [x for x in m["modules"] if x not in used]:
        del m["modules"][name]
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"candidate statistics handed from res_in to res_mlp on {n} layers "
          f"(nb >= {MIN_NB}); {len(m['ops'])} ops, {len(m['modules'])} modules -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
