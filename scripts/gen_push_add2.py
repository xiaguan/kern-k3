#!/usr/bin/env python3
"""Candidate: `l*.hidden` (land_add2) is folded into the residual push.

`l<N>.hidden` (`kern_k3_land_add2_bf16`, module `k3_land_bf16`) computes
`hidden = bf16(prefix2 + p1 (+ p2))` -- 39.4 us per call, 3 reads and a write of
58.7 MB each -- and `l<N+1>.res_in` (`kern_k3_attnres_rms_push`) is the *next*
call in the graph, reading that row straight back as the prefix candidate of its
mix.  The two are one row-wise expression apart.

`source/k3_residual_push_add2.cu` is the push of round 20 with that expression
folded in: it computes the row in registers, writes it out for the attention and
the MoE, and uses it directly -- same arithmetic, same bf16 landing, same order,
so the mix sees the value it used to read.  What goes away is one 58.7 MB read
and one launch per layer; the reads land_add2 needed (176 MB) fit under the
push's wire (176 MB of peer stores per call, local traffic at ~4.4 TB/s of the
~5.8 available), which is why this fusion is worth taking where round 18's
parameter staging was not.

The generator pairs each `land_add2_v2` call with the push that follows it (all
93 are adjacent), moves the five land_add2 arguments onto the push call -- the
fused op is the statistics push plus `p1, p2, prefix2, hidden, two` -- and drops
the land_add2 call, its op and its module if nothing else uses them.

Idempotent: re-running it only re-checks.
"""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
NEW_MODULE = "k3_residual_push_add2"
NEW_OP = "attnres_rms_push_add2"
PUSH_OPS = ("attnres_rms_push_stats", "attnres_rms_push_v3")
HIDDEN_OP = "land_add2_v2"
NEW_PARAMS = ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
              "out buffer<bf16>", "i32"]
MARKERS = ("kern_k3_attnres_rms_push", "bf16(prefix2 + p1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "manifests/k3-tp4-prefill-16k.json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or a.manifest
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    body = (ROOT / kernels[NEW_MODULE]["source"]).read_text()
    for marker in MARKERS:
        assert marker in body, f"{kernels[NEW_MODULE]['source']} does not carry {marker}"
    m = json.loads(a.manifest.read_text())
    calls = m["programs"]["prefill"]["calls"]

    if NEW_OP not in m["ops"]:
        # The fused kernel computes the prefix row itself, so the push's first
        # parameter (the row it used to read) is gone: the op drops it and every
        # later parameter shifts down by one, in the params and in the launch's
        # mapping (the launch also carries rank and nranks, which the op does
        # not).  What is left is 15 op parameters and the kernel's 17.
        op = json.loads(json.dumps(m["ops"]["attnres_rms_push_stats"]))
        l = op["impl"]["launches"][0]
        assert op["params"][0] == "in buffer<bf16>" and l["args"][0] == {"param": 0}, l["args"][:2]
        op["params"] = op["params"][1:] + NEW_PARAMS
        l["params"] = l["params"][1:] + NEW_PARAMS
        margs = []
        for arg in l["args"][1:]:
            margs.append({"param": arg["param"] - 1} if "param" in arg else arg)
        l["args"] = margs + [{"param": len(op["params"]) - len(NEW_PARAMS) + i}
                             for i in range(len(NEW_PARAMS))]
        l["module"] = NEW_MODULE
        m["ops"][NEW_OP] = op

    fused = 0
    drop = []
    for i, c in enumerate(calls):
        if c["op"] != HIDDEN_OP:
            continue
        assert i + 1 < len(calls), c["label"]
        nxt = calls[i + 1]
        assert nxt["op"] in PUSH_OPS, (c["label"], nxt["op"])
        assert len(c["args"]) == 6, c["args"]
        p1, p2, prefix2, hidden, two, nb = c["args"]
        assert isinstance(two, dict) and "i32" in two, two
        assert nxt["args"][8] == nb, (nxt["label"], nxt["args"][8], nb)
        if nxt["op"] == "attnres_rms_push_v3":
            # The plain push has no statistics arguments yet: it gets the
            # layer's own `sw_mlp`, which the landing beside it already reads,
            # and the workspace the other layers use.  Below MIN_NB the landing
            # does not read the statistics back, so they are written and
            # ignored -- one 28.7 KB vector per row of otherwise idle read
            # bandwidth, against folding a whole kernel away.
            # the landing that would read them is the push layer's own, but
            # these layers are below MIN_NB and read nothing back, so any real
            # `sw_mlp` of the right shape serves; take the one beside the
            # land_add2 being folded in.
            sw_mlp = None
            for prefix in (c["label"].split(".", 1)[0], nxt["label"].split(".", 1)[0]):
                for other in calls:
                    if other["label"] == prefix + ".res_mlp" and len(other["args"]) > 3:
                        cand = other["args"][3].get("buf") if isinstance(other["args"][3], dict) else None
                        if cand and cand.endswith(".sw_mlp"):
                            sw_mlp = cand
                if sw_mlp:
                    break
            if not sw_mlp:
                for other in calls:
                    for arg in other["args"]:
                        if isinstance(arg, dict) and str(arg.get("buf", "")).endswith(".sw_mlp"):
                            sw_mlp = arg["buf"]
                            break
                    if sw_mlp:
                        break
            assert sw_mlp and sw_mlp.endswith(".sw_mlp"), (c["label"], nxt["label"], sw_mlp)
            assert len(nxt["args"]) == 9, len(nxt["args"])
            nxt["args"] = nxt["args"] + [{"buf": sw_mlp}, {"buf": "attn_stats"}]
        # the fused kernel no longer reads the prefix row: drop that argument
        assert len(nxt["args"]) == 11, (nxt["label"], len(nxt["args"]))
        nxt["args"] = nxt["args"][1:]
        nxt["op"] = NEW_OP
        nxt["args"] = nxt["args"] + [p1, p2, prefix2, hidden, two]
        drop.append(i)
        fused += 1

    for i in reversed(drop):
        del calls[i]
    assert fused >= 90, fused

    leftover = [c for c in calls if c["op"] == HIDDEN_OP]
    if not leftover:
        del m["ops"][HIDDEN_OP]
    for name in [n for n in m["modules"] if n == "k3_land_bf16" and not leftover]:
        del m["modules"][name]
    live = {c["op"] for prog in m["programs"].values() for c in prog["calls"]}
    for name in [n for n in m["ops"] if n not in live]:
        del m["ops"][name]
    used = {l["module"] for o in m["ops"].values() for l in o["impl"]["launches"] if "module" in l}
    for name in [n for n in m["modules"] if n not in used]:
        del m["modules"][name]
    for name in (NEW_MODULE,):
        m["modules"][name] = {"source": f"{name}.cubin", "sha256": kernels[name]["sha256"]}

    for c in calls:
        if c["op"] == NEW_OP:
            assert len(c["args"]) == 15, (c["label"], len(c["args"]))
    n = sum(1 for c in calls if c["op"] == NEW_OP)
    out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"land_add2 folded into the push on {n} layers ({len(calls)} calls, "
          f"{len(m['ops'])} ops, {len(m['modules'])} modules) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
