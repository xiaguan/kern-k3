#!/usr/bin/env python3
"""Build corpus.json for `kern test --record`: prose, code and legal text
cut into prompts of a few thousand characters, mixed so no two neighbours
come from the same source. Sources are public files; nothing here is
model-specific, the tokenizer runs inside kern."""
import argparse
import json
import pathlib
import random

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=pathlib.Path(__file__).with_name("corpus.json"))
ap.add_argument("--chars", type=int, default=6000, help="characters per prompt")
ap.add_argument("--prompts", type=int, default=48)
ap.add_argument("--decode", type=int, default=16, help="positions fed one at a time at each prompt's tail")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--exclude", nargs="*", default=["tray0", "tray1", "pod4", "/mnt/", "/home/"], help="drop a piece containing any of these")
ap.add_argument("sources", nargs="+", help="files or directories (*.md, *.rs, *.py, *.txt, license texts)")
a = ap.parse_args()

def files(p):
    p = pathlib.Path(p)
    if p.is_file():
        return [p]
    ok = lambda f: f.suffix in {".md", ".rs", ".py", ".txt"} or f.parent.name == "common-licenses"
    return sorted(f for f in p.rglob("*") if f.is_file() and ok(f) and "node_modules" not in f.parts and f.stat().st_size > a.chars)

def prose(text):
    return sum(c.isprintable() or c in "\n\t" for c in text) > 0.98 * len(text)

pieces = []
for src in a.sources:
    for f in files(src):
        text = f.read_text(errors="ignore")
        if not prose(text):
            continue
        for i in range(0, len(text) - a.chars, a.chars):
            piece = text[i:i + a.chars]
            if not any(x in piece for x in a.exclude):
                pieces.append((str(f), piece))
rng = random.Random(a.seed)
rng.shuffle(pieces)
chosen, last = [], None
for src, text in pieces:
    if src == last:
        continue
    chosen.append(text)
    last = src
    if len(chosen) == a.prompts:
        break
json.dump({"decode": a.decode, "prompts": chosen}, open(a.out, "w"), ensure_ascii=False, indent=0)
print(f"{len(chosen)} prompts of {a.chars} chars, decode {a.decode} -> {a.out}")
