#!/usr/bin/env python3
"""compose.py COMMIT...: replay adopted commits from the agents' branches onto main, one at a time.

Run inside the eval container, in a worktree checked out on main with a clean tree. For
each commit, in the order given: every file it touched except the manifests is taken from
it (kernels.toml as the union of main's entries and the commit's), its scripts/gen_*.py is
run against main's current default manifest to regenerate the candidate, which then becomes
the default; scripts/check.py build; bench16k twice each of the previous default and the
candidate, alternating; judge16k on all 48 prompts. A candidate that passes and gains more
than 1 ms is committed to main (-s) with the original subject and the measured numbers; one
that does not is reverted and reported. The report goes to stdout and to --out.
"""
import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path.cwd()
DEFAULT = Path("manifests/k3-tp4-prefill-16k.json")


def sh(*args, check=True, capture=True):
    r = subprocess.run(args, text=True, capture_output=capture)
    if check and r.returncode:
        sys.exit(f"{' '.join(args)}\n{r.stdout}{r.stderr}")
    return r.stdout.strip() if capture else ""


def p50(report):
    return max(s["graph"]["stats"]["p50"] for s in json.load(open(report))["scenarios"]) / 1000


def union_kernels(commit):
    main = tomllib.loads(Path("kernels.toml").read_text())["kernels"]
    theirs_text = sh("git", "show", f"{commit}:kernels.toml")
    theirs = tomllib.loads(theirs_text)["kernels"]
    new = [k for k in theirs if k not in main]
    if not new:
        return []
    blocks = re.split(r"(?m)^(?=\[kernels\.)", theirs_text)
    keep = [b for b in blocks if any(b.startswith(f'[kernels."{k}"]') or b.startswith(f"[kernels.{k}]") for k in new)]
    text = Path("kernels.toml").read_text().rstrip("\n") + "\n\n" + "\n".join(b.rstrip("\n") + "\n" for b in keep)
    Path("kernels.toml").write_text(text)
    return new


def replay(commit, out_dir):
    subject = sh("git", "log", "-1", "--format=%s", commit)
    files = sh("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit).split()
    gens = [f for f in files if re.fullmatch(r"scripts/gen_.*\.py", f)]
    if len(gens) != 1:
        return {"commit": commit, "subject": subject, "status": "skipped", "why": f"{len(gens)} generator scripts in the commit"}
    taken = [f for f in files if not f.startswith("manifests/") and f != "kernels.toml"]
    if taken:
        sh("git", "checkout", commit, "--", *taken)
    new_kernels = union_kernels(commit) if "kernels.toml" in files else []
    shutil.copy(DEFAULT, out_dir / "default-before.json")
    candidate = sh("python3", gens[0]).splitlines()[-1]
    cand_path = Path("manifests") / candidate
    if not cand_path.exists():
        return revert(commit, subject, f"{gens[0]} did not produce manifests/{candidate}")
    shutil.move(cand_path, DEFAULT)
    r = subprocess.run(["python3", "scripts/check.py", "build"], text=True, capture_output=True)
    if r.returncode:
        return revert(commit, subject, f"check.py build failed:\n{r.stdout}{r.stderr}"[-2000:])
    base, cand = [], []
    for i in (1, 2):
        sh("bench16k", str(out_dir / "default-before.json"), str(out_dir / f"base{i}.json"), capture=False)
        base.append(p50(out_dir / f"base{i}.json"))
        sh("bench16k", str(DEFAULT), str(out_dir / f"cand{i}.json"), capture=False)
        cand.append(p50(out_dir / f"cand{i}.json"))
    gain = statistics.mean(base) - statistics.mean(cand)
    judge = subprocess.run(["judge16k", str(DEFAULT), str(out_dir / "judge.json")], text=True, capture_output=True)
    passed = judge.returncode == 0
    verdict = [l for l in judge.stdout.splitlines() if l.startswith(("PASS", "FAIL", "INCONCLUSIVE"))]
    numbers = f"default {' / '.join(f'{x:.3f}' for x in base)} ms, candidate {' / '.join(f'{x:.3f}' for x in cand)} ms, {gain:+.3f} ms; judge16k {' '.join(verdict)[:160]}"
    if not passed or gain < 1.0:
        return revert(commit, subject, f"not adopted: {numbers}")
    body = sh("git", "log", "-1", "--format=%b", commit)
    message = f"{subject}\n\n{body}\n\nComposed onto main from {commit[:7]}: {numbers}"
    sh("git", "add", "-A")
    sh("git", "commit", "-q", "-s", "-m", message)
    return {"commit": commit, "subject": subject, "status": "adopted", "numbers": numbers, "new_kernels": new_kernels, "main": sh("git", "rev-parse", "--short", "HEAD")}


def revert(commit, subject, why):
    sh("git", "checkout", "--", ".")
    sh("git", "clean", "-fdq", "--", "manifests", "source", "prebuilt", "scripts", "results")
    return {"commit": commit, "subject": subject, "status": "rejected", "why": why}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("commits", nargs="+")
    ap.add_argument("--out", default="compose")
    args = ap.parse_args()
    if sh("git", "status", "--porcelain"):
        sys.exit("the tree is not clean")
    if sh("git", "rev-parse", "--abbrev-ref", "HEAD") != "main":
        sys.exit("compose on main")
    results = []
    for commit in args.commits:
        out_dir = Path(args.out) / commit[:7]
        out_dir.mkdir(parents=True, exist_ok=True)
        results.append(replay(sh("git", "rev-parse", commit), out_dir))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    Path(args.out, "report.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
