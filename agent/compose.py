#!/usr/bin/env python3
"""compose.py COMMIT...: replay adopted commits from the agents' branches onto main, one at a time.

Run inside the eval container, in a worktree checked out on main with a clean tree. For
each commit, in the order given: its kernel sources and the generator it added are taken
(kernels.toml as the union of main's entries and the commit's; READMEs, check harnesses and
probes stay on the branch), the generator is run against main's current default manifest to
regenerate the candidate, which then becomes the default; scripts/build.py and check.py build;
bench16k twice each of the previous default and the candidate, alternating; judge16k on all
48 prompts. A candidate that passes and gains more than 1 ms is committed to main (-s) with
the original subject and body, the --notes record of what was tried on the way, and the
measured numbers; one that does not is reverted and reported. The report goes to stdout and
to --out.
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
NOTES = None


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


def evaluate(tool, manifest, out):
    """Run bench16k or judge16k; a run that dies before writing its report (the load phase crashes now and then) is retried once."""
    for _ in range(2):
        r = subprocess.run([tool, str(manifest), str(out)], text=True, capture_output=True)
        if Path(out).exists():
            return r
    sys.exit(f"{tool} {manifest} produced no report twice:\n{r.stdout[-1500:]}{r.stderr[-1500:]}")


def replay(commit, out_dir):
    subject = sh("git", "log", "-1", "--format=%s", commit)
    files = sh("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit).split()
    added = sh("git", "diff-tree", "--no-commit-id", "--name-only", "-r", "--diff-filter=A", commit).split()
    gens = [f for f in (added or files) if re.fullmatch(r"scripts/gen_.*\.py", f)]
    if len(gens) != 1:
        return {"commit": commit, "subject": subject, "status": "skipped", "why": f"{len(gens)} generator scripts to run in the commit"}
    # only what the manifest needs to be reproduced: kernel sources, the generator, the kernel entries;
    # a branch's README, check harnesses and probes stay in its own record
    taken = [f for f in files if f.startswith("source/")] + gens
    dropped = [f for f in files if f not in taken and not f.startswith("manifests/") and f != "kernels.toml"]
    sh("git", "checkout", commit, "--", *taken)
    new_kernels = union_kernels(commit) if "kernels.toml" in files else []
    shutil.copy(DEFAULT, out_dir / "default-before.json")
    candidate = sh("python3", gens[0]).splitlines()[-1]
    cand_path = Path("manifests") / candidate
    if not cand_path.exists():
        return revert(commit, subject, f"{gens[0]} did not produce manifests/{candidate}")
    shutil.move(cand_path, DEFAULT)
    for step in (["python3", "scripts/build.py"], ["python3", "scripts/check.py", "build"]):
        r = subprocess.run(step, text=True, capture_output=True)
        if r.returncode:
            return revert(commit, subject, f"{' '.join(step)} failed:\n{r.stdout}{r.stderr}"[-2000:])
    base, cand = [], []
    for i in (1, 2):
        evaluate("bench16k", out_dir / "default-before.json", out_dir / f"base{i}.json")
        base.append(p50(out_dir / f"base{i}.json"))
        evaluate("bench16k", DEFAULT, out_dir / f"cand{i}.json")
        cand.append(p50(out_dir / f"cand{i}.json"))
    gain = statistics.mean(base) - statistics.mean(cand)
    judge = evaluate("judge16k", DEFAULT, out_dir / "judge.json")
    passed = judge.returncode == 0
    verdict = [l for l in judge.stdout.splitlines() if l.startswith(("PASS", "FAIL", "INCONCLUSIVE"))]
    numbers = f"default {' / '.join(f'{x:.3f}' for x in base)} ms, candidate {' / '.join(f'{x:.3f}' for x in cand)} ms, {gain:+.3f} ms; judge16k {' '.join(verdict)[:160]}"
    if not passed or gain < 1.0:
        return revert(commit, subject, f"not adopted: {numbers}")
    body = sh("git", "log", "-1", "--format=%b", commit).split("\nSigned-off-by:")[0].rstrip()
    note = Path(NOTES, f"{commit[:7]}.md") if NOTES else None
    trail = f"\n\n{note.read_text().strip()}" if note and note.exists() else ""
    message = f"{subject}\n\n{body}{trail}\n\nComposed onto main from {commit[:7]}: {numbers}"
    sh("git", "add", "-A")
    sh("git", "commit", "-q", "-s", "-m", message)
    return {"commit": commit, "subject": subject, "status": "adopted", "numbers": numbers, "new_kernels": new_kernels, "dropped": dropped, "main": sh("git", "rev-parse", "--short", "HEAD")}


def revert(commit, subject, why):
    sh("git", "checkout", "--", ".")
    sh("git", "clean", "-fdq", "--", "manifests", "source", "prebuilt", "scripts", "results")
    return {"commit": commit, "subject": subject, "status": "rejected", "why": why}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("commits", nargs="+")
    ap.add_argument("--out", default="compose")
    ap.add_argument("--notes", help="directory of <sha7>.md files appended to the adopted commit's message: what was tried on the way")
    args = ap.parse_args()
    global NOTES
    NOTES = args.notes
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
