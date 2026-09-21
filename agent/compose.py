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
BEFORE = Path("build-before")
NOTES = None
PAIRS, GAIN = 2, 1.0


def sh(*args, check=True, capture=True):
    r = subprocess.run(args, text=True, capture_output=capture)
    if check and r.returncode:
        sys.exit(f"{' '.join(args)}\n{r.stdout}{r.stderr}")
    return r.stdout.strip() if capture else ""


def p50(report):
    return max(s["graph"]["stats"]["p50"] for s in json.load(open(report))["scenarios"]) / 1000


def blocks(text):
    """kernels.toml as {name: block text}, in file order."""
    out = {}
    for b in re.split(r"(?m)^(?=\[kernels\.)", text):
        m = re.match(r'\[kernels\.(?:"([^"]+)"|([^\]]+))\]', b)
        if m:
            out[m.group(1) or m.group(2)] = b.rstrip("\n") + "\n"
    return out


def union_kernels(commit):
    """Main's entries, with the commit's version of every entry the commit has (a rebuilt module keeps its name and changes its sha); the changed names are returned."""
    mine_text = Path("kernels.toml").read_text()
    mine, theirs = blocks(mine_text), blocks(sh("git", "show", f"{commit}:kernels.toml"))
    base = blocks(sh("git", "show", f"{commit}^:kernels.toml"))
    changed = [k for k, b in theirs.items() if base.get(k) != b]
    if not changed:
        return []
    head = re.split(r"(?m)^(?=\[kernels\.)", mine_text)[0]
    merged = {**mine, **{k: theirs[k] for k in changed}}
    Path("kernels.toml").write_text(head + "\n".join(merged.values()))
    return changed


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
    gens = [f for f in files if re.fullmatch(r"scripts/gen_.*\.py", f)]
    if not gens or len([g for g in gens if g in added]) > 1:
        return {"commit": commit, "subject": subject, "status": "skipped", "why": f"{len(gens)} generator scripts touched, {len([g for g in gens if g in added])} added"}
    # only what the manifest needs to be reproduced: kernel sources, the generators, the kernel entries;
    # a branch's README, check harnesses and probes stay in its own record
    taken = [f for f in files if f.startswith(("source/", "prebuilt/"))] + gens
    dropped = [f for f in files if f not in taken and not f.startswith("manifests/") and f != "kernels.toml"]
    sh("git", "checkout", commit, "--", *taken)
    new_kernels = union_kernels(commit) if "kernels.toml" in files else []
    shutil.copy(DEFAULT, out_dir / "default-before.json")
    # a generator may read its module's sha from build/, so the new entries are built first;
    # a rebuilt module keeps its file name, so the previous default benches from a snapshot
    shutil.rmtree(BEFORE, ignore_errors=True)
    shutil.copytree("build", BEFORE)
    r = subprocess.run(["python3", "scripts/build.py"], text=True, capture_output=True)
    if r.returncode:
        return revert(commit, subject, f"scripts/build.py failed:\n{r.stdout}{r.stderr}"[-2000:])
    # a generator the commit modified is an earlier step it builds on (idempotent by contract):
    # replay those first, then the one it added, each on what the previous one produced;
    # a commit may also just change what an existing generator produces and add none
    start = DEFAULT.read_bytes()
    for gen in sorted(gens, key=lambda g: g in added):
        r = subprocess.run(["python3", gen], text=True, capture_output=True)
        produced = sh("git", "ls-files", "--others", "--exclude-standard", "manifests").split()
        if len(produced) == 1:
            shutil.move(produced[0], DEFAULT)
        if r.returncode or len(produced) > 1:
            return revert(commit, subject, f"{gen}: exit {r.returncode}, produced {produced}\n{r.stdout}{r.stderr}"[-2000:])
    # a generator may rewrite the default in place; the chain as a whole must have changed it
    if DEFAULT.read_bytes() == start:
        return revert(commit, subject, f"{gens}: replayed, the default is unchanged")
    r = subprocess.run(["python3", "scripts/check.py", "build"], text=True, capture_output=True)
    if r.returncode:
        return revert(commit, subject, f"scripts/check.py build failed:\n{r.stdout}{r.stderr}"[-2000:])
    base, cand = [], []
    for i in range(1, PAIRS + 1):
        Path("build").rename("build-candidate")
        BEFORE.rename("build")
        try:
            evaluate("bench16k", out_dir / "default-before.json", out_dir / f"base{i}.json")
        finally:
            Path("build").rename(BEFORE)
            Path("build-candidate").rename("build")
        base.append(p50(out_dir / f"base{i}.json"))
        evaluate("bench16k", DEFAULT, out_dir / f"cand{i}.json")
        cand.append(p50(out_dir / f"cand{i}.json"))
    shutil.rmtree(BEFORE)
    gain = statistics.mean(base) - statistics.mean(cand)
    judge = evaluate("judge16k", DEFAULT, out_dir / "judge.json")
    passed = judge.returncode == 0
    verdict = [l for l in judge.stdout.splitlines() if l.startswith(("PASS", "FAIL", "INCONCLUSIVE"))]
    numbers = f"default {' / '.join(f'{x:.3f}' for x in base)} ms, candidate {' / '.join(f'{x:.3f}' for x in cand)} ms, {gain:+.3f} ms; judge16k {' '.join(verdict)[:160]}"
    if not passed or gain < GAIN:
        return revert(commit, subject, f"not adopted: {numbers}")
    body = sh("git", "log", "-1", "--format=%b", commit).split("\nSigned-off-by:")[0].rstrip()
    note = Path(NOTES, f"{commit[:7]}.md") if NOTES else None
    trail = f"\n\n{note.read_text().strip()}" if note and note.exists() else ""
    message = f"{subject}\n\n{body}{trail}\n\nComposed onto main from {commit[:7]}: {numbers}"
    sh("git", "add", "-A", "--", "source", "prebuilt", "scripts", "kernels.toml", "manifests")
    sh("git", "commit", "-q", "-s", "-m", message)
    return {"commit": commit, "subject": subject, "status": "adopted", "numbers": numbers, "new_kernels": new_kernels, "dropped": dropped, "main": sh("git", "rev-parse", "--short", "HEAD")}


def revert(commit, subject, why):
    shutil.rmtree(BEFORE, ignore_errors=True)
    sh("git", "reset", "-q", "--hard", "HEAD")
    sh("git", "clean", "-fdq", "--", "manifests", "source", "prebuilt", "scripts", "results")
    return {"commit": commit, "subject": subject, "status": "rejected", "why": why}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("commits", nargs="+")
    ap.add_argument("--out", default="compose")
    ap.add_argument("--pairs", type=int, default=2, help="interleaved default/candidate bench pairs")
    ap.add_argument("--gain", type=float, default=1.0, help="ms of mean gain below which a PASS is not adopted")
    ap.add_argument("--notes", help="directory of <sha7>.md files appended to the adopted commit's message: what was tried on the way")
    args = ap.parse_args()
    global NOTES, PAIRS, GAIN
    NOTES, PAIRS, GAIN = args.notes, args.pairs, args.gain
    if sh("git", "status", "--porcelain"):
        sys.exit("the tree is not clean")
    if sh("git", "rev-parse", "--abbrev-ref", "HEAD") != "main":
        sys.exit("compose on main")
    sh("python3", "scripts/build.py")  # build/ is untracked: bring it back to main's kernels
    results = []
    for commit in args.commits:
        out_dir = Path(args.out) / commit[:7]
        out_dir.mkdir(parents=True, exist_ok=True)
        results.append(replay(sh("git", "rev-parse", commit), out_dir))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    Path(args.out, "report.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
