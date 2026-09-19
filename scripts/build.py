#!/usr/bin/env python3
import hashlib
import os
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    out = ROOT / "build"
    out.mkdir(exist_ok=True)
    mismatches = []
    for name, kernel in kernels.items():
        if "source" not in kernel or "build" in kernel:
            continue
        dst = out / f"{name}.cubin"
        defines = [f"-D{k}={v}" for k, v in sorted(kernel.get("defines", {}).items())]
        subprocess.run([os.environ.get("NVCC", "nvcc"), "-cubin", "-arch=sm_103a",
                        *defines, "-o", str(dst), kernel["source"]], cwd=ROOT, check=True)
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        matched = sha == kernel["sha256"]
        print(f"{name}: {sha} ({'matches manifest' if matched else 'different from manifest'})", flush=True)
        if not matched:
            mismatches.append(name)
    if mismatches:
        raise SystemExit("Built cubins differ from the pinned manifest: " + ", ".join(mismatches))


if __name__ == "__main__":
    main()
