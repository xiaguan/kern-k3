#!/usr/bin/env python3
import hashlib
from pathlib import Path
import shutil

cache = Path.home() / ".cache/kern/blobs"
cache.mkdir(parents=True, exist_ok=True)
for cubin in Path("build").glob("*.cubin"):
    shutil.copyfile(cubin, cache / hashlib.sha256(cubin.read_bytes()).hexdigest())
