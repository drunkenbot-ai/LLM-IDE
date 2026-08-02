#!/usr/bin/env python3
"""Prepare the private Python runtime used by the installer."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Runtime directory to create.")
    parser.add_argument("--gpu", action="store_true", help="Install the CUDA runtime selected by runtime_setup.py.")
    args = parser.parse_args()
    runtime = args.output.resolve()
    subprocess.run([sys.executable, "-m", "venv", str(runtime)], check=True)
    python = runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    requirements = ROOT / "requirements.txt"
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
    if args.gpu:
        subprocess.run([str(python), str(ROOT / "runtime_setup.py")], check=True)
    for cache_name in ("__pycache__", "pip", "setuptools", "wheel"):
        for path in runtime.rglob(cache_name):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
