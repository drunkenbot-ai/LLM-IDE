#!/usr/bin/env python3
"""Prepare the private Python runtime used by the installer."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Runtime directory to create.")
    parser.add_argument("--gpu", action="store_true", help="Install the CUDA runtime selected by runtime_setup.py.")
    args = parser.parse_args()
    runtime = args.output.resolve()
    # Force a real interpreter copy. Some CI Python installations expose a
    # launcher that points at the runner's temporary hostedtoolcache path.
    venv_command = [sys.executable, "-m", "venv"]
    if sys.platform == "win32":
        venv_command.append("--copies")
    subprocess.run([*venv_command, str(runtime)], check=True)
    python = runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    pip_check = subprocess.run([str(python), "-m", "pip", "--version"], check=False)
    if pip_check.returncode != 0:
        subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    requirements = ROOT / "requirements.txt"
    filtered = runtime / "requirements-base.txt"
    filtered.write_text(
        "\n".join(
            line for line in requirements.read_text(encoding="utf-8").splitlines()
            if line.strip().lower() not in {"torch", "torch==", "torchvision"}
        ) + "\n",
        encoding="utf-8",
    )
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(filtered)], check=True)
    filtered.unlink()
    for cache_name in ("__pycache__",):
        for path in runtime.rglob(cache_name):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
