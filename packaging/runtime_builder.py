#!/usr/bin/env python3
"""Prepare the private Python runtime used by the installer."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
HOSTEDTOOLCACHE_MARKER = b"hostedtoolcache"


def _copy_windows_runtime(runtime: Path) -> Path:
    """Copy the complete Python installation so the runtime is self-contained."""
    source = Path(sys.base_prefix).resolve()
    if source == runtime or source in runtime.parents:
        raise RuntimeError(f"Cannot copy Python installation into itself: {source}")
    shutil.copytree(source, runtime, dirs_exist_ok=True)
    return runtime / "python.exe"


def _validate_windows_runtime(runtime: Path, python: Path) -> None:
    """Reject Windows runtimes that retain a hosted runner interpreter path."""
    candidates = [python, *runtime.glob("pyvenv.cfg"), *runtime.glob("*._pth")]
    candidates.extend(runtime.glob("*.pth"))
    for path in candidates:
        if not path.is_file():
            continue
        if HOSTEDTOOLCACHE_MARKER in path.read_bytes().lower():
            raise RuntimeError(
                f"Generated Windows runtime contains a hostedtoolcache reference in {path}. "
                "Use a self-contained Python installation instead of a hosted runner launcher."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Runtime directory to create.")
    parser.add_argument("--gpu", action="store_true", help="Install the CUDA runtime selected by runtime_setup.py.")
    args = parser.parse_args()
    runtime = args.output.resolve()
    if sys.platform == "win32":
        python = _copy_windows_runtime(runtime)
        _validate_windows_runtime(runtime, python)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(runtime)], check=True)
        python = runtime / "bin/python"
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
