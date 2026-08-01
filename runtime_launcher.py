#!/usr/bin/env python3
"""Launch the application with its private Python runtime."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import sys


def private_python(root: Path) -> Path:
    if platform.system() == "Windows":
        return root / "runtime" / "python.exe"
    return root / "runtime" / "bin" / "python"


def launch(root: Path) -> int:
    interpreter = private_python(root)
    script = root / "run_app.py"
    if not interpreter.exists():
        raise RuntimeError(f"Private Python runtime is missing: {interpreter}")
    if not script.exists():
        raise RuntimeError(f"Application entry point is missing: {script}")
    environment = os.environ.copy()
    environment["DRUNKENBOT_APP_ROOT"] = str(root)
    return subprocess.call([str(interpreter), str(script)], cwd=root, env=environment)


if __name__ == "__main__":
    raise SystemExit(launch(Path(__file__).resolve().parent))
