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
        return root / "runtime" / "Scripts" / "python.exe"
    return root / "runtime" / "bin" / "python"


def launch(root: Path) -> int:
    interpreter = private_python(root)
    script = root / "run_app.py"
    setup_script = root / "runtime_setup.py"
    if not interpreter.exists():
        raise RuntimeError(f"Private Python runtime is missing: {interpreter}")
    if not script.exists():
        raise RuntimeError(f"Application entry point is missing: {script}")
    if not setup_script.exists():
        raise RuntimeError(f"Runtime setup is missing: {setup_script}")
    environment = os.environ.copy()
    environment["DRUNKENBOT_APP_ROOT"] = str(root)
    setup_result = subprocess.run(
        [str(interpreter), str(setup_script), "--ensure"],
        cwd=root,
        env=environment,
        check=False,
    )
    if setup_result.returncode != 0:
        raise RuntimeError(
            "Hardware runtime setup failed. Run runtime_setup.py manually "
            "from the installation folder to see the detailed installer error."
        )
    return subprocess.call([str(interpreter), str(script)], cwd=root, env=environment)


if __name__ == "__main__":
    application_root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    if application_root.name == "_internal":
        application_root = application_root.parent
    raise SystemExit(launch(application_root))
