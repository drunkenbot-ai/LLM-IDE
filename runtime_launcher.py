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
    setup_kwargs: dict[str, object] = {
        "cwd": root,
        "env": environment,
        "check": False,
    }
    if platform.system() == "Windows":
        setup_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    setup_result = subprocess.run(
        [str(interpreter), str(setup_script), "--ensure"],
        **setup_kwargs,
    )
    if setup_result.returncode != 0:
        log_path = root / "runtime_setup.log"
        verification = subprocess.run(
            [str(interpreter), "-c", "import torch; print(torch.__version__)"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if verification.returncode == 0:
            return subprocess.call([str(interpreter), str(script)], cwd=root, env=environment)
        details = ""
        if log_path.exists():
            details = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
        raise RuntimeError(
            f"Hardware runtime setup failed (exit code {setup_result.returncode}).\n"
            f"Detailed log: {log_path}\n{details}"
        )
    return subprocess.call([str(interpreter), str(script)], cwd=root, env=environment)


if __name__ == "__main__":
    application_root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    if application_root.name == "_internal":
        application_root = application_root.parent
    raise SystemExit(launch(application_root))
