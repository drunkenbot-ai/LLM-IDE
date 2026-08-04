#!/usr/bin/env python3
"""Launch the application with its private Python runtime."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import sys


def setup_log_path(root: Path) -> Path:
    """Return the log path shared by setup and launcher diagnostics."""
    configured = os.environ.get("DRUNKENBOT_RUNTIME_SETUP_LOG")
    if configured:
        return Path(configured)
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Logs"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "DrunkenBot-IDE" / "runtime_setup.log"


def private_python(root: Path) -> Path:
    if platform.system() == "Windows":
        return root / "runtime" / "python.exe"
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
    environment["PYTHONNOUSERSITE"] = "1"
    environment["DRUNKENBOT_APP_ROOT"] = str(root)
    log_path = setup_log_path(root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment["DRUNKENBOT_RUNTIME_SETUP_LOG"] = str(log_path)
    setup_kwargs: dict[str, object] = {
        "cwd": root,
        "env": environment,
        "check": False,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    setup_result = subprocess.run(
        [str(interpreter), str(setup_script), "--ensure"],
        **setup_kwargs,
    )
    with log_path.open("a", encoding="utf-8") as log:
        if setup_result.stdout:
            log.write("\n[Launcher] Runtime setup output:\n")
            log.write(setup_result.stdout)
            log.flush()
    verification = subprocess.run(
        [str(interpreter), "-c", "import torch; print(torch.__version__)"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            "\n[Launcher] Torch verification: "
            f"exit={verification.returncode} "
            f"stdout={verification.stdout.strip()!r} "
            f"stderr={verification.stderr.strip()!r}\n"
        )
    if setup_result.returncode != 0 or verification.returncode != 0:
        details = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(
            f"Hardware runtime setup failed (setup exit code {setup_result.returncode}, "
            f"verification exit code {verification.returncode}).\n"
            f"Detailed log: {log_path}\n{details}"
        )
    return subprocess.call([str(interpreter), str(script)], cwd=root, env=environment)


if __name__ == "__main__":
    application_root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    if application_root.name == "_internal":
        application_root = application_root.parent
    raise SystemExit(launch(application_root))
