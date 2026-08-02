#!/usr/bin/env python3
"""Install the optional hardware-specific PyTorch runtime."""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass


TORCH_VERSION = "2.5.1"
TORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
}
PROFILE_FILE = "torch-runtime-profile.txt"


@dataclass(frozen=True)
class RuntimeChoice:
    profile: str
    reason: str


def _nvidia_driver_major() -> int | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    match = re.search(r"(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def choose_runtime(system: str | None = None, driver_major: int | None = None) -> RuntimeChoice:
    system = system or platform.system()
    if system not in {"Windows", "Linux"}:
        return RuntimeChoice("cpu", f"{system} does not support the CUDA runtime")
    driver_major = _nvidia_driver_major() if driver_major is None else driver_major
    if driver_major is None:
        return RuntimeChoice("cpu", "NVIDIA driver was not detected")
    if driver_major >= 550:
        return RuntimeChoice("cu124", f"NVIDIA driver {driver_major} supports CUDA 12.4")
    if driver_major >= 525:
        return RuntimeChoice("cu121", f"NVIDIA driver {driver_major} supports CUDA 12.1")
    return RuntimeChoice("cpu", f"NVIDIA driver {driver_major} is too old for supported CUDA wheels")


_SETUP_LOG = None


def install_runtime(python_executable: str, choice: RuntimeChoice) -> None:
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        item for item in environment.get("PATH", "").split(os.pathsep)
        if "mingw" not in item.lower() and "scoop" not in item.lower()
    )
    process = subprocess.Popen(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            f"torch=={TORCH_VERSION}",
            "--index-url",
            TORCH_INDEXES[choice.profile],
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", file=sys.stdout, flush=True)
        print(line, end="", file=_SETUP_LOG, flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, process.args)


def ensure_runtime(python_executable: str, root: Path) -> RuntimeChoice:
    choice = choose_runtime()
    marker = root / PROFILE_FILE
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == choice.profile:
        return choice
    install_runtime(python_executable, choice)
    verification = subprocess.run(
        [python_executable, "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if verification.returncode != 0:
        raise RuntimeError(f"Torch verification failed: {verification.stderr.strip()}")
    marker.write_text(choice.profile, encoding="utf-8")
    return choice


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Private packaged Python executable.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ensure", action="store_true")
    args = parser.parse_args()
    log_path = Path(__file__).resolve().parent / "runtime_setup.log"
    with log_path.open("a", encoding="utf-8") as log:
        global _SETUP_LOG
        _SETUP_LOG = log
        try:
            print(f"Runtime setup starting with Python {sys.version}", flush=True)
            print(f"Runtime setup starting with Python {sys.version}", file=log, flush=True)
            choice = choose_runtime()
            print(f"Selected {choice.profile}: {choice.reason}", flush=True)
            print(f"Selected {choice.profile}: {choice.reason}", file=log, flush=True)
            if args.ensure:
                choice = ensure_runtime(args.python, Path(__file__).resolve().parent)
            elif not args.dry_run:
                install_runtime(args.python, choice)
            print(f"Runtime setup completed: {choice.profile}", flush=True)
            print(f"Runtime setup completed: {choice.profile}", file=log, flush=True)
        except Exception as exc:
            print(f"Runtime setup failed: {exc!r}", flush=True)
            print(f"Runtime setup failed: {exc!r}", file=log, flush=True)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
