#!/usr/bin/env python3
"""Install the optional hardware-specific PyTorch runtime."""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


TORCH_VERSION = "2.5.1"
TORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
}


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


def install_runtime(python_executable: str, choice: RuntimeChoice) -> None:
    subprocess.run(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            f"torch=={TORCH_VERSION}",
            "--index-url",
            TORCH_INDEXES[choice.profile],
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Private packaged Python executable.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    choice = choose_runtime()
    print(f"Selected {choice.profile}: {choice.reason}")
    if not args.dry_run:
        install_runtime(args.python, choice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
