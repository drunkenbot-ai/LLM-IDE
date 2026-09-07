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


def _log_path(root: Path) -> Path:
    """Choose a writable setup-log location on installed systems."""
    system = platform.system()
    if system == "Windows":
        platform_log = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "Darwin":
        platform_log = Path.home() / "Library" / "Logs"
    else:
        platform_log = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    candidates = [
        platform_log / "DrunkenBot-IDE" / "runtime_setup.log",
        root / "runtime_setup.log",
    ]
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with candidate.open("a", encoding="utf-8"):
                pass
            return candidate
        except OSError:
            continue
    raise OSError("Could not create a writable runtime setup log.")


def get_triton_specifier(torch_version: str) -> str:
    """Return the compatible triton-windows requirement for a given PyTorch version.

    The official triton-windows wheels correspond to PyTorch releases:
    - PyTorch 2.5.x -> triton-windows 3.1.x
    - PyTorch 2.6.x -> triton-windows 3.2.x
    - PyTorch 2.7.x -> triton-windows 3.3.x
    - PyTorch 2.8.x -> triton-windows 3.4.x
    - PyTorch 2.9.x -> triton-windows 3.5.x
    - PyTorch 2.10.x -> triton-windows 3.6.x

    Args:
        torch_version: PyTorch version string (e.g. '2.5.1').

    Returns:
        A pip-compatible requirement specifier for triton-windows.
    """
    match = re.match(r"^(\d+)\.(\d+)", torch_version)
    if not match:
        return "triton-windows"
    major, minor = int(match.group(1)), int(match.group(2))
    if major == 2 and minor >= 5:
        triton_minor = minor - 4
        return f"triton-windows>=3.{triton_minor}.0,<3.{triton_minor + 1}.0"
    if major == 2 and minor == 4:
        return "triton-windows<3.1.0"
    return "triton-windows"


def _run_pip(python_executable: str, args: list[str], environment: dict[str, str]) -> None:
    """Execute a pip command streaming output to stdout and the setup log.

    Args:
        python_executable: Path to the Python executable.
        args: Command-line arguments passed to pip.
        environment: Environment variable mapping.

    Raises:
        subprocess.CalledProcessError: If the pip command exits with non-zero status.
    """
    command = [python_executable, "-m", "pip", *args]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", file=sys.stdout, flush=True)
        if _SETUP_LOG is not None:
            print(line, end="", file=_SETUP_LOG, flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def install_runtime(python_executable: str, choice: RuntimeChoice) -> None:
    """Install hardware-specific PyTorch and optional Triton runtime wheels.

    Args:
        python_executable: Path to the Python executable.
        choice: Selected runtime choice.
    """
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = os.pathsep.join(
        item for item in environment.get("PATH", "").split(os.pathsep)
        if "mingw" not in item.lower() and "scoop" not in item.lower()
    )
    _run_pip(
        python_executable,
        [
            "install",
            "--no-warn-script-location",
            f"torch=={TORCH_VERSION}",
            "--index-url",
            TORCH_INDEXES[choice.profile],
        ],
        environment,
    )

    if platform.system() == "Windows" and choice.profile != "cpu":
        triton_spec = get_triton_specifier(TORCH_VERSION)
        msg = f"Installing {triton_spec} for Windows CUDA runtime...\n"
        print(msg, end="", file=sys.stdout, flush=True)
        if _SETUP_LOG is not None:
            print(msg, end="", file=_SETUP_LOG, flush=True)
        try:
            _run_pip(
                python_executable,
                [
                    "install",
                    "--no-warn-script-location",
                    triton_spec,
                ],
                environment,
            )
        except Exception as exc:
            warning = f"Warning: Failed to install {triton_spec}: {exc!r}. Running without Triton.\n"
            print(warning, end="", file=sys.stdout, flush=True)
            if _SETUP_LOG is not None:
                print(warning, end="", file=_SETUP_LOG, flush=True)


def ensure_runtime(python_executable: str, root: Path) -> RuntimeChoice:
    """Ensure the expected runtime wheels are installed and verified.

    Args:
        python_executable: Path to the Python executable.
        root: Root directory containing PROFILE_FILE.

    Returns:
        The selected RuntimeChoice.
    """
    choice = choose_runtime()
    marker = root / PROFILE_FILE
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == choice.profile:
        if platform.system() == "Windows" and choice.profile != "cpu":
            triton_check = subprocess.run(
                [python_executable, "-c", "import triton"],
                cwd=root,
                capture_output=True,
                check=False,
            )
            if triton_check.returncode != 0:
                install_runtime(python_executable, choice)
        return choice

    install_runtime(python_executable, choice)
    verification = subprocess.run(
        [python_executable, "-c", "import torch; print(torch.__version__)"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    print(
        f"Torch verification exit code: {verification.returncode}\n"
        f"Torch verification output: {verification.stdout.strip()}\n"
        f"Torch verification error: {verification.stderr.strip()}",
        file=_SETUP_LOG,
        flush=True,
    )
    if verification.returncode != 0:
        raise RuntimeError(f"Torch verification failed: {verification.stderr.strip()}")

    if platform.system() == "Windows" and choice.profile != "cpu":
        triton_ver = subprocess.run(
            [python_executable, "-c", "import triton; print(triton.__version__)"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        print(
            f"Triton verification exit code: {triton_ver.returncode}\n"
            f"Triton verification output: {triton_ver.stdout.strip()}\n"
            f"Triton verification error: {triton_ver.stderr.strip()}",
            file=_SETUP_LOG,
            flush=True,
        )

    marker.write_text(choice.profile, encoding="utf-8")
    return choice


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Private packaged Python executable.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ensure", action="store_true")
    args = parser.parse_args()
    configured_log = os.environ.get("DRUNKENBOT_RUNTIME_SETUP_LOG")
    log_path = Path(configured_log) if configured_log else _log_path(Path(__file__).resolve().parent)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DRUNKENBOT_RUNTIME_SETUP_LOG"] = str(log_path)
    os.environ["PYTHONNOUSERSITE"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
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
