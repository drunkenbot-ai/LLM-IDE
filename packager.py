#!/usr/bin/env python3
"""Build a self-contained distributable for the current operating system.

PyInstaller must be installed in the build environment. Windows builds are
architecture-specific: build once with 32-bit Python and once with 64-bit
Python to produce both installers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "packaging" / "artifacts"
APP_NAME = "DrunkenBot-LLM-IDE"


def _architecture() -> str:
    return "x64" if sys.maxsize > 2**32 else "x86"


def _platform_name() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    if system == "darwin":
        return "macos"
    if system == "linux":
        return "linux"
    raise RuntimeError(f"Unsupported operating system: {system}")


def _asset_args(separator: str) -> list[str]:
    assets: list[str] = []
    for relative in ("drunken_bot_logo_small.png", "fonts"):
        source = ROOT / relative
        if source.exists():
            assets.extend(["--add-data", f"{source}{separator}{relative}"])
    return assets


def build(*, clean: bool) -> Path:
    target = _platform_name()
    architecture = _architecture()
    tag = f"{target}-{architecture}"
    work_dir = ROOT / "packaging" / "build" / tag
    dist_dir = OUTPUT_ROOT / tag
    if clean:
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(dist_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    separator = ";" if target == "windows" else ":"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--hidden-import",
        "llm_trainer.ui.app",
        "--collect-submodules",
        "llm_trainer",
        "--name",
        APP_NAME,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(work_dir),
        *_asset_args(separator),
        str(ROOT / "run_app.py"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    bundle = dist_dir / APP_NAME
    if target == "windows":
        installer = OUTPUT_ROOT / f"{APP_NAME}-{architecture}-Setup.exe"
        iscc = shutil.which("ISCC.exe") or shutil.which("iscc")
        if not iscc:
            raise RuntimeError(
                "Inno Setup (ISCC.exe) is required for Windows installers. "
                "Install Inno Setup and rerun the packager."
            )
        script = work_dir / "installer.iss"
        script.write_text(
            "\n".join(
                [
                    "[Setup]",
                    f'AppName={APP_NAME}',
                    f'OutputBaseFilename={APP_NAME}-{architecture}-Setup',
                    f'DefaultDirName={{autopf}}\\{APP_NAME}',
                    f'OutputDir={OUTPUT_ROOT}',
                    "Uninstallable=yes",
                    "Compression=lzma2",
                    "SolidCompression=yes",
                    "[Files]",
                    f'Source: "{bundle}\\*"; DestDir: "{{app}}"; Flags: recursesubdirs ignoreversion',
                    "[Icons]",
                    f'Name: "{{autoprograms}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.exe"',
                    f'Name: "{{commondesktop}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.exe"',
                ]
            ),
            encoding="utf-8",
        )
        subprocess.run([iscc, str(script)], check=True)
        return installer
    elif target == "macos":
        archive = OUTPUT_ROOT / f"{APP_NAME}-macos-{architecture}.zip"
    else:
        archive = OUTPUT_ROOT / f"{APP_NAME}-linux-{architecture}.tar.gz"
    if archive.exists():
        archive.unlink()
    if target == "linux":
        subprocess.run(["tar", "-czf", str(archive), "-C", str(dist_dir), APP_NAME], check=True)
    else:
        shutil.make_archive(str(archive.with_suffix("")), "zip", dist_dir, APP_NAME)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-clean", action="store_true", help="Keep previous PyInstaller intermediates.")
    args = parser.parse_args()
    if not (ROOT / "run_app.py").exists():
        raise RuntimeError("run_app.py was not found")
    artifact = build(clean=not args.no_clean)
    print(f"Created {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
