# Building installers

## Build prerequisites

- Windows 64-bit for the Windows installer.
- Python 3.12 or newer for the build environment.
- Inno Setup, with `ISCC.exe` available on `PATH`.
- A working C/C++ toolchain for packages that need native wheels.
- Internet access for Python packages and PyTorch wheels.
- Sufficient disk space for the private runtime, PyInstaller output, and
  optional CUDA packages.

Install the Python build dependencies in a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
```

The application dependencies in `requirements.txt` are installed into the
private runtime by the packager. The build environment only needs them so
PyInstaller can inspect imports and collect native modules.

## One-command Windows builds

Build the CPU installer:

```bash
python packager.py
```

Build the CUDA-enabled installer:

```bash
python packager.py --gpu
```

The packager creates a private runtime, installs the pinned dependencies,
builds the launcher bundle, copies `engine/` and `interface/` directly along
with application assets and fonts, runs Inno Setup, and writes the installer to
`packaging/artifacts/`. The historical `llm_trainer/` compatibility package is
kept in the source tree but is not the primary packaged application path.

The CUDA build detects the NVIDIA driver using `nvidia-smi`. It selects the
supported PyTorch wheel index and falls back to CPU when no compatible driver
is found.

## Reusing a prepared runtime

To avoid rebuilding the private runtime:

```bash
python packager.py --runtime-dir packaging/runtime
```

Use `--no-clean` to retain PyInstaller intermediates while troubleshooting.

## Included and excluded files

The installer includes the private Python runtime, application code, third
party packages, fonts, logo images, and other application assets. Training corpus data is not bundled; users download or select it through the
Dataset Sources page after installation.

Run `python tools/check_dependency_boundaries.py` before packaging to verify
that the non-Qt engine does not import the desktop interface and that new
interface code does not depend on the legacy package.

Build intermediates and installers are written under `packaging/` and are
ignored by Git.

## Architecture

Windows builds use the architecture of the Python interpreter running the
packager. Use a 64-bit Python installation for the supported Windows build.
GPU PyTorch and CUDA are not supported by the 32-bit build.
