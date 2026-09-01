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
python packaging/packager.py
```

Build the CUDA-enabled installer:

```bash
python packaging/packager.py --gpu
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
python packaging/packager.py --runtime-dir packaging/runtime
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

### Standalone training process

Local pretraining and fine-tuning share
`interface.training_process_controller.TrainingProcessController`. It creates
an engine `TrainingJobSpec`, atomically writes the versioned worker request,
launches `python -m engine.training_worker --request <absolute-path>`, and
supervises the durable manifest and SQLite telemetry with a 750 ms `QTimer`.
Training is not executed or supervised by `TaskWorker`/`QThread`; unrelated
background UI work continues to use the existing task runner.

The engine worker owns the output-directory lock, manifest heartbeat,
notification delivery, and batched WAL telemetry writes. The UI reads metrics
and events incrementally by row ID, bounds in-memory history, coalesces metric
rendering to one snapshot per refresh, and skips Live chart painting while the
page is hidden. Window close only detaches the timer, allowing the process to
survive and be identity-checked when the project reopens.

The project file persists a versioned reference containing the returned run
ID, manifest path, control path, and telemetry database path. Reattach tries
that exact manifest first and falls back to scanning configured model and
fine-tune output directories. Metric and event readers maintain independent
last-row-ID cursors across each attached UI session.

Forced termination is fail-closed: the UI first writes a cooperative stop
control and only signals after the timeout when
`manifest_process_is_current()` confirms both PID and process creation
identity. Never replace this check with a numeric PID lookup.

This integration requires engine PR #11
(`e1862f574ff2a809f8d0b25794956376c132e7b5`) to merge before the LLM-IDE
change. That revision uses a persistent cross-platform mutex for stale claim
recovery and fails closed when exclusive mutation ownership cannot be proven.
It also emits one bounded dataset diagnostic per affected source and one
terminal completion event, allowing the UI to report large malformed JSON/JSONL
inputs without per-record log floods while still completing partial builds.
