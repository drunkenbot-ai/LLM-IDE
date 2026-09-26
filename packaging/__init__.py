"""Build and bundled-runtime helpers for the application."""
import sys
import importlib.util
from pathlib import Path

# Forward system packaging.version if third-party libraries (like huggingface datasets) need it
try:
    for _p in sys.path[1:]:
        _cand = Path(_p) / "packaging" / "version.py"
        if _cand.exists():
            _spec = importlib.util.spec_from_file_location("packaging.version", str(_cand))
            if _spec and _spec.loader:
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                version = _mod
                sys.modules["packaging.version"] = _mod
                break
except Exception:
    pass
