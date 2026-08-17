from __future__ import annotations

import ast
import builtins
import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


ROOT = Path(__file__).resolve().parents[1]
INTERFACE = ROOT / "interface"


def _interface_modules() -> list[str]:
    return sorted(
        f"interface.{path.relative_to(INTERFACE).with_suffix('').as_posix().replace('/', '.')}"
        for path in INTERFACE.rglob("*.py")
        if path.name != "__init__.py"
    )


def _uses_app_globals_shim(tree: ast.AST) -> bool:
    """True when a module pulls shared runtime names via globals().update(vars(app))."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "update":
            continue
        receiver = node.func.value
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "globals"
        ):
            return True
    return False


def _shim_module_paths() -> list[Path]:
    """Interface modules that receive shared names dynamically from interface.app."""
    paths = []
    for path in INTERFACE.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _uses_app_globals_shim(tree):
            paths.append(path)
    return sorted(paths)


class InterfaceImportTests(unittest.TestCase):
    def test_every_interface_module_imports_with_actionable_errors(self) -> None:
        failures: list[str] = []
        for name in _interface_modules():
            try:
                importlib.import_module(name)
            except Exception as exc:  # Deliberately do not hide dependency/name errors.
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        self.assertFalse(failures, "Interface import failures:\n" + "\n".join(failures))

    def test_dynamic_mixins_have_all_shared_names(self) -> None:
        app = importlib.import_module("interface.app")
        missing: dict[str, list[str]] = {}
        for path in _shim_module_paths():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            assigned = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
            }
            imported = {
                alias.asname or alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            params: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    a = node.args
                    for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                        params.add(arg.arg)
                    if a.vararg:
                        params.add(a.vararg.arg)
                    if a.kwarg:
                        params.add(a.kwarg.arg)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    params.add(node.name)
                if isinstance(node, ast.ExceptHandler) and node.name:
                    params.add(node.name)
            loaded = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            names = loaded - assigned - imported - params - set(dir(builtins))
            absent = sorted(name for name in names if not hasattr(app, name))
            if absent:
                missing[path.name] = absent
        self.assertFalse(missing, f"Missing app shared names: {missing}")

    def test_interface_undefined_names_with_ruff(self) -> None:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "interface", "--select", "F821",
                 "--isolated", "--output-format", "concise"],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
        except OSError:
            self.skipTest("Ruff is not installed; mandatory AST import/namespace checks still run")
        if "No module named ruff" in result.stderr:
            self.skipTest("Ruff is not installed; mandatory AST import/namespace checks still run")
        if result.returncode not in (0, 1):
            self.fail(result.stderr or result.stdout)
        findings = []
        app = importlib.import_module("interface.app")
        known_dynamic_names = {"StartupValidationSplash"}
        shim_filenames = {path.name for path in _shim_module_paths()}
        for line in (result.stdout + result.stderr).splitlines():
            if ": F821 " not in line:
                continue
            if "startup_validation.py" in line and "StartupValidationSplash" in line:
                continue
            path, _, detail = line.partition(": F821 ")
            symbol = detail.split("`", 2)[1] if "`" in detail else detail
            file_part = path.rsplit(":", 2)[0]
            filename = Path(file_part.replace("\\", "/")).name
            # Mixin modules intentionally receive their globals from app.py.
            if filename in shim_filenames and hasattr(app, symbol):
                continue
            if filename == "startup_validation.py" and (
                symbol in known_dynamic_names or "StartupValidationSplash" in line
            ):
                continue
            findings.append(line)
        self.assertFalse(findings, "Ruff F821 findings:\n" + "\n".join(findings))

    def test_main_window_constructs_offscreen(self) -> None:
        app = QApplication.instance() or QApplication([])
        module = importlib.import_module("interface.app")
        window = module.MainWindow()
        try:
            self.assertTrue(window.isWidgetType())
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
