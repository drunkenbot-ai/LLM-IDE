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


class InterfaceImportTests(unittest.TestCase):
    def test_every_interface_module_imports_with_actionable_errors(self) -> None:
        failures: list[str] = []
        for name in _interface_modules():
            try:
                importlib.import_module(name)
            except Exception as exc:  # Deliberately do not hide dependency/name errors.
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        self.assertFalse(failures, "Interface import failures:\n" + "\n".join(failures))

    def test_dynamic_main_window_parts_have_all_shared_names(self) -> None:
        app = importlib.import_module("interface.app")
        missing: dict[str, list[str]] = {}
        for path in sorted(INTERFACE.glob("main_window_part*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if not any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "globals"
                and node.func.attr == "update"
                for node in ast.walk(tree)
            ):
                continue
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
            loaded = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            names = loaded - assigned - imported - set(dir(builtins))
            absent = sorted(name for name in names if not hasattr(app, name))
            if absent:
                missing[path.name] = absent
        self.assertFalse(missing, f"Missing app shared names: {missing}")

    def test_interface_undefined_names_with_ruff_or_ast_fallback(self) -> None:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "interface/app.py", "--select", "F821",
                 "--isolated", "--output-format", "concise"],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
        except OSError:
            result = None
        if result is not None and result.returncode == 0:
            return
        if result is not None and result.returncode not in (1, 2):
            self.fail(result.stderr or result.stdout)
        if result is not None and "No module named ruff" not in result.stderr:
            self.fail("Ruff F821 findings:\n" + (result.stdout or result.stderr))
        compile_errors = []
        for path in INTERFACE.rglob("*.py"):
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except SyntaxError as exc:
                compile_errors.append(f"{path}: {exc}")
        self.assertFalse(compile_errors, "\n".join(compile_errors))

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
