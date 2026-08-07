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
        for line in result.stdout.splitlines():
            if ": F821 " not in line:
                continue
            path, _, detail = line.partition(": F821 ")
            symbol = detail.split("`", 2)[1] if "`" in detail else detail
            filename = Path(path.replace("\\", "/")).name
            # Mixin modules intentionally receive their globals from app.py.
            if filename.startswith("main_window_part") and hasattr(app, symbol):
                continue
            if filename == "startup_validation.py" and symbol in known_dynamic_names:
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
