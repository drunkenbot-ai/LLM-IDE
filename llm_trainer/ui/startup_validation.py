"""Startup validation helpers and repository test execution."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from PySide6.QtWidgets import QApplication

from engine.app_logging import DEFAULT_LOG_DIR

APP_HOME_DIR = Path.home() / ".drunkenbot_ide"
DEFAULT_CACHE_DIR = APP_HOME_DIR / "cache"
DEFAULT_PROJECTS_DIR = APP_HOME_DIR / "projects"


def _validate_writable_directory(path: Path) -> None:
    """Ensure a directory exists and can be written."""
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".startup_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _test_display_name(label: str) -> str:
    """Extract the concise unittest method name from verbose output."""
    value = label.removeprefix("Test: ").strip()
    return value.split(" ", 1)[0].removesuffix("...")


def _discover_test_labels(tests_root: Path) -> list[str]:
    """Discover unittest-style test method labels without executing tests."""
    labels: list[str] = []
    for path in sorted(tests_root.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            labels.extend(
                method.name
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name.startswith("test")
            )
    return labels


def _run_startup_tests(repo_root: Path, tests_root: Path, on_test: Optional[Any] = None) -> None:
    """Run repository tests and raise on failure."""
    if not tests_root.exists():
        raise RuntimeError(f"Tests folder not found: {tests_root}")
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v", "-p", "test_*.py"]
    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        clean_line = line.strip()
        if clean_line:
            output_lines.append(clean_line)
            if on_test is not None and clean_line.startswith("test"):
                on_test(f"Test: {clean_line}")
            QApplication.processEvents()
    return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(output_lines[-25:]).strip()
        raise RuntimeError(f"Startup tests failed.\n{tail}")


def _run_startup_validations(splash: StartupValidationSplash) -> None:
    """Run startup checks shown on the splash screen."""
    # ``interface`` is a top-level package after the engine/interface split.
    repo_root = Path(__file__).resolve().parents[2]
    tests_root = repo_root / "tests"
    required_modules = [
        "PySide6", "torch", "PyPDF2", "numpy", "tokenizers",
        "llm_trainer.dataset_build", "llm_trainer.training", "llm_trainer.ui.app",
    ]
    steps: list[tuple[str, Any]] = [
        ("Checking log folder", lambda: _validate_writable_directory(DEFAULT_LOG_DIR)),
        ("Checking cache folder", lambda: _validate_writable_directory(DEFAULT_CACHE_DIR)),
        ("Checking projects folder", lambda: _validate_writable_directory(DEFAULT_PROJECTS_DIR)),
        ("Checking required imports", lambda: [importlib.import_module(name) for name in required_modules]),
    ]
    # Populate the checklist before the subprocess starts.  Test callbacks
    # update these entries while unittest is streaming verbose output.
    splash.set_checks(_discover_test_labels(tests_root) if tests_root.is_dir() else [])
    if tests_root.is_dir():
        steps.append((
            "Running test suite",
            lambda: _run_startup_tests(
                repo_root,
                tests_root,
                lambda label: (
                    splash.add_check(_test_display_name(label)),
                    splash.mark_check_done(_test_display_name(label)),
                ),
            ),
        ))
    else:
        splash.append_log("Repository tests are not included in this packaged installation; skipping test suite.")
    splash.append_log(f"Workspace: {repo_root}")
    for index, (label, action) in enumerate(steps, start=1):
        splash.update_step(f"{label}...", index - 1, len(steps))
        action()
        splash.append_log(f"Completed: {label}")
    splash.update_step("Startup checks complete", len(steps), len(steps))
    splash.append_log("All startup validations passed.")

