#!/usr/bin/env python3

from __future__ import annotations

import importlib
import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from interface.startup_splash import StartupSplash


def _load_app(result: dict[str, object]) -> None:
    """Load the application module on the Qt thread for startup tests."""
    try:
        result["module"] = importlib.import_module("interface.app")
    except Exception as exc:
        result["error"] = exc


def _show_startup_error(exc: Exception) -> None:
    """Log a startup failure and show its details before exiting."""
    details = traceback.format_exc()
    print(details, file=sys.stderr, end="")
    try:
        dialog = QMessageBox()
        dialog.setIcon(QMessageBox.Critical)
        dialog.setWindowTitle("DrunkenBot-IDE could not start")
        dialog.setText(f"DrunkenBot-IDE could not start: {exc}")
        dialog.setDetailedText(details)
        dialog.exec()
    except Exception as dialog_exc:
        print(f"Unable to display startup error dialog: {dialog_exc}", file=sys.stderr)


def main() -> None:
    app = QApplication(sys.argv)
    splash = StartupSplash()
    splash.show()
    app.processEvents()
    splash.append_log("Preparing application components...")
    try:
        result: dict[str, object] = {}
        _load_app(result)
        if "error" in result:
            raise result["error"]  # type: ignore[misc]
        module = result["module"]
    except Exception as exc:
        try:
            splash.append_log(f"Unable to load application: {exc}")
        except Exception:
            pass
        _show_startup_error(exc)
        return
    try:
        module.main(app=app, splash=splash)  # type: ignore[union-attr]
    except Exception as exc:
        try:
            splash.append_log(f"Unable to start application: {exc}")
        except Exception:
            pass
        _show_startup_error(exc)
        return
    if app.property("startup_aborted"):
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
