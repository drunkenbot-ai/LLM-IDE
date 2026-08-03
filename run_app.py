#!/usr/bin/env python3

from __future__ import annotations

import importlib
import sys

from PySide6.QtWidgets import QApplication

from interface.startup_splash import StartupSplash


def _load_app(result: dict[str, object]) -> None:
    """Load the application module on the Qt thread for startup tests."""
    try:
        result["module"] = importlib.import_module("interface.app")
    except Exception as exc:
        result["error"] = exc


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
        splash.append_log(f"Unable to load application: {exc}")
        raise
    module.main(app=app, splash=splash)  # type: ignore[union-attr]
    if app.property("startup_aborted"):
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
