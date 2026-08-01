#!/usr/bin/env python3

from __future__ import annotations

import importlib
import sys
from threading import Thread
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from llm_trainer.ui.startup_splash import StartupSplash


def _load_app(result: dict[str, Any]) -> None:
    try:
        result["module"] = importlib.import_module("llm_trainer.ui.app")
    except Exception as exc:
        result["error"] = exc


def main() -> None:
    app = QApplication(sys.argv)
    splash = StartupSplash()
    splash.show()
    app.processEvents()
    splash.append_log("Preparing application components...")

    result: dict[str, Any] = {}
    loader = Thread(target=_load_app, args=(result,), daemon=True)
    loader.start()
    poll = QTimer()

    def continue_startup() -> None:
        if loader.is_alive():
            return
        poll.stop()
        if "error" in result:
            splash.append_log(f"Unable to load application: {result['error']}")
            return
        result["module"].main(app=app, splash=splash)

    poll.timeout.connect(continue_startup)
    poll.start(50)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
