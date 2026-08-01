#!/usr/bin/env python3

from __future__ import annotations

import importlib
import sys
from threading import Thread
from typing import Any, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class BootstrapSplash(QWidget):
    """Show feedback while the heavyweight UI module is imported."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DrunkenBot LLM-IDE")
        self.setMinimumSize(420, 180)
        layout = QVBoxLayout(self)
        self.status = QLabel("Starting DrunkenBot LLM-IDE...\nLoading application components.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def append_log(self, message: str) -> None:
        self.status.setText(f"{self.status.text()}\n{message}")


def _load_app_module(result: dict[str, Any]) -> None:
    try:
        result["module"] = importlib.import_module("llm_trainer.ui.app")
    except Exception as exc:
        result["error"] = exc


def main() -> None:
    """Start with a lightweight splash before importing ML/UI dependencies."""

    app = QApplication(sys.argv)
    splash = BootstrapSplash()
    splash.show()
    app.processEvents()

    result: dict[str, Any] = {}
    splash.append_log("Preparing Python and ML dependencies...")
    loader = Thread(target=_load_app_module, args=(result,), daemon=True)
    loader.start()

    poll = QTimer()

    def finish_import() -> None:
        if loader.is_alive():
            return
        poll.stop()
        error: Optional[BaseException] = result.get("error")
        if error is not None:
            splash.append_log(f"Unable to load application: {error}")
            return
        splash.close()
        result["module"].main(app=app)

    poll.timeout.connect(finish_import)
    poll.start(50)
    app.exec()


if __name__ == "__main__":
    main()
