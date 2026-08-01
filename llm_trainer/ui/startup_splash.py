from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout


class StartupSplash(QDialog):
    """Lightweight splash that can be shown before the full UI is imported."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DrunkenBot LLM-IDE")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setMinimumSize(560, 220)
        layout = QVBoxLayout(self)
        self.status = QLabel("Starting DrunkenBot LLM-IDE...\nLoading application components.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

    def append_log(self, message: str) -> None:
        self.status.setText(f"{self.status.text()}\n{message}")

    def set_checks(self, checks: list[str]) -> None:
        self._checks_total = len(checks)

    def update_step(self, text: str, index: int, total: int) -> None:
        self.status.setText(text)
        self.progress.setValue(int(index / max(total, 1) * 100))

    def mark_check_running(self, label: str) -> None:
        self.append_log(f"Running: {label}")

    def mark_check_done(self, label: str) -> None:
        self.append_log(f"Completed: {label}")

    def mark_check_failed(self, label: str) -> None:
        self.append_log(f"Failed: {label}")
