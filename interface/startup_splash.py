from __future__ import annotations

import html
import os
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QHBoxLayout, QProgressBar, QTextBrowser, QVBoxLayout

from interface.theme import DARK_THEME, apply_theme, current_theme, load_startup_theme


class StartupSplash(QDialog):
    """Splash screen shown while startup validation is running."""

    def __init__(self) -> None:
        super().__init__()
        apply_theme(load_startup_theme())
        self.setWindowTitle("DrunkenBot-IDE")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setModal(True)
        self.setMinimumSize(560, 760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        header = QHBoxLayout()
        logo = QLabel()
        logo.setFixedSize(128, 128)
        logo_candidates = [
            Path(__file__).resolve().parent / "drunken_bot_logo_small.png",
        ]
        if hasattr(sys, "_MEIPASS"):
            logo_candidates.insert(0, Path(sys._MEIPASS) / "drunken_bot_logo_small.png")
        app_root = os.environ.get("DRUNKENBOT_APP_ROOT")
        if app_root:
            logo_candidates.insert(0, Path(app_root) / "interface" / "drunken_bot_logo_small.png")
        pixmap = QPixmap(str(next((path for path in logo_candidates if path.exists()), logo_candidates[0])))
        if pixmap.isNull():
            logo.setText("DB")
            logo.setAlignment(Qt.AlignCenter)
            logo.setObjectName("Logo")
        else:
            logo.setPixmap(pixmap.scaled(118, 118, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo.setAlignment(Qt.AlignCenter)
        title = QLabel("DrunkenBot-IDE")
        title.setObjectName("Title")
        title.setFont(QFont("Arial", 22))
        header.addWidget(logo)
        header.addSpacing(10)
        header.addWidget(title, 1)
        layout.addLayout(header)
        self.status = QLabel("Preparing checks...")
        self.status.setObjectName("Step")
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        layout.addWidget(self.progress)
        self.checks_view = QTextBrowser()
        self.checks_view.setOpenExternalLinks(False)
        self.checks_view.setReadOnly(True)
        layout.addWidget(self.checks_view, 1)
        self.footer_label = QLabel("")
        layout.addWidget(self.footer_label)
        self._checks = {}
        self._check_order = []

    def append_log(self, message: str) -> None:
        self.footer_label.setText(message)

    def set_checks(self, checks: list[str]) -> None:
        self._check_order = list(checks)
        self._checks = {label: "pending" for label in checks}
        self._render_checks()

    def add_check(self, label: str) -> None:
        """Add a dynamically discovered check to the checklist.

        Args:
            label: Human-readable check name.
        """
        if label in self._checks:
            self._checks[label] = "running"
            self._render_checks()
            return
        self._check_order.append(label)
        self._checks[label] = "running"
        self._render_checks()

    def update_step(self, text: str, index: int, total: int) -> None:
        self.status.setText(text)
        self.progress.setValue(int(index / max(total, 1) * 100))

    def mark_check_running(self, label: str) -> None:
        self._checks[label] = "running"
        self._render_checks()

    def mark_check_done(self, label: str) -> None:
        self._checks[label] = "done"
        self._render_checks()

    def mark_check_failed(self, label: str) -> None:
        self._checks[label] = "failed"
        self._render_checks()

    def _render_checks(self) -> None:
        colors = (
            {"done": "#ffffff", "running": "#e2cfaa", "failed": "#ff9a9a", "pending": "#bdbdbd"}
            if current_theme() == DARK_THEME
            else {"done": "#202020", "running": "#7a5100", "failed": "#a82d2d", "pending": "#555555"}
        )
        rows = ["<ul style='margin:0; padding-left:18px; line-height:1.8;'>"]
        for label in self._check_order:
            state = self._checks.get(label, "pending")
            escaped = html.escape(label)
            marker = {"done": "[OK]", "running": "[*]", "failed": "[FAIL]"}.get(state, "-")
            color = colors.get(state, colors["pending"])
            rows.append(f"<li style='color:{color};'>{marker} {escaped}</li>")
        rows.append("</ul>")
        self.checks_view.setHtml("".join(rows))

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
