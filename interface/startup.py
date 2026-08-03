from __future__ import annotations

"""Startup validation and project selection UI."""
import ctypes
from datetime import datetime
import html
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QProgressBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .startup_validation import (
    _run_startup_tests,
    _run_startup_validations,
    _validate_writable_directory,
)


APP_NAME = "DrunkenBot LLM-IDE"
WINDOWS_APP_ID = "DrunkenBot.LLMIDE"
LOGGER = logging.getLogger(__name__)
APP_HOME_DIR = Path.home() / ".drunkenbot_ide"
DEFAULT_CACHE_DIR = APP_HOME_DIR / "cache"
DEFAULT_PROJECTS_DIR = APP_HOME_DIR / "projects"
RECENT_PROJECTS_PATH = APP_HOME_DIR / "recent_projects.json"
_WINDOWS_ICON_HANDLES: list[int] = []
_LOGO_FONT_FAMILY: Optional[str] = None


def _main_window():
    """Load the main window lazily to avoid a startup-module import cycle."""
    from .app import MainWindow

    return MainWindow

def _load_recent_projects(limit: int = 12) -> list[Path]:
    """Return recently opened project files that still exist."""

    try:
        payload = json.loads(RECENT_PROJECTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    results: list[Path] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        path_text = str(item.get("path", "")).strip()
        if not path_text:
            continue
        path = Path(path_text)
        if path.exists() and path.is_file():
            results.append(path)
        if len(results) >= limit:
            break
    return results


def _register_recent_project(project_file: Path, limit: int = 12) -> None:
    """Insert/update a project file in recent history."""

    APP_HOME_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().isoformat() + "Z"
    try:
        payload = json.loads(RECENT_PROJECTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = []
    rows: list[dict[str, str]] = []
    resolved_new = project_file.resolve()
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        path_text = str(item.get("path", "")).strip()
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists() or not path.is_file():
            continue
        if path.resolve() == resolved_new:
            continue
        rows.append(
            {
                "path": str(path),
                "last_opened": str(item.get("last_opened", now)),
            }
        )
    rows.insert(0, {"path": str(project_file), "last_opened": now})
    RECENT_PROJECTS_PATH.write_text(json.dumps(rows[:limit], indent=2), encoding="utf-8")


def _apply_windows_taskbar_icon(widget: QWidget) -> None:
    """Apply the app icon to a Qt widget taskbar entry on Windows."""

    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception:
        LOGGER.exception("Could not set Windows app user model ID for widget")
    icon_path = _main_window()._ensure_windows_icon_file()
    if icon_path is None:
        return
    hwnd = int(widget.winId())
    if not hwnd:
        return
    wm_seticon = 0x0080
    icon_small = 0
    icon_big = 1
    image_icon = 1
    lr_loadfromfile = 0x0010
    user32 = ctypes.windll.user32
    hicon_big = user32.LoadImageW(None, str(icon_path), image_icon, 256, 256, lr_loadfromfile)
    hicon_small = user32.LoadImageW(None, str(icon_path), image_icon, 32, 32, lr_loadfromfile)
    if hicon_big:
        user32.SendMessageW(hwnd, wm_seticon, icon_big, hicon_big)
        _WINDOWS_ICON_HANDLES.append(hicon_big)
    if hicon_small:
        user32.SendMessageW(hwnd, wm_seticon, icon_small, hicon_small)
        _WINDOWS_ICON_HANDLES.append(hicon_small)


def _logo_font_family() -> Optional[str]:
    """Load and cache the custom logo font family when available."""

    global _LOGO_FONT_FAMILY
    if _LOGO_FONT_FAMILY is not None:
        return _LOGO_FONT_FAMILY
    font_path = Path(__file__).resolve().parents[1] / "fonts" / "Blue-Whale Heavy.otf"
    if not font_path.exists():
        _LOGO_FONT_FAMILY = ""
        return None
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    if font_id < 0:
        _LOGO_FONT_FAMILY = ""
        return None
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        _LOGO_FONT_FAMILY = ""
        return None
    _LOGO_FONT_FAMILY = families[0]
    return _LOGO_FONT_FAMILY


class StartupValidationSplash(QDialog):
    """Modal splash screen that shows startup validation progress."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setModal(True)
        self.setMinimumSize(560, 760)
        self.setFont(QFont("Arial", 10))
        self._checks: dict[str, str] = {}
        self._check_order: list[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QDialog { background: #111111; color: #d0d0d0; border: 0; border-radius: 0; font-family: Arial, "Segoe UI", sans-serif; }
            QLabel#Title { color: #d0d0d0; font-size: 22px; }
            QLabel#Subtitle { color: #bfbfbf; font-size: 13px; }
            QLabel#Step { color: #c7c7c7; font-size: 13px; }
            QTextBrowser { background: #111111; color: #d0d0d0; border: 0; padding: 10px; }
            QProgressBar { background: #222222; border: 0; border-radius: 2px; }
            QProgressBar::chunk { background: #bcbcbc; border-radius: 2px; }
            """
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        header = QHBoxLayout()
        logo = QLabel()
        logo.setFixedSize(128, 128)
        logo_pixmap = _main_window()._app_logo_pixmap(118)
        if logo_pixmap.isNull():
            logo.setText("DB")
            logo.setAlignment(Qt.AlignCenter)
            logo.setStyleSheet("color:#f5b041;font-size:38px;")
        else:
            logo.setPixmap(logo_pixmap)
            logo.setAlignment(Qt.AlignCenter)
        title_box = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("Title")
        logo_family = _logo_font_family()
        if logo_family:
            title.setFont(QFont(logo_family, 22))
        title_box.addWidget(title)
        title_box.addSpacing(4)
        header.addWidget(logo)
        header.addSpacing(10)
        header.addLayout(title_box, 1)
        root.addLayout(header)

        self.step_label = QLabel("Preparing checks...")
        self.step_label.setObjectName("Step")
        root.addWidget(self.step_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.checks_view = QTextBrowser()
        self.checks_view.setOpenExternalLinks(False)
        self.checks_view.setReadOnly(True)
        root.addWidget(self.checks_view, 1)
        self.footer_label = QLabel("")
        self.footer_label.setObjectName("Subtitle")
        root.addWidget(self.footer_label)

    def update_step(self, text: str, index: int, total: int) -> None:
        self.step_label.setText(text)
        percent = int((max(0, index) / max(1, total)) * 100)
        self.progress.setValue(percent)
        QApplication.processEvents()

    def set_checks(self, checks: list[str]) -> None:
        """Initialize the checklist in pending state."""

        self._check_order = list(checks)
        self._checks = {label: "pending" for label in checks}
        self._render_checks()

    def add_check(self, label: str) -> None:
        """Add a dynamically discovered check to the startup checklist.

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

    def mark_check_running(self, label: str) -> None:
        self._checks[label] = "running"
        self._render_checks()

    def mark_check_done(self, label: str) -> None:
        self._checks[label] = "done"
        self._render_checks()

    def mark_check_failed(self, label: str) -> None:
        self._checks[label] = "failed"
        self._render_checks()

    def append_log(self, text: str) -> None:
        self.footer_label.setText(text)
        QApplication.processEvents()

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        _apply_windows_taskbar_icon(self)

    def _render_checks(self) -> None:
        rows: list[str] = ["<ul style='margin:0; padding-left:18px; line-height:1.8;'>"]
        for label in self._check_order:
            state = self._checks.get(label, "pending")
            escaped = html.escape(label)
            if state == "done":
                rows.append(f"<li style='color:#ffffff;'>[OK] {escaped}</li>")
            elif state == "running":
                rows.append(f"<li style='color:#e2cfaa;'>[*] {escaped}</li>")
            elif state == "failed":
                rows.append(f"<li style='color:#ff9a9a;'>[FAIL] {escaped}</li>")
            else:
                rows.append(f"<li style='color:#bdbdbd;'>- {escaped}</li>")
        rows.append("</ul>")
        self.checks_view.setHtml("".join(rows))
        QApplication.processEvents()


class ProjectChoiceDialog(QDialog):
    """Prompt shown after startup checks to choose project creation/open flow."""

    def __init__(self) -> None:
        super().__init__()
        self.choice = ""
        self.selected_project_file: Optional[Path] = None
        self.setWindowTitle(APP_NAME)
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.setMinimumSize(760, 520)
        self.setFont(QFont("Arial", 10))
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QDialog { background: #111111; color: #eeeeee; border: 0; border-radius: 0; font-family: Arial, "Segoe UI", sans-serif; }
            QLabel#Title { color: #f5b041; font-size: 24px; }
            QLabel#Body { color: #dddddd; font-size: 13px; }
            QLabel#CardTitle { color: #f1f1f1; font-size: 16px; }
            QLabel#CardBody { color: #c9c9c9; font-size: 12px; }
            QWidget#ChoiceCard { background: #171717; border: 1px solid #3a3a3a; border-radius: 8px; }
            QListWidget { background: #171717; color: #d8d8d8; border: 1px solid #3a3a3a; border-radius: 8px; padding: 4px; }
            QListWidget::item { padding: 6px 8px; }
            QListWidget::item:selected { background: #2a2a2a; color: #ffffff; }
            QPushButton { background: #242424; color: #eeeeee; border: 0; border-radius: 6px; padding: 8px 12px; }
            QPushButton:hover { background: #f5b041; color: #151515; }
            """
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        logo = QLabel()
        logo_pixmap = _main_window()._app_logo_pixmap(144)
        if logo_pixmap.isNull():
            logo.setText("DB")
            logo.setStyleSheet("color:#f5b041;font-size:56px;")
            logo.setAlignment(Qt.AlignCenter)
        else:
            logo.setPixmap(logo_pixmap)
            logo.setAlignment(Qt.AlignCenter)
        root.addWidget(logo, 0, Qt.AlignHCenter)

        title = QLabel("Get started")
        title.setObjectName("Title")
        logo_family = _logo_font_family()
        if logo_family:
            title.setFont(QFont(logo_family, 26))
        title.setAlignment(Qt.AlignLeft)
        root.addWidget(title)

        body = QLabel(
            "Startup checks are complete.\n"
            "Choose how you want to begin with DrunkenBot LLM-IDE."
        )
        body.setObjectName("Body")
        body.setAlignment(Qt.AlignLeft)
        root.addWidget(body)

        new_card = QWidget()
        new_card.setObjectName("ChoiceCard")
        new_layout = QVBoxLayout(new_card)
        new_layout.setContentsMargins(16, 14, 16, 14)
        new_layout.setSpacing(8)
        new_title = QLabel("Create a new project")
        new_title.setObjectName("CardTitle")
        new_body = QLabel("Start with a clean workspace, default folders, and bundled starter data.")
        new_body.setObjectName("CardBody")
        new_body.setWordWrap(True)
        new_button = QPushButton("Create New Project")
        new_layout.addWidget(new_title)
        new_layout.addWidget(new_body)
        new_layout.addWidget(new_button, 0, Qt.AlignLeft)
        root.addWidget(new_card)

        open_card = QWidget()
        open_card.setObjectName("ChoiceCard")
        open_layout = QVBoxLayout(open_card)
        open_layout.setContentsMargins(16, 14, 16, 14)
        open_layout.setSpacing(8)
        open_title = QLabel("Open an existing project")
        open_title.setObjectName("CardTitle")
        open_body = QLabel("Open a saved project.json and continue where you left off.")
        open_body.setObjectName("CardBody")
        open_body.setWordWrap(True)
        open_button = QPushButton("Open Existing Project")
        open_layout.addWidget(open_title)
        open_layout.addWidget(open_body)
        open_layout.addWidget(open_button, 0, Qt.AlignLeft)
        root.addWidget(open_card)

        test_chat_card = QWidget()
        test_chat_card.setObjectName("ChoiceCard")
        test_chat_layout = QVBoxLayout(test_chat_card)
        test_chat_layout.setContentsMargins(16, 14, 16, 14)
        test_chat_layout.setSpacing(8)
        test_chat_title = QLabel("Test local LLM")
        test_chat_title.setObjectName("CardTitle")
        test_chat_body = QLabel("Jump directly to the Chat tab to load a local model and start chatting.")
        test_chat_body.setObjectName("CardBody")
        test_chat_body.setWordWrap(True)
        test_chat_button = QPushButton("Test Local LLM")
        test_chat_layout.addWidget(test_chat_title)
        test_chat_layout.addWidget(test_chat_body)
        test_chat_layout.addWidget(test_chat_button, 0, Qt.AlignLeft)
        root.addWidget(test_chat_card)

        recent_paths = _load_recent_projects()
        self.recent_list: Optional[QListWidget] = None
        if recent_paths:
            recent_card = QWidget()
            recent_card.setObjectName("ChoiceCard")
            recent_layout = QVBoxLayout(recent_card)
            recent_layout.setContentsMargins(16, 14, 16, 14)
            recent_layout.setSpacing(8)
            recent_title = QLabel("Recent projects")
            recent_title.setObjectName("CardTitle")
            recent_layout.addWidget(recent_title)
            self.recent_list = QListWidget()
            for path in recent_paths:
                item = QListWidgetItem(str(path))
                item.setData(Qt.UserRole, str(path))
                self.recent_list.addItem(item)
            self.recent_list.setCurrentRow(0)
            recent_layout.addWidget(self.recent_list)
            recent_button = QPushButton("Open Selected Recent Project")
            recent_button.clicked.connect(self._open_selected_recent)
            recent_layout.addWidget(recent_button, 0, Qt.AlignLeft)
            root.addWidget(recent_card)

        row = QHBoxLayout()
        row.addStretch(1)
        exit_button = QPushButton("Exit")
        new_button.clicked.connect(lambda: self._choose("new"))
        open_button.clicked.connect(lambda: self._choose("open"))
        test_chat_button.clicked.connect(lambda: self._choose("test_local_llm"))
        exit_button.clicked.connect(self.reject)
        row.addWidget(exit_button)
        root.addLayout(row)

    def _choose(self, choice: str) -> None:
        self.choice = choice
        self.accept()

    def _open_selected_recent(self) -> None:
        if self.recent_list is None:
            return
        item = self.recent_list.currentItem()
        if item is None:
            return
        raw = item.data(Qt.UserRole)
        if not raw:
            return
        self.selected_project_file = Path(str(raw))
        self._choose("recent")

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        _apply_windows_taskbar_icon(self)
