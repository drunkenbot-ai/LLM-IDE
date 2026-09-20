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
    is_dev_mode,
)
from .theme import DARK_THEME, apply_theme, current_theme, load_startup_theme


APP_NAME = "DrunkenBot-IDE"
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
    font_path = Path(__file__).resolve().parent / "fonts" / "Blue-Whale Heavy.otf"
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
            logo.setObjectName("Logo")
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

        self.step_label = QLabel("Starting DrunkenBot-IDE...")
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
        colors = (
            {"done": "#ffffff", "running": "#e2cfaa", "failed": "#ff9a9a", "pending": "#bdbdbd"}
            if current_theme() == DARK_THEME
            else {"done": "#202020", "running": "#7a5100", "failed": "#a82d2d", "pending": "#555555"}
        )
        rows: list[str] = ["<ul style='margin:0; padding-left:18px; line-height:1.8;'>"]
        for label in self._check_order:
            state = self._checks.get(label, "pending")
            escaped = html.escape(label)
            if state == "done":
                rows.append(f"<li style='color:{colors['done']};'>[OK] {escaped}</li>")
            elif state == "running":
                rows.append(f"<li style='color:{colors['running']};'>[*] {escaped}</li>")
            elif state == "failed":
                rows.append(f"<li style='color:{colors['failed']};'>[FAIL] {escaped}</li>")
            else:
                rows.append(f"<li style='color:{colors['pending']};'>- {escaped}</li>")
        rows.append("</ul>")
        self.checks_view.setHtml("".join(rows))
        QApplication.processEvents()


CHOICE_DIALOG_STYLESHEET = """
QDialog {
    background-color: #0b0d14;
    color: #e2e8f0;
}
QWidget#ChoiceCard {
    background-color: #121520;
    border: 1px solid #1f2538;
    border-radius: 12px;
}
QWidget#ChoiceCard:hover {
    border: 1px solid #333d59;
    background-color: #151a28;
}
QWidget#RecentCard {
    background-color: #0f121b;
    border: 1px solid #1c2233;
    border-radius: 10px;
}
QLabel#DialogTitle {
    color: #f8fafc;
    font-size: 17px;
    font-weight: 800;
}
QLabel#VersionBadge {
    background-color: #1e2538;
    color: #94a3b8;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 600;
}
QLabel#DialogSubtitle {
    color: #64748b;
    font-size: 11px;
}
QLabel#CardTitle {
    color: #f8fafc;
    font-size: 14px;
    font-weight: 700;
}
QLabel#CardBody {
    color: #8391a8;
    font-size: 11px;
    line-height: 14px;
}
QLabel#BadgeAmber {
    background-color: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 6px;
    font-size: 9px;
    font-weight: 800;
    padding: 2px 6px;
}
QLabel#BadgeCyan {
    background-color: rgba(6, 182, 212, 0.15);
    color: #06b6d4;
    border: 1px solid rgba(6, 182, 212, 0.35);
    border-radius: 6px;
    font-size: 9px;
    font-weight: 800;
    padding: 2px 6px;
}
QLabel#BadgePurple {
    background-color: rgba(139, 92, 246, 0.15);
    color: #a78bfa;
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 6px;
    font-size: 9px;
    font-weight: 800;
    padding: 2px 6px;
}
QLabel#RecentHeader {
    color: #64748b;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QPushButton#PrimaryAmberBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d97706, stop:1 #f59e0b);
    color: #0b0d14;
    font-weight: 700;
    font-size: 11px;
    border: none;
    border-radius: 7px;
    padding: 8px 12px;
}
QPushButton#PrimaryAmberBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f59e0b, stop:1 #fbbf24);
}
QPushButton#SecondaryCyanBtn {
    background-color: #0e1c2a;
    color: #38bdf8;
    border: 1px solid #0284c7;
    font-weight: 700;
    font-size: 11px;
    border-radius: 7px;
    padding: 8px 12px;
}
QPushButton#SecondaryCyanBtn:hover {
    background-color: #132a3e;
    border-color: #38bdf8;
    color: #e0f2fe;
}
QPushButton#SecondaryPurpleBtn {
    background-color: #1a142e;
    color: #c084fc;
    border: 1px solid #7c3aed;
    font-weight: 700;
    font-size: 11px;
    border-radius: 7px;
    padding: 8px 12px;
}
QPushButton#SecondaryPurpleBtn:hover {
    background-color: #251b42;
    border-color: #a855f7;
    color: #f3e8ff;
}
QPushButton#ExitBtn {
    background-color: #141722;
    color: #94a3b8;
    border: 1px solid #232838;
    font-size: 11px;
    font-weight: 600;
    border-radius: 6px;
    padding: 5px 14px;
}
QPushButton#ExitBtn:hover {
    background-color: #24161a;
    border-color: #ef4444;
    color: #f87171;
}
QListWidget#RecentList {
    background-color: #090b10;
    border: 1px solid #1c2233;
    border-radius: 6px;
    color: #94a3b8;
    font-family: Consolas, 'Courier New', monospace;
    font-size: 11px;
    padding: 2px;
}
QListWidget#RecentList::item {
    padding: 4px 8px;
    border-radius: 4px;
    color: #cbd5e1;
}
QListWidget#RecentList::item:hover {
    background-color: #161b28;
    color: #f8fafc;
}
QListWidget#RecentList::item:selected {
    background-color: #1e2840;
    color: #38bdf8;
    font-weight: 600;
}
QPushButton#OpenRecentBtn {
    background-color: #141824;
    color: #cbd5e1;
    border: 1px solid #252e45;
    font-size: 11px;
    font-weight: 600;
    border-radius: 6px;
    padding: 5px 12px;
}
QPushButton#OpenRecentBtn:hover {
    background-color: #1e263a;
    border-color: #38bdf8;
    color: #38bdf8;
}
"""


class ProjectChoiceDialog(QDialog):
    """Sleek, compact launcher dialog matching the IDE dark neon design."""

    def __init__(self) -> None:
        super().__init__()
        self.choice = ""
        self.selected_project_file: Optional[Path] = None
        self.setWindowTitle(APP_NAME)
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.setStyleSheet(CHOICE_DIALOG_STYLESHEET)
        recent_paths = _load_recent_projects()
        target_height = 430 if recent_paths else 320
        self.resize(760, target_height)
        self.setMinimumSize(720, 300)
        self._build_ui(recent_paths)

    def _build_ui(self, recent_paths: list[Path]) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        # =====================================================================
        # Header: Logo + App Name + Version Badge + Subtitle + Exit Button
        # =====================================================================
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(12)

        logo = QLabel()
        logo.setFixedSize(38, 38)
        logo_pixmap = _main_window()._app_logo_pixmap(38)
        if logo_pixmap.isNull():
            logo.setText("DB")
            logo.setAlignment(Qt.AlignCenter)
        else:
            logo.setPixmap(logo_pixmap)
            logo.setAlignment(Qt.AlignCenter)
        header.addWidget(logo)

        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)

        title = QLabel(APP_NAME)
        title.setObjectName("DialogTitle")
        logo_family = _logo_font_family()
        if logo_family:
            title.setFont(QFont(logo_family, 17))
        title_row.addWidget(title)

        badge = QLabel("v1.4")
        badge.setObjectName("VersionBadge")
        title_row.addWidget(badge)
        title_row.addStretch(1)
        title_box.addLayout(title_row)

        subtitle = QLabel("Local LLM Pre-Training, Fine-Tuning & Inference Studio")
        subtitle.setObjectName("DialogSubtitle")
        title_box.addWidget(subtitle)

        header.addLayout(title_box, 1)

        exit_button = QPushButton("Exit")
        exit_button.setObjectName("ExitBtn")
        exit_button.clicked.connect(self.reject)
        header.addWidget(exit_button, 0, Qt.AlignVCenter)

        root.addLayout(header)

        # =====================================================================
        # 3-Column Action Cards (Side-by-Side Horizontal Layout)
        # =====================================================================
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)

        # Card 1: Create New Project
        new_card = QWidget()
        new_card.setObjectName("ChoiceCard")
        new_layout = QVBoxLayout(new_card)
        new_layout.setContentsMargins(14, 14, 14, 14)
        new_layout.setSpacing(8)

        badge_row1 = QHBoxLayout()
        b1 = QLabel("★ NEW PROJECT")
        b1.setObjectName("BadgeAmber")
        badge_row1.addWidget(b1)
        badge_row1.addStretch(1)
        new_layout.addLayout(badge_row1)

        new_title = QLabel("Create Project")
        new_title.setObjectName("CardTitle")
        new_layout.addWidget(new_title)

        new_body = QLabel("Start fresh with default folder structures, bundled datasets, and preset architectures.")
        new_body.setObjectName("CardBody")
        new_body.setWordWrap(True)
        new_layout.addWidget(new_body, 1)

        new_button = QPushButton("+ Create New Project")
        new_button.setObjectName("PrimaryAmberBtn")
        new_button.clicked.connect(lambda: self._choose("new"))
        new_layout.addWidget(new_button)
        cards_layout.addWidget(new_card, 1)

        # Card 2: Open Existing Project
        open_card = QWidget()
        open_card.setObjectName("ChoiceCard")
        open_layout = QVBoxLayout(open_card)
        open_layout.setContentsMargins(14, 14, 14, 14)
        open_layout.setSpacing(8)

        badge_row2 = QHBoxLayout()
        b2 = QLabel("📂 OPEN PROJECT")
        b2.setObjectName("BadgeCyan")
        badge_row2.addWidget(b2)
        badge_row2.addStretch(1)
        open_layout.addLayout(badge_row2)

        open_title = QLabel("Open Project")
        open_title.setObjectName("CardTitle")
        open_layout.addWidget(open_title)

        open_body = QLabel("Open an existing project.json to continue model training, evaluation, or export.")
        open_body.setObjectName("CardBody")
        open_body.setWordWrap(True)
        open_layout.addWidget(open_body, 1)

        open_button = QPushButton("Browse Project...")
        open_button.setObjectName("SecondaryCyanBtn")
        open_button.clicked.connect(lambda: self._choose("open"))
        open_layout.addWidget(open_button)
        cards_layout.addWidget(open_card, 1)

        # Card 3: Test Local LLM
        test_chat_card = QWidget()
        test_chat_card.setObjectName("ChoiceCard")
        test_chat_layout = QVBoxLayout(test_chat_card)
        test_chat_layout.setContentsMargins(14, 14, 14, 14)
        test_chat_layout.setSpacing(8)

        badge_row3 = QHBoxLayout()
        b3 = QLabel("💬 CHAT LAB")
        b3.setObjectName("BadgePurple")
        badge_row3.addWidget(b3)
        badge_row3.addStretch(1)
        test_chat_layout.addLayout(badge_row3)

        test_chat_title = QLabel("Test Local LLM")
        test_chat_title.setObjectName("CardTitle")
        test_chat_layout.addWidget(test_chat_title)

        test_chat_body = QLabel("Jump straight into the inference playground to test local GGUF or PyTorch models.")
        test_chat_body.setObjectName("CardBody")
        test_chat_body.setWordWrap(True)
        test_chat_layout.addWidget(test_chat_body, 1)

        test_chat_button = QPushButton("Test Local LLM")
        test_chat_button.setObjectName("SecondaryPurpleBtn")
        test_chat_button.clicked.connect(lambda: self._choose("test_local_llm"))
        test_chat_layout.addWidget(test_chat_button)
        cards_layout.addWidget(test_chat_card, 1)

        root.addLayout(cards_layout, 1)

        # =====================================================================
        # Bottom Section: Recent Projects
        # =====================================================================
        self.recent_list: Optional[QListWidget] = None
        if recent_paths:
            recent_card = QWidget()
            recent_card.setObjectName("RecentCard")
            recent_layout = QVBoxLayout(recent_card)
            recent_layout.setContentsMargins(12, 10, 12, 10)
            recent_layout.setSpacing(6)

            top_row = QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)
            recent_title = QLabel("RECENT PROJECTS")
            recent_title.setObjectName("RecentHeader")
            top_row.addWidget(recent_title)
            top_row.addStretch(1)

            recent_button = QPushButton("Open Selected")
            recent_button.setObjectName("OpenRecentBtn")
            recent_button.clicked.connect(self._open_selected_recent)
            top_row.addWidget(recent_button)
            recent_layout.addLayout(top_row)

            self.recent_list = QListWidget()
            self.recent_list.setObjectName("RecentList")
            self.recent_list.setFixedHeight(64)
            for path in recent_paths:
                item = QListWidgetItem(str(path))
                item.setData(Qt.UserRole, str(path))
                self.recent_list.addItem(item)
            self.recent_list.setCurrentRow(0)
            self.recent_list.itemDoubleClicked.connect(lambda _item: self._open_selected_recent())
            recent_layout.addWidget(self.recent_list)

            root.addWidget(recent_card)

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
