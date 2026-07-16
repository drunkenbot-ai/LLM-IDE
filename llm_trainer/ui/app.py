from __future__ import annotations

import ctypes
from datetime import datetime
import html
import importlib
import json
import logging
import math
import os
from queue import Empty, Queue
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread
from typing import Any, Optional, Union

import torch
from PySide6.QtCore import QObject, QEvent, QPoint, Qt, QThread, QTimer, Slot, qInstallMessageHandler
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QSpinBox,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from llm_trainer.app_logging import qt_message_handler, setup_logging
from llm_trainer.app_logging import DEFAULT_LOG_DIR
from llm_trainer.config import DatasetConfig, ModelConfig, TrainingConfig
from llm_trainer.conversation_datasets import CONVERSATION_DATASET_PRESETS, dataset_ids_for_stage, dataset_stage_label
from llm_trainer.contracts import BackendKind
from llm_trainer.contracts.jobs import RuntimeSpec, TrainingJobSpec
from llm_trainer.coordinator import CoordinatorApiServer, JobManager, create_job_artifact_bundle
from llm_trainer.evaluation import DEFAULT_BENCHMARK_PROMPTS, evaluate_checkpoint, normalize_prompts
from llm_trainer.export import export_gguf_with_llama_cpp, export_hf_microgpt_package, export_llama_adapter_package, export_project_bundle, quantize_checkpoint
from llm_trainer.fine_tuning_service import run_fine_tuning_job
from llm_trainer.llama_chat import LlamaChatSession, load_llama_chat_session, stream_chat_reply
from llm_trainer.lineage import read_json
from llm_trainer.microgpt_chat import load_microgpt_chat_session, stream_microgpt_chat_reply
from llm_trainer.notifier import NotificationManager, default_notifier_config_path, ensure_notifier_config
from llm_trainer.runpod_cloud import (
    RunPodClient,
    RunPodConfig,
    create_runpod_worker_bundle,
    default_runpod_config_path,
    ensure_runpod_config,
    load_runpod_config,
    public_url_is_cloud_reachable,
    save_runpod_config,
)
from llm_trainer.dataset_build import build_dataset
from llm_trainer.dataset_preview import check_project_health, scan_dataset_preview
from llm_trainer.telemetry_store import initialize_store, insert_metric, latest_run, rows_until, telemetry_db_path
from llm_trainer.training import check_resume_compatibility, latest_checkpoint
from llm_trainer.training_planning import estimate_training_resources, format_bytes
from llm_trainer.training_service import run_training_job
from llm_trainer.ui.chat_widgets import ChatMessageWidget
from llm_trainer.ui.markdown_renderer import markdown_to_html
from llm_trainer.ui.workers import ProcessTaskWorker, TaskWorker
from llm_trainer.ui.tabs.benchmark_tab import build_benchmark_tab
from llm_trainer.ui.tabs.chat_tab import build_chat_tab
from llm_trainer.ui.tabs.dataset_tab import build_dataset_tab
from llm_trainer.ui.tabs.dataset_plan_tab import (
    DATASET_DOMAIN_DEFAULTS,
    DATASET_DOMAIN_PRESETS,
    build_dataset_plan_tab,
    default_data_root,
    default_data_stage,
    dataset_plan_defaults,
    iter_default_data_files,
)
from llm_trainer.ui.tabs.live_tab import build_live_training_tab
from llm_trainer.ui.tabs.training_tab import build_training_tab
from llm_trainer.ui.tabs.export_tab import build_export_tab
from llm_trainer.ui.tabs.fine_tuning_tab import build_fine_tuning_tab
from llm_trainer.ui.tabs.job_manager_tab import build_job_manager_tab, set_table_rows

try:
    import psutil
except ImportError:
    psutil = None


APP_NAME = "DrunkenBot LLM-IDE"
WINDOWS_APP_ID = "DrunkenBot.LLMIDE"
LOGGER = logging.getLogger(__name__)
APP_HOME_DIR = Path.home() / ".micro_llm_creator"
DEFAULT_CACHE_DIR = APP_HOME_DIR / "cache"
DEFAULT_PROJECTS_DIR = APP_HOME_DIR / "projects"
RECENT_PROJECTS_PATH = APP_HOME_DIR / "recent_projects.json"
_WINDOWS_ICON_HANDLES: list[int] = []
_LOGO_FONT_FAMILY: Optional[str] = None


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
    icon_path = MainWindow._ensure_windows_icon_file()
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
    font_path = Path(__file__).resolve().parents[2] / "fonts" / "Blue-Whale Heavy.otf"
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
        logo_pixmap = MainWindow._app_logo_pixmap(118)
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
                rows.append(f"<li style='color:#ffffff;'>✓ {escaped}</li>")
            elif state == "running":
                rows.append(f"<li style='color:#e2cfaa;'>● {escaped}</li>")
            elif state == "failed":
                rows.append(f"<li style='color:#ff9a9a;'>✗ {escaped}</li>")
            else:
                rows.append(f"<li style='color:#bdbdbd;'>• {escaped}</li>")
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
        logo_pixmap = MainWindow._app_logo_pixmap(144)
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


def _validate_writable_directory(path: Path) -> None:
    """Ensure a directory exists and can be written."""

    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".startup_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _run_startup_validations(splash: StartupValidationSplash) -> None:
    """Run startup checks shown on the splash screen."""

    repo_root = Path(__file__).resolve().parents[2]
    tests_root = repo_root / "tests"
    required_modules = [
        "PySide6",
        "torch",
        "PyPDF2",
        "numpy",
        "tokenizers",
        "llm_trainer.dataset_build",
        "llm_trainer.training",
        "llm_trainer.ui.app",
    ]

    steps: list[tuple[str, Any]] = [
        ("Checking log folder", lambda: _validate_writable_directory(DEFAULT_LOG_DIR)),
        ("Checking cache folder", lambda: _validate_writable_directory(DEFAULT_CACHE_DIR)),
        ("Checking projects folder", lambda: _validate_writable_directory(DEFAULT_PROJECTS_DIR)),
        (
            "Checking required imports",
            lambda: [importlib.import_module(module_name) for module_name in required_modules],
        ),
        ("Running test suite", lambda: _run_startup_tests(repo_root, tests_root)),
    ]

    splash.set_checks([label for label, _ in steps])
    splash.append_log(f"Workspace: {repo_root}")
    for index, (label, action) in enumerate(steps, start=1):
        splash.update_step(f"[{index}/{len(steps)}] {label}...", index - 1, len(steps))
        splash.mark_check_running(label)
        try:
            action()
        except Exception:
            splash.mark_check_failed(label)
            raise
        splash.mark_check_done(label)
        splash.append_log(f"Completed: {label}")
    splash.update_step("Startup checks complete", len(steps), len(steps))
    splash.append_log("All startup validations passed.")


def _run_startup_tests(repo_root: Path, tests_root: Path) -> None:
    """Run repository tests and raise on failure."""

    if not tests_root.exists():
        raise RuntimeError(f"Tests folder not found: {tests_root}")
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
    ]
    result = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        output = (result.stdout + "\n" + result.stderr).strip()
        tail = "\n".join(output.splitlines()[-25:])
        raise RuntimeError(f"Startup tests failed.\n{tail}")


class MainWindow(QMainWindow):
    """Main PySide6 window for DrunkenBot LLM-IDE."""

    def __init__(self) -> None:
        """Create the main application window."""

        super().__init__()
        self.log_file_path = setup_logging()
        LOGGER.info("Creating %s main window", APP_NAME)
        if QApplication.instance():
            QApplication.instance().setFont(QFont("Arial", 10))
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(self._app_icon())
        self._windows_icon_handles: list[int] = []
        self.resize(1240, 820)
        self.thread: Optional[QThread] = None
        self.worker: Optional[TaskWorker] = None
        self.stop_event: Optional[Event] = None
        self.progress_queue: Optional[Queue] = None
        self.active_log: Optional[QTextEdit] = None
        self.active_progress_bar: Optional[QProgressBar] = None
        self.active_button: Optional[QPushButton] = None
        self.active_stop_button: Optional[QPushButton] = None
        self.active_button_text = ""
        self.active_button_restore_text = ""
        self.active_task_kind = ""
        self.notification_manager: Optional[NotificationManager] = None
        self.current_project_file: Optional[Path] = None
        self.telemetry_db_path: Optional[Path] = None
        self.telemetry_run_id = ""
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_scrub_active = False
        self.hardware_meter_labels: dict[int, QLabel] = {}
        self.training_cards: list[QWidget] = []
        self.training_controls_grid: Optional[QGridLayout] = None
        self.training_controls_columns = 3
        self.training_health_points: list[tuple[int, Optional[float], Optional[float]]] = []
        self.active_training_log: Optional[QTextEdit] = None
        self.active_training_progress: Optional[QProgressBar] = None
        self.active_training_final_button_text = "Start Training"
        self.active_training_output_dir: Optional[Path] = None
        self.interrupt_count = 0
        self.chat_session: Optional[LlamaChatSession] = None
        self.chat_markdown = ""
        self.chat_stream_prefix = ""
        self.chat_stream_reply = ""
        self.current_assistant_browser: Optional[QTextBrowser] = None
        self.current_assistant_meta: Optional[QLabel] = None
        self.current_assistant_message: Optional[ChatMessageWidget] = None
        self.pending_user_message = ""
        self.spinner_index = 0
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self._tick_spinner)
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._drain_progress_queue)
        self.job_manager = JobManager()
        self.coordinator_server: Optional[CoordinatorApiServer] = None
        self.coordinator_thread: Optional[Thread] = None
        self.job_manager_timer = QTimer(self)
        self.job_manager_timer.setInterval(2500)
        self.job_manager_timer.timeout.connect(self.refresh_job_manager_tab)

        self._apply_style()

        shell = self._build_shell()
        self.setCentralWidget(shell)
        self._install_ui_event_logging(shell)
        self._install_wheel_guard(shell)
        self._refresh_notification_manager()
        self.job_manager_timer.start()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Prevent accidental wheel changes on compact option widgets.

        Args:
            watched: Widget receiving the event.
            event: Qt event.

        Returns:
            True when the event is handled by the filter.
        """

        guarded_types = (QSpinBox, QDoubleSpinBox, QComboBox)
        if isinstance(watched, guarded_types):
            if event.type() == QEvent.Type.MouseButtonPress:
                watched.setProperty("_wheel_enabled_after_click", True)
            elif event.type() == QEvent.Type.FocusOut:
                watched.setProperty("_wheel_enabled_after_click", False)
            elif event.type() == QEvent.Type.Wheel and not watched.property("_wheel_enabled_after_click"):
                return True
        return super().eventFilter(watched, event)

    def _install_wheel_guard(self, root: QWidget) -> None:
        """Require a click before spin boxes and combos react to mouse wheel.

        Args:
            root: Root widget to scan for child controls.
        """

        for widget in root.findChildren(QWidget):
            if not isinstance(widget, (QSpinBox, QDoubleSpinBox, QComboBox)):
                continue
            widget.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            widget.setProperty("_wheel_enabled_after_click", False)
            widget.installEventFilter(self)

    def _install_ui_event_logging(self, root: QWidget) -> None:
        """Log user-facing widget actions and parameter changes.

        Args:
            root: Root widget to scan for child controls.
        """

        for widget in root.findChildren(QWidget):
            if isinstance(widget, QAbstractButton):
                if widget.isCheckable():
                    widget.toggled.connect(
                        lambda checked, item=widget: self._log_ui_event("toggled", item, checked)
                    )
                else:
                    widget.clicked.connect(
                        lambda checked=False, item=widget: self._log_ui_event("clicked", item, checked)
                    )
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(
                    lambda value, item=widget: self._log_ui_event("changed", item, value)
                )
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(
                    lambda value, item=widget: self._log_ui_event("changed", item, value)
                )
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(
                    lambda value, item=widget: self._log_ui_event("changed", item, value)
                )
            elif isinstance(widget, QLineEdit):
                widget.editingFinished.connect(
                    lambda item=widget: self._log_ui_event("edited", item, item.text())
                )

    def _log_ui_event(self, action: str, widget: QWidget, value: Any) -> None:
        """Log a UI action or parameter value.

        Args:
            action: Event label.
            widget: Widget that emitted the event.
            value: Current value.
        """

        if action == "clicked" and isinstance(widget, QAbstractButton) and not widget.isCheckable():
            LOGGER.info("UI clicked: %s", self._widget_log_name(widget))
            return
        LOGGER.info("UI %s: %s = %s", action, self._widget_log_name(widget), value)

    @staticmethod
    def _widget_log_name(widget: QWidget) -> str:
        """Return a useful log label for a widget.

        Args:
            widget: Widget to describe.

        Returns:
            Human-readable widget label.
        """

        if isinstance(widget, QAbstractButton) and widget.text():
            return widget.text().replace("\n", " ")
        if isinstance(widget, QLineEdit) and widget.placeholderText():
            return widget.placeholderText()
        if widget.objectName():
            return widget.objectName()
        return widget.__class__.__name__

    def _apply_style(self) -> None:
        """Load the application stylesheet from the QSS module file."""

        qss_path = Path(__file__).with_name("styles.qss")
        self.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    def _build_shell(self) -> QWidget:
        """Build the top-level dashboard shell.

        Returns:
            Root shell widget.
        """

        shell = QWidget()
        shell.setObjectName("AppShell")
        root = QVBoxLayout(shell)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        top = QWidget()
        top.setObjectName("TopBar")
        self.top_bar = top
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(16, 8, 16, 8)
        top_layout.setSpacing(8)
        logo = QLabel()
        logo.setObjectName("Logo")
        logo_pixmap = self._app_logo_pixmap(36)
        if logo_pixmap.isNull():
            logo.setText("DB")
        else:
            logo.setPixmap(logo_pixmap)
        logo.setFixedSize(42, 42)
        logo.setScaledContents(False)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Project name...")
        self.search_box.setMaximumWidth(260)
        self._tip(self.search_box, f"Project name used when saving or reopening a {APP_NAME} project.")
        self.new_project_button = QPushButton("New Project")
        self.new_project_button.setMaximumWidth(130)
        self.new_project_button.clicked.connect(self.new_project)
        self._tip(self.new_project_button, f"Start a fresh {APP_NAME} project with default paths and settings.")
        self.save_project_button = QPushButton("Save Project")
        self.save_project_button.setMaximumWidth(130)
        self.save_project_button.clicked.connect(self.save_project)
        self._tip(self.save_project_button, "Save all current paths and settings into a project.json file.")
        self.open_project_button = QPushButton("Open Project")
        self.open_project_button.setMaximumWidth(130)
        self.open_project_button.clicked.connect(self.open_project)
        self._tip(self.open_project_button, "Open a saved project.json file and restore the UI settings.")
        self.dataset_status = QLabel("Dataset: not prepared")
        self.train_status = QLabel("Training: idle")
        self.export_status = QLabel("Export: waiting")
        self.chat_status = QLabel("Chat: no model loaded")
        for label in (self.dataset_status, self.train_status, self.export_status, self.chat_status):
            label.setObjectName("TopStatus")
            label.setMinimumWidth(0)
            label.setMaximumWidth(180)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            label.setWordWrap(False)
        self.project_state = QLabel("Ready")
        self.project_state.setObjectName("Metric")
        self.project_state.setMinimumWidth(0)
        self.project_state.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        top_layout.addWidget(logo)
        top_layout.addSpacing(12)
        top_layout.addWidget(self.search_box)
        top_layout.addWidget(self.new_project_button)
        top_layout.addWidget(self.save_project_button)
        top_layout.addWidget(self.open_project_button)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.dataset_status)
        top_layout.addWidget(self.train_status)
        top_layout.addWidget(self.export_status)
        top_layout.addWidget(self.chat_status)
        top_layout.addStretch(1)
        top_layout.addWidget(self.project_state)
        root.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        rail = QWidget()
        rail.setObjectName("SideRail")
        self.side_rail = rail
        rail.setFixedWidth(82)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(12, 18, 12, 18)
        rail_layout.setSpacing(12)
        self.dataset_plan_nav = self._nav_button("PLAN")
        self.dataset_nav = self._nav_button("IN")
        self.training_nav = self._nav_button("✦\nAI")
        self.training_nav.setText("AI")
        self.fine_tune_nav = self._nav_button("FT")
        self.live_nav = self._nav_button("LIVE")
        self.jobs_nav = self._nav_button("JOB")
        self.benchmark_nav = self._nav_button("◷\nBench")
        self.export_nav = self._nav_button("⇧\nX")
        self.chat_nav = self._nav_button("◌\nChat")
        self._tip(self.dataset_plan_nav, "Open Dataset Blueprint: plan the target data mix before ingestion.")
        self._tip(self.dataset_nav, "Open dataset preparation: load text/PDF files and build tokenizer data.")
        self._tip(self.training_nav, "Open model training: configure architecture and optimization settings.")
        self._tip(self.fine_tune_nav, "Open fine-tuning: adapt checkpoints with instruction, conversation, or LoRA settings.")
        self._tip(self.live_nav, "Open the live training tracker with model flow, charts, metrics, and telemetry.")
        self._tip(self.jobs_nav, "Open Job Manager: monitor workers, remote connections, assignments, and job controls.")
        self._tip(self.benchmark_nav, "Open benchmark prompts: test checkpoint quality with repeatable prompts.")
        self._tip(self.export_nav, "Open export tools: bundle or quantize the trained model artifacts.")
        self._tip(self.chat_nav, "Open Chat: load a GGUF or native MicroGPT model once and send prompts.")
        self.dataset_plan_nav.setChecked(True)
        self.dataset_plan_nav.clicked.connect(lambda: self._switch_page(0))
        self.dataset_nav.clicked.connect(lambda: self._switch_page(1))
        self.training_nav.clicked.connect(lambda: self._switch_page(2))
        self.fine_tune_nav.clicked.connect(lambda: self._switch_page(3))
        self.live_nav.clicked.connect(lambda: self._switch_page(4))
        self.jobs_nav.clicked.connect(lambda: self._switch_page(5))
        self.benchmark_nav.clicked.connect(lambda: self._switch_page(6))
        self.export_nav.clicked.connect(lambda: self._switch_page(7))
        self.chat_nav.clicked.connect(lambda: self._switch_page(8))
        rail_layout.addWidget(self.dataset_plan_nav)
        rail_layout.addWidget(self.dataset_nav)
        rail_layout.addWidget(self.training_nav)
        rail_layout.addWidget(self.fine_tune_nav)
        rail_layout.addWidget(self.live_nav)
        rail_layout.addWidget(self.jobs_nav)
        rail_layout.addWidget(self.benchmark_nav)
        rail_layout.addWidget(self.export_nav)
        rail_layout.addWidget(self.chat_nav)
        rail_layout.addStretch(1)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_dataset_plan_tab())
        self.pages.addWidget(self._build_dataset_tab())
        self.pages.addWidget(self._build_training_tab())
        self.pages.addWidget(self._build_fine_tuning_tab())
        self.pages.addWidget(self._build_live_training_tab())
        self.pages.addWidget(self._build_job_manager_tab())
        self.pages.addWidget(self._build_benchmark_tab())
        self.pages.addWidget(self._build_export_tab())
        self.pages.addWidget(self._build_chat_tab())

        body.addWidget(rail)
        body.addWidget(self.pages, 1)
        root.addLayout(body, 1)
        return shell

    def _nav_button(self, text: str) -> QPushButton:
        """Create a left-rail navigation button.

        Args:
            text: Button label.

        Returns:
            Configured navigation button.
        """

        button = QPushButton(text)
        button.setObjectName("NavButton")
        button.setCheckable(True)
        return button

    def _switch_page(self, index: int) -> None:
        """Switch the visible page.

        Args:
            index: Page index in the stacked widget.
        """

        self.pages.setCurrentIndex(index)
        buttons = [
            self.dataset_plan_nav,
            self.dataset_nav,
            self.training_nav,
            self.fine_tune_nav,
            self.live_nav,
            self.jobs_nav,
            self.benchmark_nav,
            self.export_nav,
            self.chat_nav,
        ]
        for button_index, button in enumerate(buttons):
            button.setChecked(button_index == index)
        self._refresh_training_layout()
        if index == 5:
            QTimer.singleShot(20, self.refresh_job_manager_tab)

    def show_chat_only_mode(self) -> None:
        """Collapse the UI to chat-only view for quick local LLM testing."""

        if hasattr(self, "top_bar"):
            self.top_bar.hide()
        if hasattr(self, "side_rail"):
            self.side_rail.hide()
        self._switch_page(8)
        self.setWindowTitle("DrunkenBot - Chat")
        self.resize(980, 760)

    def resizeEvent(self, event: Any) -> None:
        """Refresh responsive layouts when the main window changes size.

        Args:
            event: Qt resize event.
        """

        super().resizeEvent(event)
        self._refresh_training_layout()

    def _refresh_training_layout(self) -> None:
        """Apply responsive card columns on the training page."""

        if not self.training_cards or self.training_controls_grid is None:
            return
        width = self.pages.width() if hasattr(self, "pages") else self.width()
        if width >= 900:
            columns = 2
        else:
            columns = 1
        if columns == self.training_controls_columns:
            return
        self._set_training_card_columns(columns)

    def _set_training_card_columns(self, columns: int) -> None:
        """Reflow the training cards into the requested column count.

        Args:
            columns: Number of columns to use.
        """

        if self.training_controls_grid is None:
            return
        while self.training_controls_grid.count():
            self.training_controls_grid.takeAt(0)
        for index, card in enumerate(self.training_cards):
            row = index // columns
            column = index % columns
            self.training_controls_grid.addWidget(card, row, column)
        for column in range(2):
            self.training_controls_grid.setColumnStretch(column, 1 if column < columns else 0)
        self.training_controls_columns = columns

    def _build_dataset_plan_tab(self) -> QWidget:
        """Build the dataset blueprint page.

        Returns:
            Dataset blueprint page widget.
        """

        return build_dataset_plan_tab(self)

    def _build_dataset_tab(self) -> QWidget:
        """Build the dataset preparation page.

        Returns:
            Dataset page widget.
        """

        return build_dataset_tab(self)

    def _build_training_tab(self) -> QWidget:
        """Build the training configuration page.

        Returns:
            Training page widget.
        """

        return build_training_tab(self)

    def _build_fine_tuning_tab(self) -> QWidget:
        """Build the fine-tuning page.

        Returns:
            Fine-tuning page widget.
        """

        return build_fine_tuning_tab(self)

    def _build_live_training_tab(self) -> QWidget:
        """Build the live training tracker page.

        Returns:
            Live training tracker page widget.
        """

        return build_live_training_tab(self)

    def _build_job_manager_tab(self) -> QWidget:
        """Build the distributed job manager page.

        Returns:
            Job manager page widget.
        """

        return build_job_manager_tab(self)

    def refresh_job_manager_tab(self) -> None:
        """Refresh the job manager dashboard tables."""

        if not hasattr(self, "job_worker_table"):
            return
        workers = self.job_manager.list_workers()
        jobs = self.job_manager.list_jobs()
        heartbeats = self.job_manager.state_store.latest_heartbeats()
        worker_rows = []
        for worker in workers:
            heartbeat = heartbeats.get(worker.worker_id, {})
            metrics = heartbeat.get("metrics") or {}
            active_job = heartbeat.get("active_job_id") or self._active_job_for_worker(worker.worker_id)
            capabilities = worker.capabilities or {}
            cpu_ram_gpu = (
                f"CPU {capabilities.get('cpu_count', '-')}, "
                f"RAM {capabilities.get('system_ram_gb', '-')} GB, "
                f"VRAM {capabilities.get('total_vram_gb', '-')} GB"
            )
            if metrics:
                cpu_ram_gpu = f"{cpu_ram_gpu}, util {metrics.get('gpu_util', metrics.get('gpu_memory_percent', '-'))}"
            worker_rows.append(
                [
                    worker.worker_id,
                    worker.status.value,
                    worker.backend.value,
                    worker.device,
                    worker.last_heartbeat_at or "-",
                    active_job or "-",
                    cpu_ram_gpu,
                    ", ".join(capabilities.get("labels") or []) or "-",
                ]
            )
        set_table_rows(self.job_worker_table, worker_rows)

        job_rows = []
        for managed in jobs:
            job = managed.spec
            metrics = managed.latest_metrics
            stage_label = str(job.metadata.get("training_stage") or job.metadata.get("training_mode") or job.training.training_mode)
            job_rows.append(
                [
                    job.job_id,
                    stage_label,
                    job.status.value,
                    managed.assigned_worker_id or "-",
                    job.runtime.backend.value,
                    self._metric_pair(metrics.epoch if metrics else None, metrics.total_epochs if metrics else None),
                    self._metric_pair(metrics.step if metrics else None, metrics.total_steps if metrics else None),
                    str(job.training.batch_size),
                    str(job.model.config.layer_count),
                    self._metric_float(metrics.train_loss if metrics else None),
                    self._metric_float(metrics.tokens_per_second if metrics else None, suffix=" tok/s"),
                    managed.updated_at,
                ]
            )
        set_table_rows(self.job_table, job_rows)
        active_count = sum(1 for item in jobs if item.spec.status.value in {"assigned", "running", "paused", "stopping"})
        queued_count = sum(1 for item in jobs if item.spec.status.value == "queued")
        self.job_worker_count_label.setText(f"Workers: {len(workers)}")
        self.job_active_count_label.setText(f"Active jobs: {active_count}")
        self.job_queue_count_label.setText(f"Queued jobs: {queued_count}")
        self.job_db_label.setText(f"State DB: {self.job_manager.state_store.db_path}")
        self.job_manager_progress.setValue(100)

    def pause_all_managed_jobs(self) -> None:
        """Pause all managed jobs."""

        count = self.job_manager.pause_all_jobs()
        self.job_manager_log.append(f"Pause requested for {count} job(s).")
        self.refresh_job_manager_tab()

    def resume_all_managed_jobs(self) -> None:
        """Resume all paused managed jobs."""

        count = self.job_manager.resume_all_jobs()
        self.job_manager_log.append(f"Resumed {count} job(s).")
        self.refresh_job_manager_tab()

    def stop_all_managed_jobs(self) -> None:
        """Stop all managed jobs."""

        count = self.job_manager.stop_all_jobs()
        self.job_manager_log.append(f"Stop requested for {count} job(s).")
        self.refresh_job_manager_tab()

    def mark_stale_workers_offline(self) -> None:
        """Mark stale remote workers offline."""

        workers = self.job_manager.mark_stale_workers_offline()
        if workers:
            self.job_manager_log.append(f"Marked offline: {', '.join(workers)}")
        else:
            self.job_manager_log.append("No stale remote workers found.")
        self.refresh_job_manager_tab()

    def start_coordinator_server(self) -> None:
        """Start the coordinator API used by remote workers."""

        if self.coordinator_server is not None:
            self.job_manager_log.append("Coordinator API is already running.")
            return
        host = self.coordinator_host.text().strip() or "0.0.0.0"
        port = self.coordinator_port.value()
        artifact_root = Path(self.coordinator_artifact_root.text().strip()).expanduser()
        artifact_root.mkdir(parents=True, exist_ok=True)
        try:
            self.coordinator_server = CoordinatorApiServer(
                manager=self.job_manager,
                host=host,
                port=port,
                artifact_root=artifact_root,
            )
            self.coordinator_thread = Thread(target=self.coordinator_server.serve_forever, daemon=True)
            self.coordinator_thread.start()
        except Exception as exc:
            self.coordinator_server = None
            self.coordinator_thread = None
            QMessageBox.warning(self, "Coordinator failed", f"Could not start coordinator API:\n{exc}")
            return
        public_url = self.coordinator_public_url.text().strip() or f"http://127.0.0.1:{port}"
        self.coordinator_public_url.setText(public_url.rstrip("/"))
        self.coordinator_status_label.setText(f"Coordinator: running at {public_url.rstrip('/')}")
        self.coordinator_start_button.setEnabled(False)
        self.coordinator_stop_button.setEnabled(True)
        self.project_state.setText("Coordinator running")
        self.job_manager_log.append(f"Coordinator API started on {host}:{port}.")
        self.job_manager_log.append(f"Artifact sync root: {artifact_root}")

    def stop_coordinator_server(self) -> None:
        """Stop the coordinator API."""

        if self.coordinator_server is None:
            return
        self.coordinator_server.shutdown()
        if self.coordinator_thread is not None:
            self.coordinator_thread.join(timeout=3)
        self.coordinator_server = None
        self.coordinator_thread = None
        self.coordinator_status_label.setText("Coordinator: stopped")
        self.coordinator_start_button.setEnabled(True)
        self.coordinator_stop_button.setEnabled(False)
        self.project_state.setText("Coordinator stopped")
        self.job_manager_log.append("Coordinator API stopped.")

    def _runpod_config_path(self) -> Path:
        """Return the active RunPod config path.

        Returns:
            Project-local RunPod config path when a project is open.
        """

        project_dir = self.current_project_file.parent if self.current_project_file is not None else None
        return default_runpod_config_path(project_dir)

    def load_runpod_settings(self) -> None:
        """Load RunPod settings into the Job Manager UI."""

        if not hasattr(self, "runpod_api_key"):
            return
        config_path = self._runpod_config_path()
        try:
            config = load_runpod_config(config_path)
        except Exception as exc:
            LOGGER.error("Could not load RunPod config: %s", exc)
            self.runpod_status_label.setText(f"RunPod config error: {exc}")
            return
        self.runpod_api_key.setText(config.api_key)
        self._set_combo_text(self.runpod_gpu_type, config.gpu_type_id)
        self._set_combo_text(self.runpod_cloud_type, config.cloud_type)
        self.runpod_image.setText(config.image_name)
        self.runpod_container_disk.setValue(config.container_disk_gb)
        self.runpod_volume_gb.setValue(config.volume_gb)
        self.runpod_min_ram.setValue(config.min_ram_per_gpu)
        self.runpod_min_vcpu.setValue(config.min_vcpu_per_gpu)
        self.runpod_spot.setChecked(config.interruptible)
        self.runpod_auto_terminate.setChecked(config.auto_terminate)
        status = "configured" if config.api_key.strip() else "API key needed"
        self.runpod_status_label.setText(f"RunPod: {status} ({config_path})")

    def save_runpod_settings(self) -> None:
        """Save RunPod settings from the Job Manager UI."""

        config = self._runpod_config_from_ui()
        config_path = self._runpod_config_path()
        save_runpod_config(config_path, config)
        self.runpod_status_label.setText(f"RunPod settings saved: {config_path}")
        self.job_manager_log.append(f"RunPod settings saved: {config_path}")
        LOGGER.info("RunPod settings saved: %s", config_path)

    def _runpod_config_from_ui(self) -> RunPodConfig:
        """Collect RunPod settings from the UI.

        Returns:
            RunPod configuration.
        """

        return RunPodConfig(
            api_key=self.runpod_api_key.text().strip(),
            image_name=self.runpod_image.text().strip(),
            gpu_type_id=self.runpod_gpu_type.currentText().strip(),
            gpu_count=1,
            cloud_type=self.runpod_cloud_type.currentText().strip(),
            interruptible=self.runpod_spot.isChecked(),
            container_disk_gb=self.runpod_container_disk.value(),
            volume_gb=self.runpod_volume_gb.value(),
            min_vcpu_per_gpu=self.runpod_min_vcpu.value(),
            min_ram_per_gpu=self.runpod_min_ram.value(),
            auto_terminate=self.runpod_auto_terminate.isChecked(),
            worker_labels="runpod,gpu",
        )

    def launch_runpod_worker_for_current_training(self, training_mode: str = "pretrain", stage: str = "base") -> None:
        """Publish the current training job and launch a RunPod worker Pod.

        Args:
            training_mode: Training mode for the queued job.
            stage: Dataset/training stage label.
        """

        if isinstance(training_mode, bool):
            training_mode = "pretrain"
            stage = "base"
        try:
            config = self._runpod_config_from_ui()
            save_runpod_config(self._runpod_config_path(), config)
            coordinator_url = self.coordinator_public_url.text().strip().rstrip("/")
            if not public_url_is_cloud_reachable(coordinator_url):
                raise ValueError(
                    "RunPod needs a public Worker URL. Start a tunnel or set Worker URL to a public address, "
                    "not localhost/127.0.0.1."
                )
            if self.coordinator_server is None:
                self.start_coordinator_server()
                if self.coordinator_server is None:
                    return
            job, bundle_path = self._publish_remote_training_job_spec(
                training_mode=training_mode,
                stage=stage,
                backend_label="runpod",
            )
            artifact_root = Path(self.coordinator_artifact_root.text().strip()).expanduser()
            bootstrap_path = create_runpod_worker_bundle(Path(__file__).resolve().parents[2], artifact_root)
            bootstrap_url = f"{coordinator_url}/artifacts/{bootstrap_path.name}"
            worker_id = f"runpod-{job.job_id}"
            pod_name = f"micro-llm-{self._safe_project_name(self.search_box.text().strip() or 'project')}-{job.job_id[-8:]}"
            result = RunPodClient(config.api_key).create_worker_pod(
                config=config,
                pod_name=pod_name,
                worker_id=worker_id,
                coordinator_url=coordinator_url,
                bootstrap_url=bootstrap_url,
            )
            managed = self.job_manager.get_job(job.job_id)
            managed.spec.metadata["runpod_pod_id"] = result.pod_id
            managed.spec.metadata["runpod_worker_id"] = result.worker_id
            managed.spec.metadata["runpod_cost_per_hour"] = result.cost_per_hour
            self.job_manager._persist_job(job.job_id)
        except Exception as exc:
            LOGGER.exception("RunPod launch failed")
            QMessageBox.warning(self, "RunPod launch failed", str(exc))
            if hasattr(self, "runpod_status_label"):
                self.runpod_status_label.setText(f"RunPod launch failed: {exc}")
            return
        self.runpod_status_label.setText(
            f"RunPod pod {result.pod_id} launched for {job.job_id} ({result.gpu_name}, {result.cost_per_hour}/hr)"
        )
        self.job_manager_log.append(f"RunPod pod launched: {result.pod_id}")
        self.job_manager_log.append(f"RunPod worker: {result.worker_id}")
        self.job_manager_log.append(f"RunPod GPU: {result.gpu_name}, cost/hr: {result.cost_per_hour}")
        self.job_manager_log.append(f"Worker bootstrap: {result.bootstrap_url}")
        self.project_state.setText("RunPod worker launched")
        self.refresh_job_manager_tab()

    def publish_remote_training_job(self, training_mode: str = "pretrain", stage: str = "base") -> None:
        """Bundle the current training setup and queue it for remote workers.

        Args:
            training_mode: Trainer mode to publish, either ``pretrain`` or ``fine_tune``.
            stage: Higher-level stage label for job manager display.
        """

        if isinstance(training_mode, bool):
            training_mode = "pretrain"
            stage = "base"
        if self.coordinator_server is None:
            self.start_coordinator_server()
            if self.coordinator_server is None:
                return
        try:
            job, bundle_path = self._publish_remote_training_job_spec(training_mode=training_mode, stage=stage)
        except Exception as exc:
            QMessageBox.warning(self, "Publish failed", f"Could not publish remote job:\n{exc}")
            return
        self.job_manager_log.append(f"Published remote job: {job.job_id}")
        self.job_manager_log.append(f"Input bundle: {bundle_path}")
        self.job_manager_log.append(f"Worker download URL: {job.metadata.get('artifact_bundle_url')}")
        self.project_state.setText("Remote job queued")
        self.refresh_job_manager_tab()

    def _publish_remote_training_job_spec(
        self,
        training_mode: str = "pretrain",
        stage: str = "base",
        backend_label: str = "remote",
    ) -> tuple[TrainingJobSpec, Path]:
        """Bundle and queue the current remote training job.

        Args:
            training_mode: Trainer mode to publish.
            stage: Higher-level stage label.
            backend_label: Human-readable backend label stored in metadata.

        Returns:
            Queued job and bundle path.
        """

        job = self._current_remote_training_job(training_mode=training_mode, stage=stage)
        job.metadata["launch_backend"] = backend_label
        artifact_root = Path(self.coordinator_artifact_root.text().strip()).expanduser()
        base_url = f"{self.coordinator_public_url.text().strip().rstrip('/')}/artifacts"
        bundle_path = create_job_artifact_bundle(job, artifact_root=artifact_root, base_url=base_url)
        self.job_manager.submit(job)
        return job, bundle_path

    def _current_remote_training_job(self, training_mode: str = "pretrain", stage: str = "base") -> TrainingJobSpec:
        """Build a remote-worker job from current training controls.

        Args:
            training_mode: Trainer mode to publish.
            stage: Higher-level stage label for job manager display.

        Returns:
            Complete training job spec ready to bundle and queue.

        Raises:
            FileNotFoundError: If the prepared dataset is missing.
            ValueError: If model or training options are invalid.
        """

        dataset_dir = Path(self.train_data_dir.text().strip())
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Prepared dataset folder does not exist: {dataset_dir}")
        if not self._dataset_artifacts_exist(dataset_dir):
            raise FileNotFoundError(
                "Prepared dataset is missing tokenizer or token files. "
                "Expected tokenizer.json plus train/val tokens in .npy or .json."
            )
        vocab_size = self._current_training_vocab_size(dataset_dir)
        if vocab_size <= 0:
            raise ValueError("Could not determine tokenizer vocabulary size from the prepared dataset.")
        resume_path = Path(self.resume_checkpoint.text()) if self.resume_checkpoint.text().strip() else None
        if resume_path is None and self.resume_training.isChecked():
            resume_path = latest_checkpoint(self._training_output_dir_for_mode(training_mode) / "checkpoints")
        model_config = self._current_model_config(vocab_size=vocab_size)
        training_config = self._current_training_config(resume_path, training_mode=training_mode)
        model_config.validate()
        training_config.validate()
        job = TrainingJobSpec.local(
            dataset_dir,
            model_config,
            training_config,
            metadata={
                "project_name": self.search_box.text().strip(),
                "submitted_from": "desktop_ui",
                "coordinator_url": self.coordinator_public_url.text().strip().rstrip("/"),
                "training_mode": training_mode,
                "training_stage": stage,
            },
        )
        job.runtime = RuntimeSpec(
            backend=BackendKind.REMOTE_CLIENT,
            device=training_config.device,
            tags=[training_config.device, "remote"],
        )
        return job

    def _active_job_for_worker(self, worker_id: str) -> str:
        """Return the active job ID for a worker.

        Args:
            worker_id: Worker identifier.

        Returns:
            Active job ID or empty string.
        """

        for managed in self.job_manager.list_jobs():
            if managed.assigned_worker_id == worker_id and managed.spec.status.value in {"assigned", "running", "paused", "stopping"}:
                return managed.spec.job_id
        return ""

    @staticmethod
    def _metric_pair(value: Optional[int], total: Optional[int]) -> str:
        """Format a metric pair.

        Args:
            value: Current value.
            total: Total value.

        Returns:
            Display text.
        """

        if value is None:
            return "-"
        if total is None:
            return str(value)
        return f"{value}/{total}"

    @staticmethod
    def _metric_float(value: Optional[float], suffix: str = "") -> str:
        """Format a floating-point metric.

        Args:
            value: Metric value.
            suffix: Optional suffix.

        Returns:
            Display text.
        """

        if value is None:
            return "-"
        return f"{value:.4g}{suffix}"

    def _init_telemetry_store(self, model_dir: Path) -> None:
        """Create or reset the SQLite telemetry store for a training run.

        Args:
            model_dir: Model output directory.
        """

        self.telemetry_db_path = initialize_store(model_dir)
        self.telemetry_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_timeline_label.setText("Timeline: live")
        self.live_scrub_active = False

    def _record_live_metric(self, event: dict[str, Any]) -> None:
        """Persist one live training metric event to SQLite.

        Args:
            event: Training progress event.
        """

        if self.telemetry_db_path is None or not self.telemetry_run_id or event.get("step") is None:
            return
        self.telemetry_latest_id = insert_metric(self.telemetry_db_path, self.telemetry_run_id, event)
        self.telemetry_latest_index += 1
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, self.telemetry_latest_index)
        if not self.live_scrub_active:
            self.live_time_slider.setValue(self.telemetry_latest_index)
            self.live_timeline_label.setText("Timeline: live")
        self.live_time_slider.blockSignals(False)

    def _load_existing_telemetry(self, model_dir: Path) -> None:
        """Load the latest saved telemetry run for an opened project.

        Args:
            model_dir: Model output directory that may contain ``training_telemetry.sqlite``.
        """

        db_path = telemetry_db_path(model_dir)
        self.telemetry_db_path = db_path if db_path.exists() else None
        self.telemetry_run_id = ""
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_scrub_active = False
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_time_slider.blockSignals(False)
        self.live_timeline_label.setText("Timeline: no saved telemetry")
        self.live_sample_text.setText("Training text: -")
        if self.telemetry_db_path is None:
            return
        try:
            run_row = latest_run(self.telemetry_db_path)
            if run_row is None:
                self.live_timeline_label.setText("Timeline: no samples")
                return
            self.telemetry_run_id = str(run_row["run_id"])
            self.telemetry_latest_index = int(run_row["sample_count"] or 0)
            self.telemetry_latest_id = int(run_row["latest_id"] or 0)
        except sqlite3.Error as exc:
            self.live_timeline_label.setText("Timeline: could not load")
            self.training_log.append(f"Telemetry load warning: {exc}")
            return
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, self.telemetry_latest_index)
        self.live_time_slider.setValue(self.telemetry_latest_index)
        self.live_time_slider.blockSignals(False)
        if self.telemetry_latest_index:
            rows = self._timeline_rows_until(self.telemetry_latest_index)
            if rows:
                self._apply_timeline_rows(rows)

    def _timeline_rows_until(self, sample_index: int) -> list[sqlite3.Row]:
        """Load telemetry rows up to a selected sample index.

        Args:
            sample_index: Maximum number of samples to load for the active run.

        Returns:
            Ordered telemetry rows for the active run.
        """

        if self.telemetry_db_path is None or not self.telemetry_run_id or sample_index <= 0:
            return []
        return rows_until(self.telemetry_db_path, self.telemetry_run_id, sample_index)

    def _begin_live_scrub(self) -> None:
        """Pause live auto-follow while the timeline slider is being dragged."""

        self.live_scrub_active = True

    def _end_live_scrub(self) -> None:
        """Apply the selected timeline snapshot after slider drag."""

        self._scrub_live_timeline(self.live_time_slider.value())

    def _jump_live_timeline_to_latest(self) -> None:
        """Return timeline display to the latest live point."""

        self.live_scrub_active = False
        self.live_time_slider.setValue(self.telemetry_latest_index)
        self._scrub_live_timeline(self.telemetry_latest_index)
        self.live_timeline_label.setText("Timeline: live")

    def _scrub_live_timeline(self, sample_index: int) -> None:
        """Replay charts and live visual widgets to a selected telemetry point.

        Args:
            sample_index: Timeline sample selected by the slider.
        """

        rows = self._timeline_rows_until(sample_index)
        if not rows:
            return
        self._apply_timeline_rows(rows)

    def _apply_timeline_rows(self, rows: list[sqlite3.Row]) -> None:
        """Apply historical telemetry rows to charts and live widgets.

        Args:
            rows: Ordered SQLite telemetry rows.
        """

        def series(name: str) -> list[tuple[int, float]]:
            return [(int(row["step"]), float(row[name])) for row in rows if row[name] is not None]

        latest = rows[-1]
        self.loss_chart.set_points(series("train_loss"), series("val_loss"))
        self.optimization_chart.set_points(series("learning_rate"), series("grad_norm"))
        self.stability_chart.set_points(series("weight_norm"), series("update_ratio"))
        self.throughput_chart.set_points(series("tokens_per_second"), series("samples_per_second"))
        self.memory_chart.set_points(series("vram_allocated_gb"), series("vram_reserved_gb"))
        snapshot = {key: latest[key] for key in latest.keys()}
        sample_text = str(snapshot.get("sample_text") or "").strip()
        if sample_text:
            self.live_sample_text.setText(f"Training text: {self._compact_preview_text(sample_text, 220)}")
        else:
            self.live_sample_text.setText("Training text: -")
        self._update_live_training_metrics(
            int(latest["step"]),
            snapshot,
            snapshot.get("train_loss"),
            snapshot.get("learning_rate"),
            snapshot.get("grad_norm"),
            snapshot.get("update_ratio"),
            snapshot.get("tokens_per_second"),
            snapshot.get("samples_per_second"),
            snapshot.get("vram_allocated_gb"),
            snapshot.get("vram_reserved_gb"),
            snapshot.get("gpu_memory_percent"),
            snapshot.get("system_cpu_percent"),
            snapshot.get("system_ram_percent"),
            snapshot.get("data_loader_workers"),
        )
        timestamp = datetime.fromtimestamp(float(latest["recorded_at"])).strftime("%H:%M:%S")
        self.live_timeline_label.setText(f"Timeline: step {int(latest['step']):,} @ {timestamp}")

    @staticmethod
    def _compact_preview_text(text: str, limit: int = 220) -> str:
        """Normalize a training preview into a compact single line.

        Args:
            text: Raw decoded preview text.
            limit: Maximum number of displayed characters.

        Returns:
            Single-line text preview.
        """

        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."

    def _build_export_tab(self) -> QWidget:
        """Build the export page.

        Returns:
            Export page widget.
        """

        return build_export_tab(self)

    def _build_benchmark_tab(self) -> QWidget:
        """Build the benchmark prompt page.

        Returns:
            Benchmark page widget.
        """

        return build_benchmark_tab(self)

    def _build_chat_tab(self) -> QWidget:
        """Build the model test chat page.

        Returns:
            Chat page widget.
        """

        return build_chat_tab(self)

    def _panel(self) -> QWidget:
        """Create a base page panel.

        Returns:
            Panel widget.
        """

        page = QWidget()
        page.setObjectName("Panel")
        return page

    def _page_title(self, text: str) -> QLabel:
        """Create a page title label.

        Args:
            text: Title text.

        Returns:
            Label configured as a page title.
        """

        label = QLabel(text)
        label.setObjectName("PageTitle")
        return label

    def _metric_chip(self, text: str, tooltip: str) -> QLabel:
        """Create a compact metric display label.

        Args:
            text: Initial metric text.
            tooltip: User-facing explanation.

        Returns:
            Configured metric label.
        """

        label = QLabel(text)
        label.setObjectName("MetricChip")
        label.setMinimumWidth(150)
        label.setMinimumHeight(28)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(label, tooltip)
        return label

    def _hardware_meter(self, name: str) -> QProgressBar:
        """Create a slider-like hardware utilization meter.

        Args:
            name: Display name for the meter.

        Returns:
            Configured progress bar.
        """

        meter = QProgressBar()
        meter.setObjectName("HardwareMeter")
        meter.setRange(0, 100)
        meter.setValue(0)
        meter.setTextVisible(False)
        meter.setFixedHeight(8)
        meter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(meter, f"Live {name} utilization.")
        return meter

    def _set_meter(self, meter: QProgressBar, name: str, value: Optional[float]) -> None:
        """Update a hardware utilization meter.

        Args:
            meter: Meter to update.
            name: Display name for the meter.
            value: Utilization percentage.
        """

        if value is None:
            meter.setValue(0)
            label = self.hardware_meter_labels.get(id(meter))
            if label is not None:
                label.setText(f"{name}: -")
            return
        bounded = max(0.0, min(100.0, float(value)))
        meter.setValue(int(round(bounded)))
        label = self.hardware_meter_labels.get(id(meter))
        if label is not None:
            label.setText(f"{name}: {bounded:.1f}%")

    def _update_dataset_quality_report(self, summary: dict[str, Any]) -> None:
        """Update dataset quality chips from a summary dictionary.

        Args:
            summary: Dataset summary fields.
        """

        document_count = int(summary.get("document_count", 0) or 0)
        token_count = int(summary.get("token_count", 0) or 0)
        train_window_count = int(summary.get("train_window_count", 0) or 0)
        val_window_count = int(summary.get("val_window_count", 0) or 0)
        character_count = int(summary.get("character_count", 0) or 0)
        vocab_size = int(summary.get("tokenizer_vocab_size", summary.get("vocab_size", 0)) or 0)
        code_count = int(summary.get("code_sample_count", 0) or 0)
        prose_count = int(summary.get("prose_sample_count", 0) or 0)
        conversation_count = int(summary.get("conversation_sample_count", 0) or 0)
        cached_count = int(summary.get("cached_file_count", 0) or 0)
        processed_count = int(summary.get("processed_file_count", 0) or 0)
        skipped_count = int(summary.get("skipped_file_count", 0) or 0)
        failed_count = int(summary.get("failed_file_count", 0) or 0)
        warning = str(summary.get("warning") or "none")
        sequence_stats = summary.get("sequence_token_stats", {}) or {}
        quality_score = float(summary.get("quality_score", 0.0) or 0.0)
        quality_stars = float(summary.get("quality_stars", 0.0) or 0.0)
        quality_label = str(summary.get("quality_label") or "")
        corpus_block_count = int(summary.get("corpus_block_count", 0) or 0)
        unique_block_count = int(summary.get("unique_block_count", 0) or 0)
        duplicate_block_count = int(summary.get("duplicate_block_count", 0) or 0)
        duplicate_block_ratio = float(summary.get("duplicate_block_ratio", 0.0) or 0.0)
        if not quality_score and (token_count or train_window_count or vocab_size):
            quality_score, quality_stars, quality_label = self._estimate_dataset_rating(
                token_count,
                vocab_size,
                train_window_count,
                val_window_count,
                document_count,
                code_count,
                prose_count,
                conversation_count,
                skipped_count,
                failed_count,
                warning,
                sequence_stats,
            )
        self.dataset_quality_samples.setText(f"Documents: {document_count:,}")
        self.dataset_quality_tokens.setText(f"Tokens: {token_count:,}")
        if train_window_count or val_window_count:
            self.dataset_quality_windows.setText(f"Windows: {train_window_count:,}/{val_window_count:,}")
        else:
            self.dataset_quality_windows.setText("Windows: -")
        self.dataset_quality_vocab.setText(f"Vocab: {vocab_size:,}" if vocab_size else "Vocab: -")
        self.dataset_quality_rating.setText(
            f"Rating: {self._star_text(quality_stars)} {quality_stars:.1f}/5"
            if quality_stars
            else "Rating: -"
        )
        self.dataset_quality_code.setText(f"Code/prose/chat: {code_count:,}/{prose_count:,}/{conversation_count:,}")
        self.dataset_quality_balance.setText("Balance: prepared")
        self.dataset_quality_readiness.setText("Readiness: preview needed")
        self.dataset_quality_cache.setText(f"Files: {processed_count:,} ok, {cached_count:,} cached, {skipped_count:,} skipped, {failed_count:,} failed")
        if corpus_block_count:
            self.dataset_quality_duplicates.setText(f"Duplicates: {duplicate_block_ratio * 100:.1f}%")
            self._tip(
                self.dataset_quality_duplicates,
                (
                    f"{duplicate_block_count:,} repeated blocks out of {corpus_block_count:,}; "
                    f"{unique_block_count:,} unique blocks."
                ),
            )
        else:
            self.dataset_quality_duplicates.setText("Duplicates: -")
        self.dataset_quality_warning.setText(f"Warnings: {warning}")
        self._tip(self.dataset_quality_samples, f"{character_count:,} source characters across prepared documents.")
        if quality_stars:
            self._tip(
                self.dataset_quality_rating,
                f"{quality_label or 'Rated'} dataset: {quality_score:.1f}/100. Higher scores usually mean more usable tokens, richer vocabulary, more windows, and fewer extraction issues.",
            )
        self._tip(
            self.dataset_quality_windows,
            f"{train_window_count:,} training and {val_window_count:,} validation sliding windows.",
        )
        self._update_dataset_stat_charts(summary, code_count, prose_count, conversation_count, sequence_stats)
        if hasattr(self, "dataset_advisor") and (train_window_count or val_window_count):
            advice = [
                "Documents are source items. Windows are the actual context slices used by training.",
                f"This dataset can provide about {train_window_count:,} training windows and {val_window_count:,} validation windows.",
            ]
            if sequence_stats:
                advice.append(
                    "Approx token distribution per source: "
                    f"min {int(sequence_stats.get('min', 0) or 0):,}, "
                    f"avg {float(sequence_stats.get('average', 0.0) or 0.0):,.0f}, "
                    f"median {float(sequence_stats.get('median', 0.0) or 0.0):,.0f}, "
                    f"max {int(sequence_stats.get('max', 0) or 0):,}."
                )
            if document_count < 100 and train_window_count >= 10_000:
                advice.append(
                    "A low document count can still be useful when each document is long, because the trainer samples many overlapping windows."
                )
            if train_window_count < 1_000:
                advice.append("Add more text or lower context length if training looks repetitive.")
            if corpus_block_count:
                advice.append(
                    f"Block diversity: {unique_block_count:,}/{corpus_block_count:,} unique blocks "
                    f"({duplicate_block_ratio * 100:.1f}% repeated)."
                )
            if quality_stars:
                advice.append(f"Dataset rating: {quality_stars:.1f}/5 stars ({quality_label or 'rated'}, score {quality_score:.1f}/100).")
                for reason in list(summary.get("quality_reasons", []) or [])[:4]:
                    advice.append(f"- {reason}")
            self.dataset_advisor.setPlainText("\n".join(advice))

    def _star_text(self, stars: float) -> str:
        """Return a compact five-star display string.

        Args:
            stars: Rating from zero to five.

        Returns:
            Unicode star display with rounded whole stars.
        """

        whole = max(0, min(5, int(round(float(stars)))))
        return "★" * whole + "☆" * (5 - whole)

    def _estimate_dataset_rating(
        self,
        token_count: int,
        vocab_size: int,
        train_window_count: int,
        val_window_count: int,
        document_count: int,
        code_count: int,
        prose_count: int,
        conversation_count: int,
        skipped_count: int,
        failed_count: int,
        warning: str,
        sequence_stats: dict[str, Any],
    ) -> tuple[float, float, str]:
        """Estimate a dataset rating for older summaries that lack saved quality fields.

        Args:
            token_count: Total prepared token count.
            vocab_size: Tokenizer vocabulary size.
            train_window_count: Number of training windows.
            val_window_count: Number of validation windows.
            document_count: Number of source documents.
            code_count: Code sample count.
            prose_count: Prose sample count.
            conversation_count: Conversation/instruction sample count.
            skipped_count: Skipped source file count.
            failed_count: Failed source file count.
            warning: Dataset warning text.
            sequence_stats: Approximate source sequence statistics.

        Returns:
            Score, stars, and label.
        """

        def ratio(value: float, target: float) -> float:
            return max(0.0, min(1.0, float(value) / float(target))) if target > 0 else 0.0

        families = sum(1 for count in (code_count, prose_count, conversation_count) if count > 0)
        score = (
            30.0 * ratio(token_count, 1_000_000)
            + 20.0 * ratio(train_window_count, 50_000)
            + 18.0 * ratio(vocab_size, 8_000)
            + 12.0 * ratio(document_count, 1_000)
            + 8.0 * ratio(val_window_count, 2_000)
            + 7.0 * ratio(families, 3)
            + 5.0 * ratio(float(sequence_stats.get("average", 0.0) or 0.0), 256)
        )
        score -= min(20.0, failed_count * 3.0 + skipped_count * 0.5)
        if warning and warning != "none":
            score -= 5.0
        score = max(0.0, min(100.0, score))
        stars = round(score / 20.0 * 2.0) / 2.0
        if score >= 85:
            label = "Excellent"
        elif score >= 70:
            label = "Good"
        elif score >= 50:
            label = "Usable"
        elif score >= 30:
            label = "Weak"
        else:
            label = "Very weak"
        return score, stars, label

    def _update_dataset_stat_charts(
        self,
        summary: dict[str, Any],
        code_count: int,
        prose_count: int,
        conversation_count: int,
        sequence_stats: dict[str, Any],
    ) -> None:
        """Update dataset statistics charts.

        Args:
            summary: Dataset summary fields.
            code_count: Number of code samples.
            prose_count: Number of prose samples.
            conversation_count: Number of conversation or instruction samples.
            sequence_stats: Approximate token distribution statistics.
        """

        if not hasattr(self, "dataset_mix_chart"):
            return
        mixture_report = summary.get("mixture_report", {}) or {}
        family_rows = list((mixture_report.get("families", {}) or {}).values())
        labels: list[str] = []
        values: list[float] = []
        for row in family_rows:
            actual = float(row.get("actual_percent", 0.0) or 0.0)
            selected = int(row.get("selected_documents", 0) or 0)
            if actual > 0.0 or selected > 0:
                labels.append(str(row.get("label") or "source"))
                values.append(actual)
        if not labels:
            total = max(code_count + prose_count + conversation_count, 1)
            labels = ["Code", "Prose", "Conversation"]
            values = [
                code_count * 100.0 / total,
                prose_count * 100.0 / total,
                conversation_count * 100.0 / total,
            ]
        self.dataset_mix_chart.set_values(labels, values, "%")
        if sequence_stats:
            self.dataset_sequence_chart.set_values(
                ["Min", "Average", "Median", "Max"],
                [
                    float(sequence_stats.get("min", 0) or 0),
                    float(sequence_stats.get("average", 0.0) or 0.0),
                    float(sequence_stats.get("median", 0.0) or 0.0),
                    float(sequence_stats.get("max", 0) or 0),
                ],
            )
        else:
            self.dataset_sequence_chart.clear()

    def _reset_dataset_quality_report(self) -> None:
        """Reset dataset quality chips to their empty state."""

        self.dataset_quality_samples.setText("Documents: -")
        self.dataset_quality_tokens.setText("Tokens: -")
        self.dataset_quality_windows.setText("Windows: -")
        self.dataset_quality_vocab.setText("Vocab: -")
        self.dataset_quality_rating.setText("Rating: -")
        self.dataset_quality_code.setText("Code/prose: -")
        self.dataset_quality_balance.setText("Balance: -")
        self.dataset_quality_readiness.setText("Readiness: -")
        self.dataset_quality_cache.setText("Cache: -")
        self.dataset_quality_duplicates.setText("Duplicates: -")
        self.dataset_quality_extraction.setText("Extraction: -")
        self.dataset_quality_warning.setText("Warnings: none")
        if hasattr(self, "dataset_mix_chart"):
            self.dataset_mix_chart.clear()
        if hasattr(self, "dataset_sequence_chart"):
            self.dataset_sequence_chart.clear()
        if hasattr(self, "dataset_advisor"):
            self.dataset_advisor.setPlainText("Run Preview Dataset to get cleanup suggestions.")

    def _card(self, title: str, content_layout: Union[QVBoxLayout, QFormLayout, QGridLayout, QHBoxLayout]) -> QWidget:
        """Create a neon module card.

        Args:
            title: Card heading.
            content_layout: Layout to place inside the card.

        Returns:
            Card widget.
        """

        card = QWidget()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("SectionLabel")
        layout.addWidget(title_label)
        layout.addLayout(content_layout)
        return card

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        """Create a bounded integer input.

        Args:
            minimum: Minimum value.
            maximum: Maximum value.
            value: Initial value.

        Returns:
            Configured spin box.
        """

        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setMaximumWidth(220)
        return spin

    def _double_spin(self, minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        """Create a bounded float input.

        Args:
            minimum: Minimum value.
            maximum: Maximum value.
            value: Initial value.
            step: Increment step.
            decimals: Number of displayed decimal places.

        Returns:
            Configured double spin box.
        """

        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setMaximumWidth(220)
        return spin

    def _path_row(self, field: QLineEdit, directory: bool = True, file_filter: str = "Checkpoints (*.pt)") -> QWidget:
        """Create a path field with a browse button.

        Args:
            field: Path input widget.
            directory: Whether the browse dialog selects folders.
            file_filter: File dialog filter used when ``directory`` is false.

        Returns:
            Row widget containing the path input and button.
        """

        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        browse = QPushButton("Browse")
        browse.setFixedWidth(88)
        self._tip(browse, "Open a file/folder picker for this path.")
        field.setMinimumWidth(180)
        field.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        browse.clicked.connect(lambda: self._browse(field, directory, file_filter))
        layout.addWidget(field, 1)
        layout.addWidget(browse)
        return row

    def _multi_file_path_row(self, field: QLineEdit, file_filter: str = "All files (*)") -> QWidget:
        """Create a path field with a multi-file browse button.

        Args:
            field: Path input widget. Multiple paths are separated with semicolons.
            file_filter: File dialog filter.

        Returns:
            Row widget containing the path input and browse button.
        """

        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        browse = QPushButton("Browse")
        browse.setFixedWidth(88)
        self._tip(browse, "Choose one or more JSON/JSONL files. You can also paste a folder path.")
        field.setMinimumWidth(180)
        field.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        browse.clicked.connect(lambda: self._browse_multiple_files(field, file_filter))
        layout.addWidget(field, 1)
        layout.addWidget(browse)
        return row

    def _configure_form(self, form: QFormLayout) -> None:
        """Apply common form spacing and growth policy.

        Args:
            form: Form layout to configure.
        """

        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)

    def _configure_device_options(self) -> None:
        """Populate training device choices without duplicate CPU entries."""

        self.device.clear()
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            self.device.addItem("cuda")
            self.device.addItem("cpu")
            self.device_info.setText(f"CUDA ready: {device_name}")
            self.use_amp_default = True
        else:
            self.device.addItem("cpu")
            cuda_build = getattr(torch.backends, "cuda", None)
            built_with_cuda = bool(cuda_build and torch.backends.cuda.is_built())
            if built_with_cuda:
                detail = "CUDA build found, but no usable NVIDIA GPU/driver was detected."
            else:
                detail = "CUDA is not available in this PyTorch install."
            self.device_info.setText(detail)
            self.use_amp_default = False

    def _thin_progress(self) -> QProgressBar:
        """Create a thin bottom progress bar.

        Returns:
            Configured progress bar.
        """

        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setTextVisible(False)
        progress.setFixedHeight(4)
        progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(progress, "Progress indicator for the current page operation.")
        return progress

    def _tip(self, widget: QWidget, text: str) -> None:
        """Attach tooltip and status tip text.

        Args:
            widget: Widget receiving the tip.
            text: Tooltip text.
        """

        widget.setToolTip(text)
        widget.setStatusTip(text)

    def _render_chat_markdown(self, markdown_text: str) -> None:
        """Render chat Markdown with highlighted fenced code blocks when possible.

        Args:
            markdown_text: Markdown transcript to render.
        """

        if not hasattr(self, "current_assistant_message") or self.current_assistant_message is None:
            return
        self.current_assistant_message.set_content(markdown_text)

    def _add_chat_message(
        self,
        role: str,
        content: str,
        metrics: str = "",
        resend_prompt: Optional[str] = None,
    ) -> QTextBrowser:
        """Add one chat bubble.

        Args:
            role: Message role, either ``user`` or ``assistant``.
            content: Markdown message content.
            metrics: Optional metric text shown under assistant replies.
            resend_prompt: Prompt to resend from the bubble.

        Returns:
            Text browser used by the bubble.
        """

        should_follow = self._is_chat_near_bottom()
        max_width = max(320, int(self.chat_scroll.viewport().width() * 0.78)) if hasattr(self, "chat_scroll") else 900
        message = ChatMessageWidget(
            role,
            content,
            markdown_to_html,
            self._resend_chat_message,
            metrics=metrics,
            resend_prompt=resend_prompt,
            max_width=max_width,
        )
        self.chat_messages.insertWidget(max(self.chat_messages.count() - 1, 0), message)
        if should_follow:
            message.scroll_later(lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))
        if role == "assistant":
            self.current_assistant_message = message
            self.current_assistant_browser = message.browser
            self.current_assistant_meta = message.meta_label
        return message.browser

    def _is_chat_near_bottom(self) -> bool:
        """Return whether the chat scroll is close enough to follow streaming.

        Returns:
            True when the view should auto-scroll.
        """

        if not hasattr(self, "chat_scroll"):
            return True
        bar = self.chat_scroll.verticalScrollBar()
        return bar.maximum() - bar.value() < 48

    def _clear_chat_messages(self) -> None:
        """Remove all message bubbles."""

        while self.chat_messages.count() > 1:
            item = self.chat_messages.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.current_assistant_message = None
        self.current_assistant_browser = None
        self.current_assistant_meta = None

    def _resend_chat_message(self, prompt: str) -> None:
        """Resend text from a message bubble.

        Args:
            prompt: Prompt text to send.
        """

        self.chat_input.setPlainText(prompt)
        self.send_chat_message()

    def _set_chat_stats(self, elapsed_seconds: float, token_count: int, tokens_per_second: float) -> None:
        """Update live chat generation metrics.

        Args:
            elapsed_seconds: Elapsed generation time.
            token_count: Generated token count.
            tokens_per_second: Approximate token speed.
        """

        text = f"Time: {elapsed_seconds:.2f}s  |  Tokens: {token_count:,}  |  Speed: {tokens_per_second:.2f} tok/s"
        self.chat_stats.setText(text)
        if self.current_assistant_meta is not None:
            self.current_assistant_meta.setText(text)
            self.current_assistant_meta.setVisible(True)

    def _chat_backend_value(self) -> str:
        """Return the selected chat model backend.

        Returns:
            Stable chat backend identifier.
        """

        if not hasattr(self, "chat_model_backend"):
            return "gguf"
        return "microgpt" if self.chat_model_backend.currentText() == "MicroGPT checkpoint" else "gguf"

    def _update_chat_backend_controls(self) -> None:
        """Show controls relevant to the selected chat backend."""

        if not hasattr(self, "chat_model_backend"):
            return
        native = self._chat_backend_value() == "microgpt"
        self.gguf_path_row.setVisible(not native)
        self.microgpt_path_row.setVisible(native)
        self.llama_gpu_layers.setEnabled(not native)
        self.llama_threads.setEnabled(not native)
        self.llama_context.setEnabled(not native)
        if native:
            self._tip(self.load_llm_button, "Load the native MicroGPT checkpoint into memory once for repeated chat messages.")
        else:
            self._tip(self.load_llm_button, "Load the GGUF model into memory once for repeated chat messages.")

    def _app_icon(self) -> QIcon:
        """Create the application icon.

        Returns:
            Application icon.
        """

        return self._static_app_icon()

    @staticmethod
    def _app_logo_path() -> Path:
        """Return the bundled logo path.

        Returns:
            Logo path.
        """

        return Path(__file__).resolve().parents[2] / "drunken_bot_logo_small.png"

    @staticmethod
    def _app_logo_pixmap(size: int = 64) -> QPixmap:
        """Load the bundled logo as a pixmap.

        Args:
            size: Maximum square size.

        Returns:
            Logo pixmap, or null pixmap when the file is missing.
        """

        pixmap = QPixmap(str(MainWindow._app_logo_path()))
        if pixmap.isNull():
            return pixmap
        return pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    @staticmethod
    def _static_app_icon() -> QIcon:
        """Create the static app icon.

        Returns:
            Application icon.
        """

        logo_path = MainWindow._app_logo_path()
        if logo_path.exists():
            icon = QIcon(str(logo_path))
            if not icon.isNull():
                return icon
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(QColor("#1f1f1f")))
            painter.setPen(QPen(QColor("#f5b041"), 3))
            painter.drawRoundedRect(4, 4, 56, 56, 12, 12)
            bolt = QPolygon([
                QPoint(36, 8),
                QPoint(17, 35),
                QPoint(31, 35),
                QPoint(25, 56),
                QPoint(48, 25),
                QPoint(33, 25),
            ])
            painter.setPen(QPen(QColor("#ffd27a"), 2))
            painter.setBrush(QBrush(QColor("#f5b041")))
            painter.drawPolygon(bolt)
        finally:
            painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _windows_icon_path() -> Path:
        """Return the Windows icon file path.

        Returns:
            Path to the generated ``.ico`` file.
        """

        return Path(__file__).with_name("drunkenbot_llm_ide.ico")

    @staticmethod
    def _ensure_windows_icon_file() -> Optional[Path]:
        """Ensure the generated Windows ``.ico`` file exists.

        Returns:
            Icon path on Windows, otherwise ``None``.
        """

        if sys.platform != "win32":
            return None
        icon_path = MainWindow._windows_icon_path()
        if icon_path.exists():
            return icon_path
        icon = MainWindow._static_app_icon()
        pixmap = icon.pixmap(256, 256)
        if pixmap.isNull() or not pixmap.save(str(icon_path), "ICO"):
            return None
        return icon_path

    def apply_windows_taskbar_icon(self) -> None:
        """Apply the app icon to the native Windows window handle."""

        if sys.platform != "win32":
            return
        icon_path = self._ensure_windows_icon_file()
        if icon_path is None:
            return

        hwnd = int(self.winId())
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
            self._windows_icon_handles.append(hicon_big)
        if hicon_small:
            user32.SendMessageW(hwnd, wm_seticon, icon_small, hicon_small)
            self._windows_icon_handles.append(hicon_small)

    def _browse(self, field: QLineEdit, directory: bool, file_filter: str = "Checkpoints (*.pt)") -> None:
        """Open a file or folder picker for a path field.

        Args:
            field: Path input to update.
            directory: Whether to select a folder instead of a file.
            file_filter: File dialog filter used for files.
        """

        start_dir = self._browse_start_dir(field, directory)
        if directory:
            value = QFileDialog.getExistingDirectory(self, "Choose folder", start_dir)
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Choose file", start_dir, file_filter)
        if value:
            field.setText(value)

    def _browse_multiple_files(self, field: QLineEdit, file_filter: str) -> None:
        """Open a multi-file picker and write selected paths to a field.

        Args:
            field: Path field to update.
            file_filter: File dialog filter.
        """

        values, _ = QFileDialog.getOpenFileNames(self, "Choose files", self._browse_start_dir(field, False), file_filter)
        if values:
            field.setText("; ".join(values))

    def _browse_start_dir(self, field: QLineEdit, directory: bool) -> str:
        """Return the best initial folder for a browse dialog.

        Args:
            field: Path field being browsed.
            directory: Whether the dialog selects a folder.

        Returns:
            Existing field path, active project folder, or current folder.
        """

        text = field.text().strip()
        if text:
            path = Path(text)
            if path.exists():
                if path.is_dir():
                    return str(path)
                return str(path.parent)
            parent = path if directory else path.parent
            if parent.exists():
                return str(parent)
        if self.current_project_file is not None:
            return str(self.current_project_file.parent)
        return str(Path.cwd())

    def save_project(self) -> None:
        """Save the current project settings into a named project folder."""

        project_name = self.search_box.text().strip() or "MicroLLMProject"
        safe_name = self._safe_project_name(project_name)
        if self.current_project_file is None:
            base_dir = QFileDialog.getExistingDirectory(self, "Choose parent folder for project", self._project_dialog_start_dir())
            if not base_dir:
                return
            project_dir = Path(base_dir) / safe_name
            project_file = project_dir / "project.json"
        else:
            project_file = self.current_project_file
            project_dir = project_file.parent
        project_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_project_workspace(project_dir)
        if self.current_project_file is None:
            self._apply_project_workspace_paths(project_dir)
        project_file.write_text(json.dumps(self._project_state_dict(project_name, project_dir), indent=2), encoding="utf-8")
        self.current_project_file = project_file
        _register_recent_project(project_file)
        self._apply_project_runtime_environment(project_dir)
        self._refresh_notification_manager(project_dir)
        if hasattr(self, "runpod_api_key"):
            self.load_runpod_settings()
        self.project_state.setText("Project saved")
        LOGGER.info("Project saved: %s", project_file)
        if self.current_project_file == project_file:
            self.dataset_log.append(f"Project saved: {project_file}")
            self.dataset_log.append(f"Project workspace: {project_dir}")
            self.dataset_log.append(f"Notifier config: {project_dir / 'notifier_config.json'}")

    def new_project(self) -> None:
        """Start a fresh project and clear the active project file binding."""

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please stop or wait for the current task before creating a new project.")
            return
        if self.current_project_file is not None or self.search_box.text().strip():
            choice = QMessageBox.question(
                self,
                "New project",
                "Start a new project? Unsaved changes in the current project will not be saved automatically.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        base_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose folder where the new project will be created",
            self._project_dialog_start_dir(),
        )
        if not base_dir:
            return
        project_name = self.search_box.text().strip() or "MicroLLMProject"
        try:
            self._create_project_at(project_name, Path(base_dir))
        except Exception as exc:
            QMessageBox.warning(self, "New project failed", f"Could not create project:\n{exc}")

    def open_project(self) -> None:
        """Open a saved project file and restore UI settings."""

        project_file, _ = QFileDialog.getOpenFileName(
            self,
            "Open Micro LLM project",
            self._project_dialog_start_dir(),
            "Micro LLM project (project.json *.json);;All files (*)",
        )
        if not project_file:
            return
        try:
            self._open_project_file(Path(project_file))
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", f"Could not open project:\n{exc}")
            return

    def _create_project_at(self, project_name: str, base_dir: Path) -> Path:
        """Create and activate a new project at the selected folder.

        Args:
            project_name: User-facing project name.
            base_dir: Parent folder for the new project.

        Returns:
            Path to the created project.json file.
        """

        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        project_dir = base_dir / self._safe_project_name(project_name)
        project_file = project_dir / "project.json"
        project_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_project_workspace(project_dir)
        copied_count = self._ensure_project_training_data(project_dir)
        self.current_project_file = project_file
        self._apply_project_state(self._default_project_state())
        self.search_box.setText(project_name)
        self._apply_project_workspace_paths(project_dir)
        self._refresh_dataset_blueprint_source(project_dir / "training_data")
        self._apply_project_runtime_environment(project_dir)
        self._refresh_notification_manager(project_dir)
        if hasattr(self, "runpod_api_key"):
            self.load_runpod_settings()
        self._reset_project_runtime_state()
        project_file.write_text(json.dumps(self._project_state_dict(project_name, project_dir), indent=2), encoding="utf-8")
        _register_recent_project(project_file)
        self.project_state.setText("New project")
        LOGGER.info("New project created: %s", project_file)
        self.dataset_log.append(f"Started a new project: {project_file}")
        self.dataset_log.append(f"Project workspace: {project_dir}")
        self.dataset_log.append(f"Default training data copied: {copied_count} file(s)")
        self.dataset_log.append(f"Notifier config: {project_dir / 'notifier_config.json'}")
        return project_file

    def _open_project_file(self, project_file: Path) -> None:
        """Open and activate a project file.

        Args:
            project_file: Path to ``project.json``.
        """

        data = json.loads(project_file.read_text(encoding="utf-8"))
        self.current_project_file = project_file
        _register_recent_project(project_file)
        self._ensure_project_workspace(self.current_project_file.parent)
        dataset_state = data.get("dataset", {}) if isinstance(data, dict) else {}
        saved_default_data_paths = dataset_state.get("default_data_paths")
        self._refresh_dataset_blueprint_source(
            self.current_project_file.parent / "training_data",
            saved_paths=(list(saved_default_data_paths) if saved_default_data_paths is not None else None),
            saved_plan=dict(dataset_state.get("domain_plan", {})),
            preset=str(dataset_state.get("domain_plan_preset", "Balanced Tiny LLM")),
        )
        self._apply_project_state(data)
        self._apply_project_runtime_environment(self.current_project_file.parent)
        self._refresh_notification_manager(self.current_project_file.parent)
        if hasattr(self, "runpod_api_key"):
            self.load_runpod_settings()
        if self.model_dir.text().strip():
            self._load_existing_telemetry(Path(self.model_dir.text()))
        self.project_state.setText("Project opened")
        LOGGER.info("Project opened: %s", project_file)
        self.dataset_log.append(f"Opened project: {project_file}")
        self.dataset_log.append(f"Notifier config: {self.current_project_file.parent / 'notifier_config.json'}")
        self.refresh_model_estimate()

    def _project_dialog_start_dir(self) -> str:
        """Return the best initial folder for project dialogs.

        Returns:
            Active project folder, its parent, or the current folder.
        """

        if self.current_project_file is not None:
            return str(self.current_project_file.parent)
        text = self.dataset_dir.text().strip() if hasattr(self, "dataset_dir") else ""
        if text:
            path = Path(text)
            for candidate in (path, path.parent):
                if candidate.exists():
                    return str(candidate)
        return str(Path.cwd())

    def _ensure_project_workspace(self, project_dir: Path) -> None:
        """Create standard folders inside a project.

        Args:
            project_dir: Project root folder.
        """

        for name in ("datasets", "models", "fine_tunes", "exports", "training_data", "cache", "temp"):
            (project_dir / name).mkdir(parents=True, exist_ok=True)
        ensure_notifier_config(project_dir / "notifier_config.json")
        ensure_runpod_config(project_dir / "runpod_config.json")

    def _ensure_project_training_data(self, project_dir: Path) -> int:
        """Copy bundled default data into the project training-data folder.

        Existing files are left untouched so user edits are not overwritten.

        Args:
            project_dir: Project root folder.

        Returns:
            Number of files copied.
        """

        source_root = default_data_root()
        target_root = project_dir / "training_data"
        target_root.mkdir(parents=True, exist_ok=True)
        if not source_root.exists():
            return 0
        copied = 0
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            try:
                relative = source.relative_to(source_root)
            except ValueError:
                continue
            target = target_root / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
        LOGGER.info("Project default training data copied: project=%s files=%s", project_dir, copied)
        return copied

    def _refresh_dataset_blueprint_source(
        self,
        data_root: Path,
        saved_paths: Optional[list[Any]] = None,
        saved_plan: Optional[dict[str, Any]] = None,
        preset: str = "Balanced Tiny LLM",
    ) -> None:
        """Rebuild the Dataset Blueprint tab from a source data folder.

        Args:
            data_root: Project-local training data folder.
            saved_paths: Optional selected file paths to restore.
            saved_plan: Optional saved domain recipe.
            preset: Saved recipe preset.
        """

        if not hasattr(self, "pages"):
            self.blueprint_data_root = Path(data_root)
            return
        self.blueprint_data_root = Path(data_root)
        current_index = self.pages.currentIndex()
        old_page = self.pages.widget(0)
        new_page = self._build_dataset_plan_tab()
        self.pages.removeWidget(old_page)
        old_page.deleteLater()
        self.pages.insertWidget(0, new_page)
        if saved_plan is not None:
            self._set_dataset_plan(saved_plan, preset)
        if saved_paths is not None:
            self._set_selected_default_data_paths(saved_paths)
        elif self.current_project_file is not None:
            self._set_selected_default_data_paths(None)
        self.pages.setCurrentIndex(current_index)

    def _refresh_notification_manager(self, project_dir: Optional[Path] = None) -> None:
        """Load notification settings for the current project.

        Args:
            project_dir: Optional project root folder.
        """

        if project_dir is None and self.current_project_file is not None:
            project_dir = self.current_project_file.parent
        config_path = default_notifier_config_path(project_dir)
        self.notification_manager = NotificationManager(config_path)
        LOGGER.info("Notifier config active: %s", config_path)

    def _apply_project_workspace_paths(self, project_dir: Path) -> None:
        """Point project output fields at the standard project folders.

        Args:
            project_dir: Project root folder.
        """

        dataset_dir = project_dir / "datasets"
        model_dir = project_dir / "models"
        fine_tune_dir = project_dir / "fine_tunes"
        export_dir = project_dir / "exports"
        training_data_dir = project_dir / "training_data"
        self.dataset_dir.setText(str(dataset_dir))
        self.train_data_dir.setText(str(dataset_dir))
        self.model_dir.setText(str(model_dir))
        self.fine_tune_checkpoint.setText(str(model_dir / "final_model.pt"))
        self.fine_tune_output_dir.setText(str(fine_tune_dir / "latest"))
        self.export_model_dir.setText(str(model_dir))
        self.export_dir.setText(str(export_dir))
        self.gguf_output_path.setText(str(export_dir / "model.gguf"))
        if not self.input_dir.text().strip():
            self.input_dir.setText(str(training_data_dir))

    def _apply_project_runtime_environment(self, project_dir: Path) -> None:
        """Prefer project-local cache/temp folders for runtime work.

        Args:
            project_dir: Project root folder.
        """

        cache_dir = project_dir / "cache"
        temp_dir = project_dir / "temp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for key in ("TMPDIR", "TEMP", "TMP"):
            os.environ[key] = str(temp_dir)
        for key in ("TORCH_HOME", "HF_HOME", "TRANSFORMERS_CACHE", "PYTORCH_KERNEL_CACHE"):
            os.environ[key] = str(cache_dir / key.lower())
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    def _default_project_state(self) -> dict[str, Any]:
        """Build the default state used for a newly created project.

        Returns:
            JSON-style project state with fresh paths and default settings.
        """

        runs_dir = Path.cwd() / "runs"
        dataset_dir = runs_dir / "dataset"
        model_dir = runs_dir / "model"
        fine_tune_dir = runs_dir / "fine_tune"
        export_dir = runs_dir / "export"
        return {
            "schema": "micro_llm_creator_project",
            "version": 1,
            "project_name": "",
            "project_dir": "",
            "paths": {
                "source_vault": "",
                "dataset_core": str(dataset_dir),
                "training_dataset": str(dataset_dir),
                "model_output": str(model_dir),
                "export_model_core": str(model_dir),
                "export_output": str(export_dir),
                "llama_cpp_dir": "",
                "gguf_output_path": str(export_dir / "model.gguf"),
                "gguf_model": "",
                "microgpt_chat_model": "",
                "tokenizer_import": "",
                "resume_checkpoint": "",
                "fine_tune_checkpoint": "",
                "fine_tune_output": str(fine_tune_dir),
            },
            "dataset": {
                "domain_plan_preset": "Balanced Tiny LLM",
                "domain_plan": dataset_plan_defaults(),
                "default_data_paths": [str(path) for path, _category in iter_default_data_files()],
                "auto_vocab": True,
                "manual_vocab_size": 8000,
                "include_conversation_datasets": False,
                "dataset_stage": "base",
                "conversation_datasets": [],
                "conversation_sample_limit": 20000,
                "conversation_dataset_path": "",
                "instruction_dataset_path": "",
                "mixture_weights": {},
                "min_frequency": 2,
                "context_length": 128,
                "validation_split": 0.1,
                "lowercase": False,
                "max_workers": 4,
                "prepare_mode": "incremental",
                "tokenizer_strategy": "auto",
                "code_training_mode": True,
                "include_prose": True,
                "include_source_code": True,
                "extract_code_blocks": True,
                "preserve_indentation": True,
                "instruction_samples": True,
                "reasoning_sample_mode": "scaffold",
            },
            "training": {
                "preset": "Tiny",
                "architecture_style": "Classic GPT",
                "launch_target": "local",
                "training_mode": "pretrain",
                "training_stage": "base",
                "peft_method": "none",
                "lora_rank": 8,
                "lora_alpha": 16.0,
                "lora_dropout": 0.05,
                "lora_target_modules": "attention",
                "n_embd": 128,
                "n_head": 4,
                "n_layer": 4,
                "context_length": 128,
                "dropout": 0.1,
                "training_profile": "Stable LLM",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.0003,
                "weight_decay": 0.1,
                "gradient_accumulation": 1,
                "warmup_steps": 100,
                "eval_interval": 100,
                "max_eval_batches": 50,
                "save_interval": 500,
                "data_loader_workers": 0,
                "max_grad_norm": 1.0,
                "activation_checkpointing": False,
                "seed": 1337,
                "device": self.device.currentText(),
                "use_amp": self.use_amp_default,
                "resume": True,
                "require_compatible_resume": True,
                "benchmark_prompts": "\n\n".join(DEFAULT_BENCHMARK_PROMPTS),
                "benchmark_tokens": 128,
                "benchmark_temperature": 0.7,
                "benchmark_kv_cache": True,
            },
            "export": {
                "quantization": "FP16 checkpoint",
                "gguf_outtype": "f16",
            },
            "chat": {
                "model_backend": "gguf",
                "context": 2048,
                "cpu_threads": 4,
                "gpu_layers": -1,
                "thinking_enabled": True,
                "reasoning_effort": "Balanced",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "system_prompt": "",
            },
            "distributed": {
                "host": "0.0.0.0",
                "port": 8765,
                "artifact_root": str(Path.home() / ".micro_llm_creator" / "artifacts"),
                "public_url": "http://127.0.0.1:8765",
            },
            "artifacts": {},
        }

    def _reset_project_runtime_state(self) -> None:
        """Clear logs, progress, charts, and status labels for a new project."""

        self.dataset_log.clear()
        self.training_log.clear()
        self.fine_tune_log.clear()
        self.benchmark_log.clear()
        self.export_log.setPlainText(
            "Export options:\n"
            "- Bundle copies final_model.pt, tokenizer.json, and training_summary.json.\n"
            "- HF package writes model_core/hf_model for portable MicroGPT loading.\n"
            "- FP16 checkpoint quantization works now.\n"
            "- GGUF conversion uses llama.cpp when model_core/hf_model exists.\n"
            "- Native MicroGPT checkpoints are not written as fake GGUF files.\n"
        )
        for progress in (
            self.dataset_progress,
            self.training_progress,
            self.fine_tune_progress,
            self.benchmark_progress,
            self.export_progress,
            self.chat_progress,
        ):
            progress.setRange(0, 100)
            progress.setValue(0)
        self.dataset_status.setText("Dataset: not prepared")
        self.train_status.setText("Training: idle")
        self.export_status.setText("Export: waiting")
        self.chat_status.setText("Chat: no model loaded")
        self.prepare_button.setText("Prepare Dataset")
        self.train_button.setText("Start Training")
        self.fine_tune_button.setText("Start Fine-Tune")
        self.stop_dataset_button.setEnabled(False)
        self.stop_training_button.setEnabled(False)
        self.stop_fine_tune_button.setEnabled(False)
        self.stop_benchmark_button.setEnabled(False)
        self.stop_chat_button.setEnabled(False)
        self.load_llm_button.setText("Load Model")
        self._update_chat_backend_controls()
        self._reset_dataset_quality_report()
        self.training_epoch_metric.setText("Epoch: -")
        self.training_step_metric.setText("Step: -")
        self.training_loss_metric.setText("Train loss: -")
        self.training_val_metric.setText("Val loss: -")
        self.training_health_metric.setText("Health: -")
        self.training_health_points = []
        self.training_lr_metric.setText("LR: -")
        self.training_speed_metric.setText("Speed: -")
        self.training_grad_metric.setText("Grad: -")
        self.training_vram_metric.setText("VRAM: -")
        self.training_eta_metric.setText("ETA: -")
        self.model_size_metric.setText("Model: -")
        self.vram_estimate_metric.setText("VRAM est: -")
        self.parameter_breakdown_metric.setText("Params: -")
        self.memory_breakdown_metric.setText("Memory: -")
        self.architecture_advisor_metric.setText("Advisor: -")
        self.history_metric.setText(f"Runs: {len(self._load_training_history())}")
        self.loss_chart.clear()
        self.optimization_chart.clear()
        self.stability_chart.clear()
        self.throughput_chart.clear()
        self.memory_chart.clear()
        self.live_prediction_chart.update_distribution(0, None)
        self.live_attention_chart.update_heatmap(0, None)
        self.live_activation_chart.update_histogram(0, None)
        self.live_gradient_chart.update_flow(self.n_layer.value(), None, 0)
        self.live_sample_text.setText("Training text: -")
        self.telemetry_db_path = None
        self.telemetry_run_id = ""
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_scrub_active = False
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_time_slider.blockSignals(False)
        self.live_timeline_label.setText("Timeline: no saved telemetry")
        self._set_meter(self.live_cpu_bar, "CPU", self._system_cpu_value())
        self._set_meter(self.live_gpu_bar, "GPU memory", None)
        self._set_meter(self.live_vram_bar, "VRAM reserved", None)
        self._set_meter(self.live_ram_bar, "System RAM", self._system_ram_value())
        self.live_worker_status.setText(f"CPU workers: {self.data_loader_workers.value()}")
        self._clear_chat_messages()
        self.chat_markdown = ""
        self.chat_stream_prefix = ""
        self.chat_stream_reply = ""
        self.chat_stats.setText("Idle")
        self._add_chat_message("assistant", "Load a GGUF or MicroGPT model to start testing.")

    def _project_state_dict(self, project_name: str, project_dir: Path) -> dict[str, Any]:
        """Collect all UI state that defines a Micro LLM project.

        Args:
            project_name: User-facing project name.
            project_dir: Folder where the project file will live.

        Returns:
            JSON-serializable project state.
        """

        dataset_dir = Path(self.dataset_dir.text()) if self.dataset_dir.text().strip() else None
        model_dir = Path(self.model_dir.text()) if self.model_dir.text().strip() else None
        export_dir = Path(self.export_dir.text()) if self.export_dir.text().strip() else None
        return {
            "schema": "micro_llm_creator_project",
            "version": 1,
            "project_name": project_name,
            "project_dir": str(project_dir),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "paths": {
                "source_vault": self.input_dir.text(),
                "dataset_core": self.dataset_dir.text(),
                "training_dataset": self.train_data_dir.text(),
                "model_output": self.model_dir.text(),
                "export_model_core": self.export_model_dir.text(),
                "export_output": self.export_dir.text(),
                "llama_cpp_dir": self.llama_cpp_dir.text(),
                "gguf_output_path": self.gguf_output_path.text(),
                "gguf_model": self.gguf_path.text(),
                "microgpt_chat_model": self.microgpt_chat_path.text(),
                "tokenizer_import": self.tokenizer_path.text(),
                "resume_checkpoint": self.resume_checkpoint.text(),
                "fine_tune_checkpoint": self.fine_tune_checkpoint.text(),
                "fine_tune_output": self.fine_tune_output_dir.text(),
            },
            "dataset": {
                "domain_plan_preset": self.dataset_plan_preset.currentText() if hasattr(self, "dataset_plan_preset") else "Balanced Tiny LLM",
                "domain_plan": self._dataset_plan_from_ui(),
                "default_data_paths": [str(path) for path in self._selected_default_data_paths()],
                "auto_vocab": self.auto_vocab.isChecked(),
                "manual_vocab_size": self.manual_vocab_size.value(),
                "include_conversation_datasets": self.include_conversation_datasets.isChecked(),
                "dataset_stage": self._dataset_stage_value(),
                "conversation_datasets": self._selected_conversation_datasets(),
                "conversation_sample_limit": self.conversation_sample_limit.value(),
                "mixture_weights": self._mixture_weights_from_ui(),
                "min_frequency": self.min_frequency.value(),
                "context_length": self.context_length.value(),
                "validation_split": self.validation_split.value(),
                "lowercase": self.lowercase.isChecked(),
                "max_workers": self.max_workers.value(),
                "prepare_mode": self._prepare_mode_value(),
                "tokenizer_strategy": self._tokenizer_strategy_value(),
                "code_training_mode": self.code_training_mode.isChecked(),
                "include_prose": self.include_prose.isChecked(),
                "include_source_code": self.include_source_code.isChecked(),
                "extract_code_blocks": self.extract_code_blocks.isChecked(),
                "preserve_indentation": self.preserve_indentation.isChecked(),
                "instruction_samples": self.instruction_samples.isChecked(),
                "reasoning_sample_mode": self._reasoning_sample_mode_value(),
            },
            "training": {
                "preset": self.preset.currentText(),
                "architecture_style": self.architecture_style.currentText(),
                "launch_target": self._training_launch_target_value(),
                "fine_tune_launch_target": self._fine_tune_launch_target_value(),
                "training_stage": self._training_stage_value(),
                "n_embd": self.n_embd.value(),
                "n_head": self.n_head.value(),
                "attention_type": self._attention_type_value(),
                "kv_head_count": self.kv_head_count.value(),
                "attention_backend": self._attention_backend_value(),
                "attention_window": self.attention_window.value(),
                "training_mode": self._training_mode_value(),
                "peft_method": self._peft_method_value(),
                "lora_rank": self.lora_rank.value(),
                "lora_alpha": self.lora_alpha.value(),
                "lora_dropout": self.lora_dropout.value(),
                "lora_target_modules": self._lora_target_value(),
                "n_layer": self.n_layer.value(),
                "context_length": self.train_context_length.value(),
                "dropout": self.dropout.value(),
                "training_profile": self.training_profile.currentText(),
                "epochs": self.epochs.value(),
                "batch_size": self.batch_size.value(),
                "learning_rate": self.learning_rate.value(),
                "weight_decay": self.weight_decay.value(),
                "optimizer_name": self._optimizer_value(),
                "scheduler_name": self._scheduler_value(),
                "scheduler_min_lr_ratio": self.min_lr_ratio.value(),
                "polynomial_power": self.polynomial_power.value(),
                "gradient_accumulation": self.gradient_accumulation.value(),
                "sample_stride": self.sample_stride.value(),
                "warmup_steps": self.warmup_steps.value(),
                "eval_interval": self.eval_interval.value(),
                "max_eval_batches": self.max_eval_batches.value(),
                "save_interval": self.save_interval.value(),
                "data_loader_workers": self.data_loader_workers.value(),
                "max_grad_norm": self.max_grad_norm.value(),
                "activation_checkpointing": self.activation_checkpointing.isChecked(),
                "seed": self.seed.value(),
                "device": self.device.currentText(),
                "use_amp": self.use_amp.isChecked(),
                "precision": self._precision_value(),
                "resume": self.resume_training.isChecked(),
                "require_compatible_resume": self.resume_safety.isChecked(),
                "early_stopping": self.early_stopping.isChecked(),
                "benchmark_prompts": self.benchmark_prompts.toPlainText(),
                "benchmark_tokens": self.benchmark_tokens.value(),
                "benchmark_temperature": self.benchmark_temperature.value(),
                "benchmark_kv_cache": self.benchmark_kv_cache.isChecked(),
            },
            "export": {
                "quantization": self.quant_mode.currentText(),
                "gguf_outtype": self.gguf_outtype.currentText(),
            },
            "chat": {
                "model_backend": self._chat_backend_value(),
                "context": self.llama_context.value(),
                "cpu_threads": self.llama_threads.value(),
                "gpu_layers": self.llama_gpu_layers.value(),
                "thinking_enabled": self.thinking_enabled.isChecked(),
                "reasoning_effort": self.reasoning_effort.currentText(),
                "max_tokens": self.chat_max_tokens.value(),
                "temperature": self.chat_temperature.value(),
                "top_p": self.chat_top_p.value(),
                "repeat_penalty": self.chat_repeat_penalty.value(),
                "system_prompt": self.system_prompt.toPlainText(),
            },
            "distributed": {
                "host": self.coordinator_host.text(),
                "port": self.coordinator_port.value(),
                "artifact_root": self.coordinator_artifact_root.text(),
                "public_url": self.coordinator_public_url.text(),
            },
            "artifacts": {
                "dataset_summary": self._read_json_if_exists(dataset_dir / "dataset_summary.json") if dataset_dir else None,
                "training_summary": self._read_json_if_exists(model_dir / "training_summary.json") if model_dir else None,
                "export_summary": self._read_json_if_exists(export_dir / "export_summary.json") if export_dir else None,
            },
        }

    def _apply_project_state(self, data: dict[str, Any]) -> None:
        """Restore UI state from a saved project dictionary.

        Args:
            data: Project state loaded from JSON.
        """

        self.search_box.setText(str(data.get("project_name", "")))
        paths = data.get("paths", {})
        dataset = data.get("dataset", {})
        training = data.get("training", {})
        export = data.get("export", {})
        chat = data.get("chat", {})
        distributed = data.get("distributed", {})

        self.input_dir.setText(str(paths.get("source_vault", "")))
        self.dataset_dir.setText(str(paths.get("dataset_core", "")))
        self.train_data_dir.setText(str(paths.get("training_dataset", "")))
        self.model_dir.setText(str(paths.get("model_output", "")))
        self.export_model_dir.setText(str(paths.get("export_model_core", "")))
        self.export_dir.setText(str(paths.get("export_output", "")))
        self.llama_cpp_dir.setText(str(paths.get("llama_cpp_dir", "")))
        self.gguf_output_path.setText(str(paths.get("gguf_output_path", "")))
        self.gguf_path.setText(str(paths.get("gguf_model", "")))
        self.microgpt_chat_path.setText(str(paths.get("microgpt_chat_model", "")))
        self.tokenizer_path.setText(str(paths.get("tokenizer_import", "")))
        self.resume_checkpoint.setText(str(paths.get("resume_checkpoint", "")))
        self.fine_tune_checkpoint.setText(str(paths.get("fine_tune_checkpoint", "")))
        self.fine_tune_output_dir.setText(str(paths.get("fine_tune_output", "")))

        self._set_dataset_plan(
            dict(dataset.get("domain_plan", DATASET_DOMAIN_DEFAULTS)),
            str(dataset.get("domain_plan_preset", "Balanced Tiny LLM")),
        )
        saved_default_data_paths = dataset.get("default_data_paths")
        self._set_selected_default_data_paths(
            list(saved_default_data_paths) if saved_default_data_paths is not None else None
        )
        self.auto_vocab.setChecked(bool(dataset.get("auto_vocab", True)))
        self.manual_vocab_size.setValue(int(dataset.get("manual_vocab_size", self.manual_vocab_size.value())))
        include_conversation = bool(dataset.get("include_conversation_datasets", False))
        self._set_dataset_stage(str(dataset.get("dataset_stage", "base")))
        self.include_conversation_datasets.setChecked(include_conversation)
        self._set_selected_conversation_datasets(list(dataset.get("conversation_datasets", [])))
        self.conversation_sample_limit.setValue(int(dataset.get("conversation_sample_limit", self.conversation_sample_limit.value())))
        self._set_mixture_weights(dict(dataset.get("mixture_weights", {})))
        self.min_frequency.setValue(int(dataset.get("min_frequency", self.min_frequency.value())))
        self.context_length.setValue(int(dataset.get("context_length", self.context_length.value())))
        self.validation_split.setValue(float(dataset.get("validation_split", self.validation_split.value())))
        self.lowercase.setChecked(bool(dataset.get("lowercase", False)))
        self.max_workers.setValue(int(dataset.get("max_workers", self.max_workers.value())))
        self._set_combo_by_data(self.prepare_mode, str(dataset.get("prepare_mode", "incremental")), {
            "incremental": "Incremental update",
            "full_rebuild": "Full rebuild",
            "force_reprocess": "Force reprocess",
        })
        self._set_combo_by_data(self.tokenizer_strategy, str(dataset.get("tokenizer_strategy", "auto")), {
            "auto": "Auto",
            "train_new": "Train new tokenizer",
            "reuse_dataset": "Reuse dataset tokenizer",
            "import_tokenizer": "Import tokenizer.json",
        })
        self.code_training_mode.setChecked(bool(dataset.get("code_training_mode", True)))
        self.include_prose.setChecked(bool(dataset.get("include_prose", True)))
        self.include_source_code.setChecked(bool(dataset.get("include_source_code", True)))
        self.extract_code_blocks.setChecked(bool(dataset.get("extract_code_blocks", True)))
        self.preserve_indentation.setChecked(bool(dataset.get("preserve_indentation", True)))
        self.instruction_samples.setChecked(bool(dataset.get("instruction_samples", True)))
        self._set_combo_by_data(self.reasoning_sample_mode, str(dataset.get("reasoning_sample_mode", "scaffold")), {
            "scaffold": "Reasoning scaffold",
            "detailed": "Detailed code reasoning",
            "none": "No reasoning wrapper",
        })

        self._set_combo_text(self.preset, str(training.get("preset", self.preset.currentText())))
        self._set_combo_text(self.architecture_style, str(training.get("architecture_style", self.architecture_style.currentText())))
        self._set_combo_by_data(self.training_launch_target, str(training.get("launch_target", "local")), {
            "local": "Local machine",
            "remote": "Remote workers",
            "runpod": "RunPod cloud",
        })
        if hasattr(self, "fine_tune_launch_target"):
            self._set_combo_by_data(self.fine_tune_launch_target, str(training.get("fine_tune_launch_target", "local")), {
                "local": "Local machine",
                "remote": "Remote workers",
                "runpod": "RunPod cloud",
            })
        self.n_embd.setValue(int(training.get("n_embd", self.n_embd.value())))
        self.n_head.setValue(int(training.get("n_head", self.n_head.value())))
        self._set_combo_by_data(self.attention_type, str(training.get("attention_type", "mha")), {
            "mha": "Multi-head",
            "gqa": "Grouped-query",
            "mqa": "Multi-query",
        })
        self.kv_head_count.setValue(int(training.get("kv_head_count", self.kv_head_count.value())))
        self._set_combo_by_data(self.attention_backend, str(training.get("attention_backend", "sdpa")), {
            "sdpa": "SDPA / Flash when available",
            "manual": "Manual",
        })
        self.attention_window.setValue(int(training.get("attention_window", self.attention_window.value())))
        self._set_combo_by_data(self.training_mode, str(training.get("training_mode", "pretrain")), {
            "pretrain": "Pretrain from scratch",
            "fine_tune": "Fine-tune checkpoint",
            "instruction_fine_tune": "Instruction fine-tune",
            "conversation_fine_tune": "Conversation fine-tune",
            "code_fine_tune": "Code fine-tune",
        })
        training_stage = str(training.get("training_stage", ""))
        if training_stage == "instruction":
            self._set_combo_text(self.training_mode, "Instruction fine-tune")
        elif training_stage == "conversation":
            self._set_combo_text(self.training_mode, "Conversation fine-tune")
        elif training_stage == "code":
            self._set_combo_text(self.training_mode, "Code fine-tune")
        self._set_combo_by_data(self.peft_method, str(training.get("peft_method", "none")), {
            "none": "Full fine-tune",
            "lora": "LoRA adapters",
        })
        self.lora_rank.setValue(int(training.get("lora_rank", self.lora_rank.value())))
        self.lora_alpha.setValue(float(training.get("lora_alpha", self.lora_alpha.value())))
        self.lora_dropout.setValue(float(training.get("lora_dropout", self.lora_dropout.value())))
        self._set_combo_by_data(self.lora_targets, str(training.get("lora_target_modules", "attention")), {
            "attention": "Attention projections",
            "mlp": "MLP projections",
            "attention,mlp": "Attention + MLP",
        })
        self.n_layer.setValue(int(training.get("n_layer", self.n_layer.value())))
        self.train_context_length.setValue(int(training.get("context_length", self.train_context_length.value())))
        self.dropout.setValue(float(training.get("dropout", self.dropout.value())))
        self._set_combo_text(self.training_profile, str(training.get("training_profile", self.training_profile.currentText())))
        self.epochs.setValue(int(training.get("epochs", self.epochs.value())))
        self.batch_size.setValue(int(training.get("batch_size", self.batch_size.value())))
        self.learning_rate.setValue(float(training.get("learning_rate", self.learning_rate.value())))
        self.weight_decay.setValue(float(training.get("weight_decay", self.weight_decay.value())))
        self._set_combo_by_data(self.optimizer_name, str(training.get("optimizer_name", "adamw")), {
            "adamw": "AdamW",
            "adam": "Adam",
            "lion": "Lion",
            "adafactor": "Adafactor",
        })
        self._set_combo_by_data(self.scheduler_name, str(training.get("scheduler_name", "warmup_linear")), {
            "warmup_linear": "Warmup linear",
            "cosine": "Cosine decay",
            "polynomial": "Polynomial decay",
            "one_cycle": "One-cycle",
            "constant": "Constant",
        })
        self.min_lr_ratio.setValue(float(training.get("scheduler_min_lr_ratio", self.min_lr_ratio.value())))
        self.polynomial_power.setValue(float(training.get("polynomial_power", self.polynomial_power.value())))
        self.gradient_accumulation.setValue(int(training.get("gradient_accumulation", self.gradient_accumulation.value())))
        self.sample_stride.setValue(int(training.get("sample_stride", self.sample_stride.value())))
        self.warmup_steps.setValue(int(training.get("warmup_steps", self.warmup_steps.value())))
        self.eval_interval.setValue(int(training.get("eval_interval", self.eval_interval.value())))
        self.max_eval_batches.setValue(int(training.get("max_eval_batches", self.max_eval_batches.value())))
        self.save_interval.setValue(int(training.get("save_interval", self.save_interval.value())))
        self.data_loader_workers.setValue(int(training.get("data_loader_workers", self.data_loader_workers.value())))
        self.max_grad_norm.setValue(float(training.get("max_grad_norm", self.max_grad_norm.value())))
        self.activation_checkpointing.setChecked(bool(training.get("activation_checkpointing", False)))
        self.seed.setValue(int(training.get("seed", self.seed.value())))
        self._set_combo_text(self.device, str(training.get("device", self.device.currentText())))
        self.use_amp.setChecked(bool(training.get("use_amp", self.use_amp.isChecked())))
        self._set_combo_by_data(self.precision, str(training.get("precision", "fp16")), {
            "fp16": "FP16",
            "bf16": "BF16",
            "fp32": "FP32",
        })
        self.resume_training.setChecked(bool(training.get("resume", self.resume_training.isChecked())))
        self.resume_safety.setChecked(bool(training.get("require_compatible_resume", True)))
        self.early_stopping.setChecked(bool(training.get("early_stopping", True)))
        self.benchmark_prompts.setPlainText(str(training.get("benchmark_prompts", self.benchmark_prompts.toPlainText())))
        self.benchmark_tokens.setValue(int(training.get("benchmark_tokens", self.benchmark_tokens.value())))
        self.benchmark_temperature.setValue(float(training.get("benchmark_temperature", self.benchmark_temperature.value())))
        self.benchmark_kv_cache.setChecked(bool(training.get("benchmark_kv_cache", True)))

        self._set_combo_text(self.quant_mode, str(export.get("quantization", self.quant_mode.currentText())))
        self._set_combo_text(self.gguf_outtype, str(export.get("gguf_outtype", self.gguf_outtype.currentText())))
        self.llama_context.setValue(int(chat.get("context", self.llama_context.value())))
        self._set_combo_by_data(self.chat_model_backend, str(chat.get("model_backend", "gguf")), {
            "gguf": "GGUF / llama.cpp",
            "microgpt": "MicroGPT checkpoint",
        })
        self.llama_threads.setValue(int(chat.get("cpu_threads", self.llama_threads.value())))
        self.llama_gpu_layers.setValue(int(chat.get("gpu_layers", self.llama_gpu_layers.value())))
        self.thinking_enabled.setChecked(bool(chat.get("thinking_enabled", True)))
        self._set_combo_text(self.reasoning_effort, str(chat.get("reasoning_effort", self.reasoning_effort.currentText())))
        self.reasoning_effort.setEnabled(self.thinking_enabled.isChecked())
        self.chat_max_tokens.setValue(int(chat.get("max_tokens", self.chat_max_tokens.value())))
        self.chat_temperature.setValue(float(chat.get("temperature", self.chat_temperature.value())))
        self.chat_top_p.setValue(float(chat.get("top_p", self.chat_top_p.value())))
        self.chat_repeat_penalty.setValue(float(chat.get("repeat_penalty", self.chat_repeat_penalty.value())))
        self.system_prompt.setPlainText(str(chat.get("system_prompt", "")))
        if hasattr(self, "coordinator_host"):
            self.coordinator_host.setText(str(distributed.get("host", self.coordinator_host.text())))
            self.coordinator_port.setValue(int(distributed.get("port", self.coordinator_port.value())))
            self.coordinator_artifact_root.setText(str(distributed.get("artifact_root", self.coordinator_artifact_root.text())))
            self.coordinator_public_url.setText(str(distributed.get("public_url", self.coordinator_public_url.text())))
        self._update_tokenizer_strategy_controls()
        self._update_training_mode_controls()
        self._restore_artifact_status(data.get("artifacts", {}))
        self.refresh_fine_tune_workflow()

    def _restore_artifact_status(self, artifacts: dict[str, Any]) -> None:
        """Refresh top-bar and button state from saved or existing artifacts.

        Args:
            artifacts: Saved artifact summary dictionary.
        """

        dataset_dir = Path(self.dataset_dir.text()) if self.dataset_dir.text().strip() else None
        if dataset_dir and self._dataset_artifacts_exist(dataset_dir):
            summary = self._read_json_if_exists(dataset_dir / "dataset_summary.json") or artifacts.get("dataset_summary") or {}
            document_count = int(summary.get("document_count", 0) or 0)
            token_count = int(summary.get("token_count", 0) or 0)
            code_count = int(summary.get("code_sample_count", 0) or 0)
            prose_count = int(summary.get("prose_sample_count", 0) or 0)
            conversation_count = int(summary.get("conversation_sample_count", 0) or 0)
            vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
            self._update_dataset_quality_report(summary)
            self.prepare_button.setText("DataSet Prepared")
            self.dataset_progress.setValue(100)
            if vocab_size:
                self.auto_vocab_label.setText(f"{vocab_size:,}")
            if code_count or prose_count or conversation_count:
                self.dataset_status.setText(
                    f"Dataset: {code_count:,} code, {prose_count:,} prose, {conversation_count:,} chat, {token_count:,} tokens"
                )
            elif document_count or token_count:
                self.dataset_status.setText(f"Dataset: {document_count:,} files, {token_count:,} tokens")
            else:
                self.dataset_status.setText("Dataset: prepared")
            version = summary.get("dataset_version", {})
            if isinstance(version, dict) and version.get("version_id"):
                self.dataset_log.append(f"Dataset version: {version['version_id']}")
            self.train_data_dir.setText(str(dataset_dir))
            self.dataset_log.append(f"Dataset already prepared: {dataset_dir}")
        else:
            self.prepare_button.setText("Prepare Dataset")
            self.dataset_progress.setValue(0)
            self.dataset_status.setText("Dataset: not prepared")
            self.auto_vocab_label.setText("Auto after reading files")
            self._reset_dataset_quality_report()

        model_dir = Path(self.model_dir.text()) if self.model_dir.text().strip() else None
        if model_dir and (model_dir / "final_model.pt").exists():
            summary = self._read_json_if_exists(model_dir / "training_summary.json") or artifacts.get("training_summary") or {}
            loss = summary.get("final_train_loss")
            self.train_status.setText(f"Training: loss {float(loss):.4f}" if loss is not None else "Training: model ready")
            self.export_model_dir.setText(str(model_dir))

        export_dir = Path(self.export_dir.text()) if self.export_dir.text().strip() else None
        if export_dir and export_dir.exists() and any(export_dir.iterdir()):
            self.export_status.setText("Export: artifacts found")

    @staticmethod
    def _dataset_artifacts_exist(dataset_dir: Path) -> bool:
        """Return whether a dataset folder has the required prepared files.

        Args:
            dataset_dir: Dataset folder.

        Returns:
            True if required dataset artifacts exist.
        """

        if not dataset_dir.exists():
            return False
        if not (dataset_dir / "tokenizer.json").exists():
            return False
        has_npy_tokens = (dataset_dir / "train_tokens.npy").exists() and (dataset_dir / "val_tokens.npy").exists()
        has_json_tokens = (dataset_dir / "train_tokens.json").exists() and (dataset_dir / "val_tokens.json").exists()
        return has_npy_tokens or has_json_tokens

    @staticmethod
    def _safe_project_name(project_name: str) -> str:
        """Return a filesystem-safe project folder name.

        Args:
            project_name: Raw user project name.

        Returns:
            Safe folder name.
        """

        return re.sub(r"[^A-Za-z0-9_.-]+", "_", project_name).strip("._") or "MicroLLMProject"

    @staticmethod
    def _read_json_if_exists(path: Path) -> Optional[Any]:
        """Read a JSON file when it exists.

        Args:
            path: JSON file path.

        Returns:
            Parsed JSON or ``None``.
        """

        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        """Set combo text when the value exists.

        Args:
            combo: Combo box to update.
            text: Display text to select.
        """

        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
        elif combo.isEditable():
            combo.setEditText(text)

    def _set_combo_by_data(self, combo: QComboBox, value: str, labels: dict[str, str]) -> None:
        """Set a combo by internal saved value.

        Args:
            combo: Combo box to update.
            value: Internal saved value.
            labels: Mapping from saved value to display label.
        """

        self._set_combo_text(combo, labels.get(value, value))

    def _run_task(
        self,
        fn,
        args,
        on_finished,
        log: QTextEdit,
        progress_bar: QProgressBar,
        with_progress: bool = False,
        button: Optional[QPushButton] = None,
        stop_button: Optional[QPushButton] = None,
        busy_text: str = "Working",
        task_kind: str = "",
        isolate_process: bool = False,
    ) -> None:
        """Run a long task on a background thread.

        Args:
            fn: Callable to execute.
            args: Positional arguments for the callable.
            on_finished: Slot called with the task result.
            log: Log widget receiving progress messages.
            progress_bar: Progress bar receiving percent updates.
            with_progress: Whether to pass a progress callback to the task.
            button: Optional button to disable while running.
            stop_button: Optional stop button to enable while running.
            busy_text: Button text shown while running.
            task_kind: Optional notification stage key.
            isolate_process: Run the task inside a child process.
        """

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please wait for the current task to finish.")
            return

        LOGGER.info("Starting background task: %s", getattr(fn, "__name__", str(fn)))
        self.active_task_kind = task_kind
        if button:
            self._set_button_busy(button, busy_text)
        if stop_button:
            stop_button.setEnabled(True)
            self.active_stop_button = stop_button

        self.stop_event = Event()
        self.progress_queue = Queue()
        self.active_log = log
        self.active_progress_bar = progress_bar
        self.thread = QThread(self)
        worker_class = ProcessTaskWorker if isolate_process else TaskWorker
        self.worker = worker_class(
            fn,
            *args,
            progress_queue=self.progress_queue,
            with_progress=with_progress,
            stop_event=self.stop_event,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(on_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self._task_failed_from_worker)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.progress_timer.start(100)
        self.thread.start()

    @Slot(str)
    def _task_failed_from_worker(self, message: str) -> None:
        """Handle a worker failure on the UI thread.

        Args:
            message: Error message emitted by the worker.
        """

        if self.active_log is None or self.active_progress_bar is None:
            return
        LOGGER.error("Background task failed: %s", message)
        self._task_failed(message, self.active_log, self.active_progress_bar)

    def stop_active_task(self) -> None:
        """Request a graceful stop for the active background task."""

        if self.stop_event is None:
            return
        LOGGER.info("Stop requested for active background task")
        self.stop_event.set()
        self._notify_failure("Stop requested", "The task is stopping at the next safe point.")
        if self.active_log is not None:
            self.active_log.append("Stop requested. Finishing the current safe point...")
        if self.active_stop_button is not None:
            self.active_stop_button.setEnabled(False)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                LOGGER.exception(
                    "Failed to empty CUDA cache in _thread_finished")

    @Slot()
    def request_shutdown_from_signal(self) -> None:
        """Handle Ctrl+C from a terminal without leaving Qt threads wedged."""

        self.interrupt_count += 1
        if self.interrupt_count > 1:
            os._exit(130)
        if self.stop_event is not None:
            self.stop_event.set()
        if self.active_log is not None:
            self.active_log.append("Interrupt received. Requesting stop...")
        self.project_state.setText("Stopping")
        if self.thread is None:
            QApplication.quit()
            return
        QTimer.singleShot(3000, lambda: os._exit(130) if self.thread is not None else QApplication.quit())

    def closeEvent(self, event: Any) -> None:
        """Clean up background services before the window closes.

        Args:
            event: Qt close event.
        """

        if self.thread is not None:
            if self.stop_event is not None:
                self.stop_event.set()
            if self.active_log is not None:
                self.active_log.append("Close requested. Stopping active task first...")
            self.project_state.setText("Stopping")
            LOGGER.info("Close requested while background task is running; waiting for task shutdown")
            event.ignore()
            QTimer.singleShot(500, self.close)
            return
        if self.coordinator_server is not None:
            self.stop_coordinator_server()
        super().closeEvent(event)

    def _handle_progress(self, event: object, log: QTextEdit, progress_bar: QProgressBar) -> None:
        """Apply one progress event to UI widgets.

        Args:
            event: Progress dictionary or message.
            log: Log widget to append messages to.
            progress_bar: Progress bar to update.
        """

        if isinstance(event, dict):
            if event.get("type") == "chat_delta":
                self._apply_chat_delta(event)
                return
            message = event.get("message")
            percent = event.get("percent")
            if log in (self.training_log, getattr(self, "fine_tune_log", None)):
                self._update_training_metrics(event, update_fine_tune=log is getattr(self, "fine_tune_log", None))
            if message:
                log.append(str(message))
                if log in (self.training_log, getattr(self, "fine_tune_log", None)) and hasattr(self, "live_log"):
                    self.live_log.append(str(message))
            if percent is not None:
                progress_bar.setValue(max(0, min(100, int(percent))))
                if log in (self.training_log, getattr(self, "fine_tune_log", None)) and hasattr(self, "live_progress"):
                    self.live_progress.setValue(max(0, min(100, int(percent))))
        else:
            log.append(str(event))

    def _notify_progress(self, event: dict[str, Any]) -> None:
        """Send throttled external progress notifications for long tasks.

        Args:
            event: Progress event emitted by a worker.
        """

        if not self.active_task_kind or self.notification_manager is None:
            return
        if self.active_task_kind not in {"dataset", "training", "fine_tune"}:
            return
        title = {
            "dataset": "Dataset preparation",
            "training": "Model training",
            "fine_tune": "Fine-tuning",
        }[self.active_task_kind]
        percent = event.get("percent")
        self.notification_manager.notify_progress(
            self.active_task_kind,
            title,
            self._notification_lines_from_event(event),
            int(percent) if percent is not None else None,
        )

    def _notify_complete(self, stage_key: str, title: str, lines: list[str]) -> None:
        """Send an external completion notification when configured.

        Args:
            stage_key: Notification stage key.
            title: User-facing title.
            lines: Plain-text summary lines.
        """

        if self.notification_manager is not None:
            self.notification_manager.notify_complete(stage_key, title, lines)

    def _notify_failure(self, title: str, message: str) -> None:
        """Send an external failure or stop notification for the active task.

        Args:
            title: User-facing title.
            message: Failure details.
        """

        if self.active_task_kind and self.notification_manager is not None:
            self.notification_manager.notify_failure(self.active_task_kind, title, message)

    def _notification_lines_from_event(self, event: dict[str, Any]) -> list[str]:
        """Build compact notification text from a worker progress event.

        Args:
            event: Progress event emitted by a worker.

        Returns:
            Body lines for the notification message.
        """

        lines: list[str] = []
        if event.get("message"):
            lines.append(str(event["message"]))
        if "epoch" in event and "total_epochs" in event:
            lines.append(f"Epoch: {event['epoch']}/{event['total_epochs']}")
        if "step" in event and "total_steps" in event:
            lines.append(f"Step: {event['step']}/{event['total_steps']}")
        train_loss = self._finite_metric(event.get("train_loss"))
        if train_loss is not None:
            lines.append(f"Train loss: {float(train_loss):.4f}")
        val_loss = self._finite_metric(event.get("val_loss"))
        if val_loss is not None:
            lines.append(f"Validation loss: {float(val_loss):.4f}")
        learning_rate = self._finite_metric(event.get("learning_rate"))
        if learning_rate is not None:
            lines.append(f"Learning rate: {float(learning_rate):.2e}")
        tokens_per_second = self._finite_metric(event.get("tokens_per_second"))
        if tokens_per_second is not None:
            lines.append(f"Speed: {float(tokens_per_second):.0f} tokens/sec")
        eta_seconds = self._finite_metric(event.get("eta_seconds"))
        if eta_seconds is not None:
            lines.append(f"ETA: {self._format_duration(float(eta_seconds))}")
        vram_allocated = self._finite_metric(event.get("vram_allocated_gb"))
        vram_reserved = self._finite_metric(event.get("vram_reserved_gb"))
        if vram_allocated is not None or vram_reserved is not None:
            allocated = "-" if vram_allocated is None else f"{float(vram_allocated):.2f} GB"
            reserved = "-" if vram_reserved is None else f"{float(vram_reserved):.2f} GB"
            lines.append(f"VRAM: {allocated} allocated, {reserved} reserved")
        return lines[:10]

    def _update_training_metrics(self, event: dict[str, Any], update_fine_tune: bool = False) -> None:
        """Update training metric chips from a progress event.

        Args:
            event: Progress event emitted by the training backend.
            update_fine_tune: Whether to mirror metrics into the Fine-Tuning tab chips.
        """

        if "epoch" in event and "total_epochs" in event:
            self.training_epoch_metric.setText(f"Epoch: {event['epoch']}/{event['total_epochs']}")
            if update_fine_tune and hasattr(self, "fine_tune_epoch_metric"):
                self.fine_tune_epoch_metric.setText(f"Epoch: {event['epoch']}/{event['total_epochs']}")
        if "step" in event and "total_steps" in event:
            self.training_step_metric.setText(f"Step: {event['step']}/{event['total_steps']}")
            if update_fine_tune and hasattr(self, "fine_tune_step_metric"):
                self.fine_tune_step_metric.setText(f"Step: {event['step']}/{event['total_steps']}")
        train_loss = self._finite_metric(event.get("train_loss"))
        if train_loss is not None:
            self.training_loss_metric.setText(f"Train loss: {float(train_loss):.4f}")
            if update_fine_tune and hasattr(self, "fine_tune_loss_metric"):
                self.fine_tune_loss_metric.setText(f"Train loss: {float(train_loss):.4f}")
        val_loss = self._finite_metric(event.get("val_loss"))
        if val_loss is not None:
            self.training_val_metric.setText(f"Val loss: {float(val_loss):.4f}")
            if update_fine_tune and hasattr(self, "fine_tune_val_metric"):
                self.fine_tune_val_metric.setText(f"Val loss: {float(val_loss):.4f}")
        step = event.get("step")
        if step is not None and (train_loss is not None or val_loss is not None):
            step_int_for_loss = int(step)
            self.loss_chart.add_metrics(step_int_for_loss, train_loss, val_loss)
            self._update_training_health(step_int_for_loss, train_loss, val_loss)
        if step is None:
            return
        step_int = int(step)
        self._record_live_metric(event)
        learning_rate = self._finite_metric(event.get("learning_rate"))
        grad_norm = self._finite_metric(event.get("grad_norm"))
        weight_norm = self._finite_metric(event.get("weight_norm"))
        update_ratio = self._finite_metric(event.get("update_ratio"))
        tokens_per_second = self._finite_metric(event.get("tokens_per_second"))
        samples_per_second = self._finite_metric(event.get("samples_per_second"))
        vram_allocated = self._finite_metric(event.get("vram_allocated_gb"))
        vram_reserved = self._finite_metric(event.get("vram_reserved_gb"))
        gpu_memory = self._finite_metric(event.get("gpu_memory_percent"))
        system_cpu = self._finite_metric(event.get("system_cpu_percent"))
        system_ram = self._finite_metric(event.get("system_ram_percent"))
        data_workers = event.get("data_loader_workers")
        eta_seconds = self._finite_metric(event.get("eta_seconds"))
        if learning_rate is not None:
            self.training_lr_metric.setText(f"LR: {float(learning_rate):.2e}")
            if update_fine_tune and hasattr(self, "fine_tune_lr_metric"):
                self.fine_tune_lr_metric.setText(f"LR: {float(learning_rate):.2e}")
        if grad_norm is not None:
            self.training_grad_metric.setText(f"Grad: {float(grad_norm):.3f}")
            if update_fine_tune and hasattr(self, "fine_tune_grad_metric"):
                self.fine_tune_grad_metric.setText(f"Grad: {float(grad_norm):.3f}")
        if tokens_per_second is not None:
            self.training_speed_metric.setText(f"Speed: {float(tokens_per_second):.0f} tok/s")
            if update_fine_tune and hasattr(self, "fine_tune_speed_metric"):
                self.fine_tune_speed_metric.setText(f"Speed: {float(tokens_per_second):.0f} tok/s")
        if vram_allocated is not None:
            self.training_vram_metric.setText(f"VRAM: {float(vram_allocated):.2f} GB")
        if eta_seconds is not None:
            self.training_eta_metric.setText(f"ETA: {self._format_duration(float(eta_seconds))}")
            if update_fine_tune and hasattr(self, "fine_tune_eta_metric"):
                self.fine_tune_eta_metric.setText(f"ETA: {self._format_duration(float(eta_seconds))}")
        if learning_rate is not None or grad_norm is not None:
            self.optimization_chart.add_values(step_int, learning_rate, grad_norm)
        if weight_norm is not None or update_ratio is not None:
            self.stability_chart.add_values(step_int, weight_norm, update_ratio)
        if tokens_per_second is not None or samples_per_second is not None:
            self.throughput_chart.add_values(step_int, tokens_per_second, samples_per_second)
        if vram_allocated is not None or vram_reserved is not None:
            self.memory_chart.add_values(step_int, vram_allocated, vram_reserved)
        if hasattr(self, "live_epoch_metric"):
            self._update_live_training_metrics(
                step_int,
                event,
                train_loss,
                learning_rate,
                grad_norm,
                update_ratio,
                tokens_per_second,
                samples_per_second,
                vram_allocated,
                vram_reserved,
                gpu_memory,
                system_cpu,
                system_ram,
                data_workers,
            )

    def _update_training_health(
        self,
        step: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
    ) -> None:
        """Update the training health advisor from recent loss values.

        Args:
            step: Current optimizer step.
            train_loss: Latest training loss.
            val_loss: Latest validation loss.
        """

        self.training_health_points.append((step, train_loss, val_loss))
        self.training_health_points = self.training_health_points[-12:]
        latest_train = next((item[1] for item in reversed(self.training_health_points) if item[1] is not None), None)
        latest_val = next((item[2] for item in reversed(self.training_health_points) if item[2] is not None), None)
        val_points = [(item[0], item[2]) for item in self.training_health_points if item[2] is not None]
        if latest_train is None and latest_val is None:
            label = "Health: collecting"
            tip = "Waiting for train and validation loss."
        elif latest_train is not None and latest_val is not None and latest_train < 0.2 and latest_val > max(2.0, latest_train * 8.0):
            label = "Health: validation gap"
            tip = "Training loss is very low while validation loss is high. Check overfitting, validation split, tokenizer match, or eval settings."
        elif len(val_points) >= 3 and val_points[-1][1] > val_points[-2][1] > val_points[-3][1]:
            label = "Health: overfitting?"
            tip = "Validation loss has increased for three checks. Consider stopping, reducing epochs, or improving validation data."
        elif latest_train is not None and (latest_train > 20.0 or not math.isfinite(latest_train)):
            label = "Health: diverging"
            tip = "Training loss is unstable or extremely high. Lower learning rate and check gradients/data."
        elif latest_val is not None and latest_val > 10.0:
            label = "Health: high val loss"
            tip = "Validation loss is high. This may be early training, a difficult validation split, or a dataset/tokenizer mismatch."
        elif latest_train is not None and latest_val is not None and latest_val <= latest_train * 1.8:
            label = "Health: stable"
            tip = "Training and validation loss are reasonably close."
        else:
            label = "Health: watching"
            tip = "Collecting more loss points before making a stronger diagnosis."
        self.training_health_metric.setText(label)
        self._tip(self.training_health_metric, tip)

    @staticmethod
    def _finite_metric(value: Any) -> Optional[float]:
        """Return a finite metric value or ``None``.

        Args:
            value: Raw metric value.

        Returns:
            Finite float, or ``None`` when invalid.
        """

        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def _update_live_training_metrics(
        self,
        step: int,
        event: dict[str, Any],
        train_loss: Optional[float],
        learning_rate: Optional[float],
        grad_norm: Optional[float],
        update_ratio: Optional[float],
        tokens_per_second: Optional[float],
        samples_per_second: Optional[float],
        vram_allocated: Optional[float],
        vram_reserved: Optional[float],
        gpu_memory: Optional[float],
        system_cpu: Optional[float],
        system_ram: Optional[float],
        data_workers: Optional[int],
    ) -> None:
        """Update live tracker widgets from one training progress event.

        Args:
            step: Current optimizer step.
            event: Progress event emitted by training.
            train_loss: Latest training loss.
            learning_rate: Current learning rate.
            grad_norm: Current gradient norm.
            update_ratio: Current parameter update ratio.
            tokens_per_second: Current token throughput.
            samples_per_second: Current sample throughput.
            vram_allocated: Current CUDA allocated memory in GB.
            vram_reserved: Current CUDA reserved memory in GB.
            gpu_memory: Current GPU memory pressure percentage.
            system_cpu: Current system CPU utilization percentage.
            system_ram: Current system RAM utilization percentage.
            data_workers: CPU data-loader worker count.
        """

        total_steps = event.get("total_steps")
        if "epoch" in event and "total_epochs" in event:
            self.live_epoch_metric.setText(f"Epoch: {event['epoch']}/{event['total_epochs']}")
        if total_steps:
            self.live_step_metric.setText(f"Step: {step:,}/{int(total_steps):,}")
            data_percent = min(100.0, max(0.0, (step / max(1, int(total_steps))) * 100.0))
            self.live_data_metric.setText(f"Data: {data_percent:.1f}%")
            self.live_progress.setValue(int(data_percent))
        else:
            self.live_step_metric.setText(f"Step: {step:,}")
        if tokens_per_second is not None:
            self.live_tokens_metric.setText(f"Tokens/sec: {float(tokens_per_second):,.0f}")
        if train_loss is not None:
            self.live_loss_metric.setText(f"Loss: {float(train_loss):.4f}")
        if learning_rate is not None:
            self.live_lr_metric.setText(f"LR: {float(learning_rate):.2e}")
        sample_text = str(event.get("sample_text") or "").strip()
        if sample_text:
            self.live_sample_text.setText(f"Training text: {self._compact_preview_text(sample_text, 220)}")
        self.live_layer_status.setText(f"▣ Layers: {self.n_layer.value()}")
        self.live_head_status.setText(f"◎ Heads: {self.n_head.value()}")
        self.live_hidden_status.setText(f"▤ Hidden size: {self.n_embd.value()}")
        self.live_batch_status.setText(f"▥ Batch size: {self.batch_size.value()}")
        self.live_context_status.setText(f"▢ Context: {self.train_context_length.value()}")
        self.live_device_status.setText(f"Device: {self.device.currentText()}")
        self.live_worker_status.setText(f"CPU workers: {data_workers if data_workers is not None else self.data_loader_workers.value()}")
        self._set_meter(self.live_cpu_bar, "CPU", system_cpu if system_cpu is not None else self._system_cpu_value())
        self._set_meter(self.live_gpu_bar, "GPU memory", gpu_memory)
        if vram_allocated is not None or vram_reserved is not None:
            allocated = float(vram_allocated or 0.0)
            reserved = float(vram_reserved or 0.0)
            reserved_percent = None
            if self.device.currentText().startswith("cuda") and torch.cuda.is_available():
                try:
                    _, total_vram = torch.cuda.mem_get_info()
                    reserved_percent = min(100.0, 100.0 * reserved * (1024 ** 3) / max(total_vram, 1))
                except Exception:
                    reserved_percent = None
            self._set_meter(self.live_vram_bar, "VRAM reserved", reserved_percent)
            self.live_vram_label.setText(f"VRAM reserved: {reserved:.2f} GB ({allocated:.2f} GB active)")
        self._set_meter(self.live_ram_bar, "System RAM", system_ram if system_ram is not None else self._system_ram_value())
        latest_loss = float(train_loss) if train_loss is not None else None
        self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), step, latest_loss)
        self.live_prediction_chart.update_distribution(step, latest_loss)
        self.live_attention_chart.update_heatmap(step, grad_norm)
        self.live_activation_chart.update_histogram(step, tokens_per_second)
        self.live_gradient_chart.update_flow(self.n_layer.value(), grad_norm, step)

    def _system_ram_value(self) -> Optional[float]:
        """Read system RAM utilization for live telemetry.

        Returns:
            System RAM percentage, or None when unavailable.
        """

        if psutil is None:
            return None
        return float(psutil.virtual_memory().percent)

    def _system_cpu_value(self) -> Optional[float]:
        """Read system CPU utilization for live telemetry.

        Returns:
            System CPU percentage, or None when unavailable.
        """

        if psutil is None:
            return None
        return float(psutil.cpu_percent(interval=None))

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration for compact UI display.

        Args:
            seconds: Duration in seconds.

        Returns:
            Human-readable compact duration.
        """

        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def _apply_chat_delta(self, event: dict[str, Any]) -> None:
        """Apply one streamed chat chunk to the rendered conversation.

        Args:
            event: Chat stream progress event.
        """

        self.chat_stream_reply += str(event.get("content", ""))
        should_follow = self._is_chat_near_bottom()
        self._render_chat_markdown(self.chat_stream_reply)
        if should_follow:
            self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum())
        self._set_chat_stats(
            float(event.get("elapsed_seconds", 0.0)),
            int(event.get("token_count", 0)),
            float(event.get("tokens_per_second", 0.0)),
        )

    def _drain_progress_queue(self) -> None:
        """Drain queued worker progress events on the UI thread."""

        if self.progress_queue is None or self.active_log is None or self.active_progress_bar is None:
            return
        drained = 0
        last_percent = None
        while drained < 12:
            try:
                event = self.progress_queue.get_nowait()
            except Empty:
                break
            notification_event = event
            if isinstance(event, dict) and event.get("percent") is not None:
                last_percent = event.get("percent")
                event = {**event, "percent": None}
            self._handle_progress(event, self.active_log, self.active_progress_bar)
            if isinstance(notification_event, dict):
                self._notify_progress(notification_event)
            drained += 1
        if last_percent is not None:
            self.active_progress_bar.setValue(max(0, min(100, int(last_percent))))

    def _thread_finished(self) -> None:
        """Clean up thread bookkeeping after a worker finishes."""

        LOGGER.info("Background task thread finished")
        self._drain_progress_queue()
        if self.progress_timer.isActive():
            self.progress_timer.stop()
        self.thread = None
        self.worker = None
        self.stop_event = None
        self.progress_queue = None
        self.active_log = None
        self.active_progress_bar = None
        if self.active_stop_button is not None:
            self.active_stop_button.setEnabled(False)
        self.active_stop_button = None
        if self.active_button is not None:
            self._clear_button_busy()
        self.active_task_kind = ""

    def _task_failed(self, message: str, log: QTextEdit, progress_bar: QProgressBar) -> None:
        """Handle background task failure.

        Args:
            message: Error message.
            log: Log widget to append to.
            progress_bar: Progress bar to reset.
        """

        stopped_by_user = "stopped by user" in message.lower()
        if stopped_by_user:
            LOGGER.info("Background task stopped by user: %s", message)
        else:
            LOGGER.error("Background task error: %s", message)
        log.append(f"Stopped: {message}" if stopped_by_user else f"Error: {message}")
        self._notify_failure("Task stopped" if stopped_by_user else "Task failed", message)
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        if stopped_by_user:
            self.project_state.setText("Stopped")
        self._clear_button_busy()

    def _set_button_busy(self, button: QPushButton, text: str) -> None:
        """Disable a button and start its spinner text.

        Args:
            button: Button to mark busy.
            text: Busy label.
        """

        self.active_button = button
        self.active_button_text = text
        self.active_button_restore_text = button.text()
        self.spinner_index = 0
        button.setEnabled(False)
        button.setText(f"| {text}")
        self.spinner_timer.start(150)

    def _clear_button_busy(self, final_text: Optional[str] = None) -> None:
        """Restore the active busy button.

        Args:
            final_text: Optional final button text.
        """

        if self.spinner_timer.isActive():
            self.spinner_timer.stop()
        if self.active_button:
            self.active_button.setEnabled(True)
            self.active_button.setText(final_text or self.active_button_restore_text)
        if self.active_stop_button:
            self.active_stop_button.setEnabled(False)
        self.active_button = None
        self.active_button_text = ""
        self.active_button_restore_text = ""

    def _tick_spinner(self) -> None:
        """Advance the active button spinner frame."""

        if not self.active_button:
            return
        frames = "|/-\\"
        self.spinner_index = (self.spinner_index + 1) % len(frames)
        self.active_button.setText(f"{frames[self.spinner_index]} {self.active_button_text}")

    def _dataset_config_from_ui(self) -> DatasetConfig:
        """Collect dataset options from the current UI controls.

        Returns:
            Dataset preparation configuration.
        """

        conversation_paths: list[Path] = []
        instruction_paths: list[Path] = []
        dataset_stage = self._dataset_stage_value()
        return DatasetConfig(
            input_dir=Path(self.input_dir.text()),
            output_dir=Path(self.dataset_dir.text()),
            vocab_size=None if self.auto_vocab.isChecked() else self.manual_vocab_size.value(),
            conversation_datasets=self._selected_conversation_datasets(),
            conversation_sample_limit=self.conversation_sample_limit.value(),
            conversation_dataset_path=conversation_paths[0] if conversation_paths else None,
            instruction_dataset_path=instruction_paths[0] if instruction_paths else None,
            conversation_dataset_paths=conversation_paths,
            instruction_dataset_paths=instruction_paths,
            default_data_paths=self._selected_default_data_paths_for_stage(dataset_stage),
            mixture_weights=self._mixture_weights_from_ui(),
            min_frequency=self.min_frequency.value(),
            context_length=self.context_length.value(),
            validation_split=self.validation_split.value(),
            lowercase=self.lowercase.isChecked(),
            max_workers=self.max_workers.value(),
            code_training_mode=self.code_training_mode.isChecked(),
            include_prose=self.include_prose.isChecked(),
            include_source_code=self.include_source_code.isChecked(),
            extract_code_blocks=self.extract_code_blocks.isChecked(),
            preserve_indentation=self.preserve_indentation.isChecked(),
            generate_instruction_samples=self.instruction_samples.isChecked(),
            reasoning_sample_mode=self._reasoning_sample_mode_value(),
            prepare_mode=self._prepare_mode_value(),
            tokenizer_strategy=self._tokenizer_strategy_value(),
            tokenizer_path=Path(self.tokenizer_path.text()) if self.tokenizer_path.text().strip() else None,
            dataset_stage=dataset_stage,
        )

    def _selected_default_data_paths_for_stage(self, stage: str) -> list[Path]:
        """Return selected bundled files that match the dataset purpose.

        Args:
            stage: Dataset preparation stage.

        Returns:
            Selected paths suitable for the requested stage.
        """

        allowed_stage_files = []
        root = self.blueprint_data_root
        for path in self._selected_default_data_paths():
            file_stage = default_data_stage(path, root)
            if file_stage == "base" or file_stage == stage:
                allowed_stage_files.append(path)
        return allowed_stage_files

    @staticmethod
    def _split_path_list(text: str) -> list[Path]:
        """Split a semicolon-delimited path field.

        Args:
            text: Raw path field text.

        Returns:
            Parsed paths.
        """

        return [Path(item.strip().strip('"')) for item in text.split(";") if item.strip()]

    def check_project_health(self) -> None:
        """Run a project health check in the background."""

        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self.dataset_log.append("Checking project health...")
        self.project_state.setText("Checking health")
        self._run_task(
            check_project_health,
            (
                Path(self.input_dir.text()),
                Path(self.dataset_dir.text()),
                Path(self.model_dir.text()),
                Path(self.export_dir.text()),
                Path(self.gguf_path.text()) if self.gguf_path.text().strip() else None,
                Path(self.llama_cpp_dir.text()) if self.llama_cpp_dir.text().strip() else None,
                self.device.currentText(),
            ),
            self._health_check_finished,
            self.dataset_log,
            self.dataset_progress,
            with_progress=True,
            button=self.health_check_button,
            stop_button=self.stop_dataset_button,
            busy_text="Checking Health",
        )

    @Slot(object)
    def _health_check_finished(self, result: Any) -> None:
        """Display project health check results.

        Args:
            result: Project health result.
        """

        self.dataset_progress.setValue(100)
        self.dataset_log.append("")
        self.dataset_log.append(f"Project health: {result.status.upper()} ({result.summary})")
        for check in result.checks:
            marker = {"ok": "OK", "warning": "WARN", "error": "ERROR"}.get(check.get("status"), "INFO")
            self.dataset_log.append(f"[{marker}] {check.get('name')}: {check.get('detail')}")
        self.project_state.setText("Health checked")
        self._clear_button_busy("Check Health")

    def preview_dataset(self) -> None:
        """Run a dataset preview and quality scan in the background."""

        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self.dataset_log.append("Previewing dataset...")
        self.project_state.setText("Previewing dataset")
        self._run_task(
            scan_dataset_preview,
            (self._dataset_config_from_ui(),),
            self._dataset_preview_finished,
            self.dataset_log,
            self.dataset_progress,
            with_progress=True,
            button=self.preview_dataset_button,
            stop_button=self.stop_dataset_button,
            busy_text="Previewing Dataset",
        )

    @Slot(object)
    def _dataset_preview_finished(self, result: Any) -> None:
        """Display dataset preview and quality scan results.

        Args:
            result: Dataset preview result.
        """

        self.dataset_progress.setValue(100)
        suffix_text = ", ".join(f"{suffix}: {count}" for suffix, count in
                                result.suffix_counts.items()) or "none"
        self.dataset_log.append("")
        self.dataset_log.append(
            f"Source files: {result.source_file_count:,}; size: {result.total_bytes / (1024 * 1024):.2f} MB")
        self.dataset_log.append(f"File types: {suffix_text}")
        self.dataset_log.append(
            f"Prepared dataset artifacts: {'found' if result.prepared else 'not complete'}")
        self.dataset_log.append(
            f"Duplicate scan: {result.duplicate_count:,} file entries in {len(result.duplicate_groups):,} likely group(s).")
        self.dataset_log.append(
            f"Bad extraction scan: {result.bad_extraction_count:,} suspicious file(s).")
        self.dataset_log.append(
            f"Code/prose balance: {result.balance_label} ({result.code_preview_count:,}/{result.prose_preview_count:,}).")
        self.dataset_log.append(
            f"Training readiness: {result.readiness_label} ({result.readiness_score}/100).")
        for reason in result.readiness_reasons[:8]:
            self.dataset_log.append(f"- {reason}")
        self.dataset_quality_duplicates.setText(
            f"Duplicates: {result.duplicate_count:,}")
        self.dataset_quality_extraction.setText(
            f"Extraction: {result.bad_extraction_count:,} flagged")
        self.dataset_quality_balance.setText(
            f"Balance: {result.balance_label}")
        self.dataset_quality_readiness.setText(
            f"Readiness: {result.readiness_label} {result.readiness_score}/100")
        if result.summary:
            self._update_dataset_quality_report(result.summary)
            # dataset_quality_duplicates is intentionally left alone here:
            # _update_dataset_quality_report() just set it to the block-level
            # duplication percentage from the prepared corpus (the more useful,
            # actionable metric). Re-setting it to result.duplicate_count (a
            # raw duplicate *file* count from the earlier preview scan) would
            # silently discard that and always show the old metric instead.
            self.dataset_quality_extraction.setText(
                f"Extraction: {result.bad_extraction_count:,} flagged")
            self.dataset_quality_balance.setText(
                f"Balance: {result.balance_label}")
            self.dataset_quality_readiness.setText(
                f"Readiness: {result.readiness_label} {result.readiness_score}/100")
            tokens = int(result.summary.get("token_count", 0) or 0)
            vocab = int(result.summary.get("tokenizer_vocab_size", 0) or 0)
            self.dataset_log.append(
                f"Prepared summary: {tokens:,} tokens, vocab {vocab:,}.")
        else:
            self.dataset_quality_samples.setText(
                f"Preview: {len(result.sample_previews):,} shown")
            self.dataset_quality_tokens.setText("Tokens: not prepared")
            self.dataset_quality_windows.setText("Windows: not prepared")
            self.dataset_quality_vocab.setText("Vocab: not prepared")
            self.dataset_quality_code.setText(
                f"Code/prose: {result.code_preview_count:,}/{result.prose_preview_count:,}")
            self.dataset_quality_cache.setText(
                f"Files: {result.source_file_count:,} source")
        if result.duplicate_groups:
            self.dataset_log.append("")
            self.dataset_log.append("Likely duplicates:")
            for group in result.duplicate_groups[:8]:
                self.dataset_log.append(
                    f"- {group.get('type')}: {group.get('count')} file(s)")
                for path in group.get("files", [])[:4]:
                    self.dataset_log.append(f"    {Path(path).name}")
        if result.bad_extraction_files:
            self.dataset_log.append("")
            self.dataset_log.append("Suspicious extraction files:")
            for item in result.bad_extraction_files[:12]:
                self.dataset_log.append(
                    f"- {Path(item.get('path', '')).name}: {item.get('reasons')}")
        suggestions: list[str] = []
        if result.duplicate_groups:
            suggestions.append(
                "Remove or move duplicate files before preparing the final dataset.")
        if result.bad_extraction_files:
            suggestions.append(
                "Replace flagged PDFs with text/source versions, or remove files with bad extraction.")
        if result.balance_label == "Prose heavy" and self.code_training_mode.isChecked():
            suggestions.append(
                "Add real source-code folders or enable source-file inclusion for a stronger coding model.")
        if result.balance_label == "Code heavy":
            suggestions.append(
                "Add README/tutorial/prose explanations if you want the model to explain code well.")
        if result.readiness_label in {"Needs cleanup", "Not ready"}:
            suggestions.append(
                "Run Preview Dataset again after cleanup and only train once readiness improves.")
        if hasattr(self, "dataset_advisor"):
            if suggestions:
                self.dataset_advisor.setPlainText(
                    "\n".join(f"- {suggestion}" for suggestion in suggestions))
            else:
                self.dataset_advisor.setPlainText(
                    "No immediate cleanup suggestions. Dataset looks acceptable for the current preview.")
        if suggestions:
            self.dataset_log.append("")
            self.dataset_log.append("Cleanup suggestions:")
            for suggestion in suggestions:
                self.dataset_log.append(f"- {suggestion}")
        if result.issues:
            self.dataset_quality_warning.setText(
                f"Warnings: {len(result.issues)}")
            self.dataset_log.append("")
            self.dataset_log.append("Quality notes:")
            for issue in result.issues[:12]:
                self.dataset_log.append(f"- {issue}")
        else:
            self.dataset_quality_warning.setText("Warnings: none")
        if result.sample_previews:
            self.dataset_log.append("")
            self.dataset_log.append("Preview samples:")
            for index, sample in enumerate(result.sample_previews, start=1):
                label = sample.get("language") or sample.get("kind") or "text"
                self.dataset_log.append(
                    f"\n[{index}] {Path(sample.get('path', '')).name} ({label}, {sample.get('characters')} chars)")
                self.dataset_log.append(
                    sample.get("preview", "").replace("\n", "\n    ")[:1400])
        self.project_state.setText("Dataset previewed")
        self._clear_button_busy("Preview Dataset")

    def prepare_dataset(self) -> None:
        """Collect dataset options and start dataset preparation."""

        config = self._dataset_config_from_ui()
        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self._reset_dataset_quality_report()
        self.dataset_log.append("Preparing dataset...")
        self.dataset_log.append(f"App log file: {self.log_file_path}")
        self.dataset_log.append(f"Dataset purpose: {dataset_stage_label(config.dataset_stage)}")
        if config.conversation_dataset_paths:
            self.dataset_log.append(f"Local conversation JSON/JSONL: {len(config.conversation_dataset_paths)} path(s)")
            LOGGER.info("Local conversation JSON/JSONL datasets: %s", "; ".join(str(path) for path in config.conversation_dataset_paths))
        if config.instruction_dataset_paths:
            self.dataset_log.append(f"Local instruction JSON/JSONL: {len(config.instruction_dataset_paths)} path(s)")
            LOGGER.info("Local instruction JSON/JSONL datasets: %s", "; ".join(str(path) for path in config.instruction_dataset_paths))
        if config.default_data_paths:
            self.dataset_log.append(f"Bundled default data: {len(config.default_data_paths)} file(s)")
            LOGGER.info("Bundled default data files: %s", "; ".join(str(path) for path in config.default_data_paths))
        if self.include_conversation_datasets.isChecked():
            selected_labels = [
                action.text()
                for action in getattr(self, "conversation_dataset_actions", {}).values()
                if action.isChecked() and action.isVisible()
            ]
            if selected_labels:
                hf_cache = config.output_dir / "cache" / "huggingface"
                self.dataset_log.append(f"Online training datasets: {', '.join(selected_labels)}")
                self.dataset_log.append(f"Downloading/loading online data at: {hf_cache}")
                LOGGER.info("Online training datasets: %s", ", ".join(selected_labels))
                LOGGER.info("Downloading/loading online data at: %s", hf_cache)
            else:
                self.dataset_log.append("Online training datasets are enabled, but no dataset is selected for this purpose.")
                LOGGER.warning("Online training datasets enabled, but no dataset is selected")
        else:
            self.dataset_log.append("Online training datasets: off. Local source files only.")
            LOGGER.info("Online training datasets: off. Local source files only.")
            checked_count = sum(
                1
                for action in getattr(self, "conversation_dataset_actions", {}).values()
                if action.isChecked()
            )
            if checked_count:
                self.dataset_log.append("Checked online dataset choices are ignored until the master checkbox is enabled.")
                LOGGER.info("Checked online dataset choices are ignored until the master checkbox is enabled")
        LOGGER.info(
            "Preparing dataset: input=%s output=%s stage=%s online_datasets=%s conversation_json=%s instruction_json=%s",
            config.input_dir,
            config.output_dir,
            config.dataset_stage,
            ",".join(config.conversation_datasets) or "off",
            ";".join(str(path) for path in config.conversation_dataset_paths) or "off",
            ";".join(str(path) for path in config.instruction_dataset_paths) or "off",
        )
        self.project_state.setText("Preparing dataset")
        self.dataset_status.setText("Dataset: preparing")
        self.auto_vocab_label.setText("Calculating...")
        self._run_task(
            build_dataset,
            (config,),
            self._dataset_finished,
            self.dataset_log,
            self.dataset_progress,
            with_progress=True,
            button=self.prepare_button,
            stop_button=self.stop_dataset_button,
            busy_text="Preparing Dataset",
            task_kind="dataset",
            isolate_process=True,
        )

    @Slot(object)
    def _dataset_finished(self, result: Any) -> None:
        """Update UI after dataset preparation finishes.

        Args:
            result: Dataset build result.
        """

        self.dataset_progress.setValue(100)
        self.auto_vocab_label.setText(f"{result.vocab_size:,}")

        LOGGER.info(
            "Dataset prepared: documents=%s tokens=%s vocab=%s code=%s prose=%s conversation=%s output=%s",
            result.document_count,
            result.token_count,
            result.vocab_size,
            result.code_sample_count,
            result.prose_sample_count,
            getattr(result, "conversation_sample_count", 0),
            result.output_dir,
        )

        self.dataset_log.append(
            f"Prepared {result.document_count} documents, "
            f"{result.character_count:,} characters, "
            f"{result.token_count:,} tokens, "
            f"vocab {result.vocab_size:,}."
        )

        if getattr(result, "train_window_count", 0) or getattr(result,
                                                               "val_window_count",
                                                               0):
            self.dataset_log.append(
                f"Training windows: {result.train_window_count:,}; "
                f"validation windows: {result.val_window_count:,}."
            )

        self.dataset_log.append(
            f"Cache summary: reused {result.cached_file_count:,} file(s), "
            f"processed {result.processed_file_count:,} file(s)."
        )

        if getattr(result, "dataset_version_id", ""):
            self.dataset_log.append(
                f"Dataset version: {result.dataset_version_id}"
            )

        if result.warning:
            self.dataset_log.append(f"Recommendation: {result.warning}")

        self._update_dataset_quality_report(
            {
                "document_count": result.document_count,
                "token_count": result.token_count,
                "train_window_count": getattr(result, "train_window_count", 0),
                "val_window_count": getattr(result, "val_window_count", 0),
                "character_count": result.character_count,
                "tokenizer_vocab_size": result.vocab_size,
                "code_sample_count": result.code_sample_count,
                "prose_sample_count": result.prose_sample_count,
                "conversation_sample_count": getattr(result,
                                                     "conversation_sample_count",
                                                     0),
                "cached_file_count": result.cached_file_count,
                "processed_file_count": result.processed_file_count,
                "skipped_file_count": result.skipped_file_count,
                "failed_file_count": result.failed_file_count,
                "warning": result.warning,
                "sequence_token_stats": getattr(result, "sequence_token_stats",
                                                {}),
                "duplicate_block_count": getattr(result,
                                                 "duplicate_block_count", 0),
                "unique_block_count": getattr(result, "unique_block_count", 0),
                "corpus_block_count": getattr(result, "corpus_block_count", 0),
                "duplicate_block_ratio": getattr(result,
                                                 "duplicate_block_ratio", 0.0),
                "unique_block_ratio": getattr(result, "unique_block_ratio",
                                              1.0),
            }
        )

        self.train_data_dir.setText(str(result.output_dir))
        self.project_state.setText("Dataset ready")

        self.dataset_status.setText(
            f"Dataset: {result.document_count} files, {result.token_count:,} tokens"
        )

        if result.code_sample_count:
            self.dataset_status.setText(
                f"Dataset: {result.code_sample_count:,} code, "
                f"{result.prose_sample_count:,} prose, "
                f"{result.token_count:,} tokens"
            )

        self.refresh_model_estimate()
        self.refresh_fine_tune_workflow()

        self._notify_complete(
            "dataset",
            "Dataset preparation complete",
            [
                f"Output: {result.output_dir}",
                f"Documents: {result.document_count:,}",
                f"Characters: {result.character_count:,}",
                f"Tokens: {result.token_count:,}",
                f"Vocabulary: {result.vocab_size:,}",
                (
                    "Windows: "
                    f"{getattr(result, 'train_window_count', 0):,} training, "
                    f"{getattr(result, 'val_window_count', 0):,} validation"
                ),
                (
                    "Content mix: "
                    f"{result.code_sample_count:,} code, "
                    f"{result.prose_sample_count:,} prose, "
                    f"{getattr(result, 'conversation_sample_count', 0):,} conversation"
                ),
                (
                    "Files: "
                    f"{result.processed_file_count:,} processed, "
                    f"{result.cached_file_count:,} cached, "
                    f"{result.skipped_file_count:,} skipped, "
                    f"{result.failed_file_count:,} failed"
                ),
                f"Dataset version: {getattr(result, 'dataset_version_id', '') or '-'}",
                f"Health: {'warning - ' + result.warning if result.warning else 'ready'}",
            ],
        )

        self._clear_button_busy("DataSet Prepared")

    def _prepare_mode_value(self) -> str:
        """Return the selected dataset preparation mode.

        Returns:
            Internal mode value.
        """

        label = self.prepare_mode.currentText()
        if label == "Full rebuild":
            return "full_rebuild"
        if label == "Force reprocess":
            return "force_reprocess"
        return "incremental"

    def _tokenizer_strategy_value(self) -> str:
        """Return the selected tokenizer strategy.

        Returns:
            Internal tokenizer strategy value.
        """

        label = self.tokenizer_strategy.currentText()
        if label == "Train new tokenizer":
            return "train_new"
        if label == "Reuse dataset tokenizer":
            return "reuse_dataset"
        if label == "Import tokenizer.json":
            return "import_tokenizer"
        return "auto"

    def _reasoning_sample_mode_value(self) -> str:
        """Return the selected reasoning sample mode.

        Returns:
            Internal reasoning sample mode.
        """

        label = self.reasoning_sample_mode.currentText()
        if label == "Detailed code reasoning":
            return "detailed"
        if label == "No reasoning wrapper":
            return "none"
        return "scaffold"

    def _dataset_stage_value(self) -> str:
        """Return the selected dataset preparation stage.

        Returns:
            Dataset stage identifier.
        """

        return {
            "Base pretraining": "base",
            "Instruction fine-tune": "instruction",
            "Conversation fine-tune": "conversation",
            "Code fine-tune": "code",
        }.get(self.dataset_stage.currentText(), "base")

    def _set_dataset_stage(self, stage: str) -> None:
        """Set the dataset stage combo from an internal stage value.

        Args:
            stage: Dataset stage identifier.
        """

        self._set_combo_by_data(
            self.dataset_stage,
            stage,
            {
                "base": "Base pretraining",
                "instruction": "Instruction fine-tune",
                "conversation": "Conversation fine-tune",
                "code": "Code fine-tune",
            },
        )
        self._update_online_dataset_stage_controls()

    def _update_online_dataset_stage_controls(self) -> None:
        """Show and enable online datasets for the selected training stage."""

        if not hasattr(self, "dataset_stage"):
            return
        stage = self._dataset_stage_value()
        allowed = set(dataset_ids_for_stage(stage))
        include_online = self.include_conversation_datasets.isChecked()
        for dataset_id, action in getattr(self, "conversation_dataset_actions", {}).items():
            visible = dataset_id in allowed
            action.setVisible(visible)
            action.setEnabled(include_online and visible)
            if not visible:
                action.setChecked(False)
        if hasattr(self, "conversation_dataset_button"):
            self.conversation_dataset_button.setEnabled(include_online)
        self.conversation_sample_limit.setEnabled(include_online)
        self._update_conversation_dataset_button_text()
        stage_name = dataset_stage_label(stage)
        if include_online:
            self.conversation_datasets_status.setText(f"{stage_name}: choose online datasets from the dropdown.")
        elif stage == "base":
            self.conversation_datasets_status.setText(
                "Base pretraining: all online source types are available; balance them with Dataset Mixture."
            )
        elif stage == "instruction":
            self.conversation_datasets_status.setText("Instruction fine-tune: Alpaca/Dolly/SlimOrca are available here. TinyStories is hidden.")
        elif stage == "conversation":
            self.conversation_datasets_status.setText("Conversation fine-tune: chat datasets are available here. TinyStories is hidden.")
        else:
            self.conversation_datasets_status.setText("Code fine-tune: CodeAlpaca/Magicoder/Evol CodeAlpaca are available here.")

    def _selected_conversation_datasets(self) -> list[str]:
        """Return selected built-in conversation dataset IDs.

        Returns:
            Selected dataset identifiers.
        """

        allowed = set(dataset_ids_for_stage(self._dataset_stage_value()))
        return [
            dataset_id
            for dataset_id, action in getattr(self, "conversation_dataset_actions", {}).items()
            if dataset_id in allowed and action.isChecked() and self.include_conversation_datasets.isChecked()
        ]

    def _set_selected_conversation_datasets(self, dataset_ids: list[str]) -> None:
        """Restore selected conversation dataset actions.

        Args:
            dataset_ids: Dataset IDs to select.
        """

        selected = set(dataset_ids)
        allowed = set(dataset_ids_for_stage(self._dataset_stage_value()))
        for dataset_id, action in getattr(self, "conversation_dataset_actions", {}).items():
            action.setChecked(dataset_id in selected and dataset_id in allowed)
            action.setEnabled(self.include_conversation_datasets.isChecked() and dataset_id in allowed)
        if hasattr(self, "conversation_sample_limit"):
            self.conversation_sample_limit.setEnabled(self.include_conversation_datasets.isChecked())
        self._update_conversation_dataset_button_text()
        if hasattr(self, "conversation_datasets_status"):
            self._update_online_dataset_stage_controls()

    def _update_conversation_dataset_button_text(self) -> None:
        """Refresh the compact online dataset selector label."""

        if not hasattr(self, "conversation_dataset_button"):
            return
        allowed = set(dataset_ids_for_stage(self._dataset_stage_value())) if hasattr(self, "dataset_stage") else set()
        selected_labels = [
            action.text()
            for dataset_id, action in getattr(self, "conversation_dataset_actions", {}).items()
            if dataset_id in allowed and action.isChecked()
        ]
        if not self.include_conversation_datasets.isChecked():
            self.conversation_dataset_button.setText("Online datasets off")
        elif not selected_labels:
            self.conversation_dataset_button.setText("Choose online datasets")
        elif len(selected_labels) == 1:
            self.conversation_dataset_button.setText(selected_labels[0])
        else:
            self.conversation_dataset_button.setText(f"{len(selected_labels)} online datasets selected")

    def configure_fine_tune_dataset_builder(self) -> None:
        """Configure the Ingest tab for the selected fine-tune dataset type."""

        stage_label = self.fine_tune_dataset_builder_stage.currentText()
        stage = {
            "Instruction fine-tune": "instruction",
            "Conversation fine-tune": "conversation",
            "Code fine-tune": "code",
        }.get(stage_label, "instruction")
        starter_datasets = {
            "instruction": ["alpaca_52k"],
            "conversation": ["dailydialog"],
            "code": ["codealpaca_20k"],
        }
        self._set_dataset_stage(stage)
        self.include_conversation_datasets.setChecked(True)
        self._set_selected_conversation_datasets(starter_datasets.get(stage, []))
        if stage == "code":
            self.code_training_mode.setChecked(True)
            self.include_source_code.setChecked(True)
            self.extract_code_blocks.setChecked(True)
            self.preserve_indentation.setChecked(True)
            self._set_mixture_weights({"code_technical": 100.0, "source_code": 100.0})
        elif stage == "conversation":
            self._set_mixture_weights({"conversation": 100.0, "social_emotional": 100.0})
        else:
            self._set_mixture_weights({"instruction": 100.0, "structured_qa": 70.0, "reasoning": 30.0})
        self._switch_page(0)
        self.dataset_log.append(f"Configured Ingest for {dataset_stage_label(stage)}. Import the base tokenizer before preparing.")
        self.project_state.setText(f"Configured {dataset_stage_label(stage)} data")

    def _dataset_plan_from_ui(self) -> dict[str, float]:
        """Return high-level dataset blueprint percentages.

        Returns:
            Mapping from dataset domain key to target percentage.
        """

        if not hasattr(self, "dataset_plan_spins"):
            return dataset_plan_defaults()
        return {key: float(widget.value()) for key, widget in self.dataset_plan_spins.items()}

    def _selected_default_data_paths(self) -> list[Path]:
        """Return bundled default data files selected in the Dataset Blueprint.

        Returns:
            Selected bundled data paths.
        """

        if not hasattr(self, "default_data_actions"):
            return [path for path, _category in iter_default_data_files()]
        return [
            Path(path)
            for path, item in self.default_data_actions.items()
            if item.checkState(0) == Qt.Checked
        ]

    def _set_selected_default_data_paths(self, paths: Optional[list[Any]]) -> None:
        """Restore bundled default data checkbox selections.

        Args:
            paths: Saved bundled data file paths. ``None`` means no
                preference was ever saved (a brand-new project), and every
                file is selected by default. An explicit empty list means
                the user deliberately deselected everything, and that
                choice is restored as-is rather than falling back to
                "select everything" -- previously the two cases were
                indistinguishable, so saving a project with nothing
                selected silently reset to everything selected on reload.
        """

        if not hasattr(self, "default_data_actions"):
            return
        if paths is None:
            selected = set(self.default_data_actions)
        else:
            selected = {str(Path(path)) for path in paths}
        self.default_data_tree_updating = True
        try:
            for path, item in self.default_data_actions.items():
                item.setCheckState(0, Qt.Checked if path in selected else Qt.Unchecked)
            self._refresh_default_data_category_states()
        finally:
            self.default_data_tree_updating = False

    def _handle_default_data_tree_changed(self, item: Any, column: int) -> None:
        """Handle category and file toggles in the bundled data tree.

        Args:
            item: Changed tree item.
            column: Changed column index.
        """

        if column != 0 or getattr(self, "default_data_tree_updating", False):
            return
        data = item.data(0, Qt.UserRole) or {}
        if data.get("kind") != "category":
            self.default_data_tree_updating = True
            try:
                self._refresh_default_data_category_states()
            finally:
                self.default_data_tree_updating = False
            if hasattr(self, "_mixture_weights_state"):
                delattr(self, "_mixture_weights_state")
            return
        state = item.checkState(0)
        if state == Qt.PartiallyChecked:
            return
        self.default_data_tree_updating = True
        try:
            for index in range(item.childCount()):
                item.child(index).setCheckState(0, state)
        finally:
            self.default_data_tree_updating = False
        if hasattr(self, "_mixture_weights_state"):
            delattr(self, "_mixture_weights_state")

    def _refresh_default_data_category_states(self) -> None:
        """Refresh category checkbox states from child file selections."""

        if not hasattr(self, "default_data_category_items"):
            return
        for category_item in self.default_data_category_items.values():
            checked = 0
            partial = False
            for index in range(category_item.childCount()):
                state = category_item.child(index).checkState(0)
                if state == Qt.Checked:
                    checked += 1
                elif state == Qt.PartiallyChecked:
                    partial = True
            if partial or 0 < checked < category_item.childCount():
                category_item.setCheckState(0, Qt.PartiallyChecked)
            elif checked == category_item.childCount() and category_item.childCount() > 0:
                category_item.setCheckState(0, Qt.Checked)
            else:
                category_item.setCheckState(0, Qt.Unchecked)

    def _set_dataset_plan(self, plan: dict[str, Any], preset: str = "Custom") -> None:
        """Restore high-level dataset blueprint controls.

        Args:
            plan: Saved dataset domain percentages.
            preset: Saved preset label.
        """

        if not hasattr(self, "dataset_plan_spins"):
            return
        self._restoring_dataset_plan = True
        try:
            values = {**dataset_plan_defaults(), **(plan or {})}
            for key, widget in self.dataset_plan_spins.items():
                widget.blockSignals(True)
                try:
                    widget.setValue(float(values.get(key, 0.0)))
                except (TypeError, ValueError):
                    widget.setValue(0.0)
                widget.blockSignals(False)
            self.dataset_plan_preset.blockSignals(True)
            if preset in DATASET_DOMAIN_PRESETS or preset == "Custom":
                self.dataset_plan_preset.setCurrentText(preset)
            else:
                self.dataset_plan_preset.setCurrentText("Custom")
            self.dataset_plan_preset.blockSignals(False)
        finally:
            self._restoring_dataset_plan = False
        self._update_dataset_plan_total()

    def _dataset_plan_mark_custom(self, *_args: Any) -> None:
        """Mark the dataset blueprint as custom after manual edits."""

        if getattr(self, "_restoring_dataset_plan", False):
            return
        if hasattr(self, "_mixture_weights_state"):
            delattr(self, "_mixture_weights_state")
        if hasattr(self, "dataset_plan_preset") and self.dataset_plan_preset.currentText() != "Custom":
            self.dataset_plan_preset.blockSignals(True)
            self.dataset_plan_preset.setCurrentText("Custom")
            self.dataset_plan_preset.blockSignals(False)

    def _update_dataset_plan_total(self) -> None:
        """Refresh the visible dataset blueprint total."""

        if not hasattr(self, "dataset_plan_total_label"):
            return
        total = sum(self._dataset_plan_from_ui().values())
        self.dataset_plan_total_label.setText(f"Total: {total:.1f}%")
        self.dataset_plan_total_label.setProperty("state", "ok" if abs(total - 100.0) <= 0.1 else "warning")
        self.dataset_plan_total_label.style().unpolish(self.dataset_plan_total_label)
        self.dataset_plan_total_label.style().polish(self.dataset_plan_total_label)

    def normalize_dataset_plan(self) -> None:
        """Scale high-level dataset blueprint values to 100 percent."""

        values = self._dataset_plan_from_ui()
        total = sum(values.values())
        if total <= 0:
            self._set_dataset_plan(dataset_plan_defaults(), "Balanced Tiny LLM")
            if hasattr(self, "_mixture_weights_state"):
                delattr(self, "_mixture_weights_state")
            return
        normalized = {key: value * 100.0 / total for key, value in values.items()}
        self._set_dataset_plan(normalized, "Custom")
        if hasattr(self, "_mixture_weights_state"):
            delattr(self, "_mixture_weights_state")
        LOGGER.info("Dataset blueprint normalized: %s", normalized)

    def apply_dataset_plan_preset(self, preset: str) -> None:
        """Apply a named dataset blueprint preset.

        Args:
            preset: Preset label from the Dataset Blueprint combo box.
        """

        if preset == "Custom" or preset not in DATASET_DOMAIN_PRESETS:
            return
        self._set_dataset_plan(DATASET_DOMAIN_PRESETS[preset], preset)
        if hasattr(self, "_mixture_weights_state"):
            delattr(self, "_mixture_weights_state")
        LOGGER.info("Dataset blueprint preset applied: %s", preset)

    def apply_dataset_plan_to_ingestion(self) -> None:
        """Apply the high-level dataset blueprint as active mixture weights."""

        plan = self._dataset_plan_from_ui()
        total = sum(plan.values())
        if total <= 0:
            plan = dataset_plan_defaults()
            total = sum(plan.values())
        normalized = {key: value * 100.0 / total for key, value in plan.items()}
        mixture = {
            **normalized,
            "local_prose": 0.0,
            "source_code": 0.0,
            "online_base": 0.0,
            "instruction": 0.0,
            "conversation": 0.0,
        }
        self._set_mixture_weights(mixture)
        if hasattr(self, "dataset_log"):
            self.dataset_log.append("Dataset blueprint applied to ingestion mixture.")
        self.project_state.setText("Blueprint applied")
        LOGGER.info("Dataset blueprint applied to ingestion mixture: plan=%s mixture=%s", normalized, mixture)

    def _mixture_weights_from_ui(self) -> dict[str, float]:
        """Return dataset mixture weights from the Ingest tab.

        Returns:
            Mapping from mixture source family to percentage.
        """

        if not hasattr(self, "mixture_local_prose"):
            if not hasattr(self, "_mixture_weights_state"):
                self.apply_dataset_plan_to_ingestion()
            return dict(getattr(self, "_mixture_weights_state", dataset_plan_defaults()))
        return {
            "local_prose": float(self.mixture_local_prose.value()),
            "source_code": float(self.mixture_source_code.value()),
            "online_base": float(self.mixture_online_base.value()),
            "instruction": float(self.mixture_instruction.value()),
            "conversation": float(self.mixture_conversation.value()),
        }

    def _set_mixture_weights(self, weights: dict[str, Any]) -> None:
        """Restore dataset mixture weights.

        Args:
            weights: Saved mixture weights by source family.
        """

        self._mixture_weights_state = dict(weights or {})
        if not hasattr(self, "mixture_local_prose"):
            return
        defaults = {
            **dataset_plan_defaults(),
            "local_prose": 50.0,
            "source_code": 30.0,
            "online_base": 20.0,
            "instruction": 0.0,
            "conversation": 0.0,
        }
        values = {**defaults, **(weights or {})}
        widgets = {
            "local_prose": self.mixture_local_prose,
            "source_code": self.mixture_source_code,
            "online_base": self.mixture_online_base,
            "instruction": self.mixture_instruction,
            "conversation": self.mixture_conversation,
        }
        for key, widget in widgets.items():
            try:
                widget.setValue(float(values.get(key, defaults[key])))
            except (TypeError, ValueError):
                widget.setValue(defaults[key])
        self._update_mixture_total()

    def _update_mixture_total(self) -> None:
        """Refresh the visible dataset mixture total."""

        if not hasattr(self, "mixture_total_label"):
            return
        total = sum(self._mixture_weights_from_ui().values())
        self.mixture_total_label.setText(f"Total: {total:.1f}%")
        self.mixture_total_label.setProperty("state", "ok" if abs(total - 100.0) <= 0.1 else "warning")
        self.mixture_total_label.style().unpolish(self.mixture_total_label)
        self.mixture_total_label.style().polish(self.mixture_total_label)

    def _normalize_mixture_weights(self) -> None:
        """Scale dataset mixture values so they total 100 percent."""

        weights = self._mixture_weights_from_ui()
        total = sum(weights.values())
        if total <= 0:
            self._set_mixture_weights({"local_prose": 100.0})
            return
        self._set_mixture_weights({key: value * 100.0 / total for key, value in weights.items()})

    def _training_launch_target_value(self) -> str:
        """Return whether training should launch locally or remotely.

        Returns:
            ``local`` or ``remote``.
        """

        if self.training_launch_target.currentText() == "RunPod cloud":
            return "runpod"
        return "remote" if self.training_launch_target.currentText() == "Remote workers" else "local"

    def _fine_tune_launch_target_value(self) -> str:
        """Return whether fine-tuning should launch locally or remotely.

        Returns:
            ``local`` or ``remote``.
        """

        if not hasattr(self, "fine_tune_launch_target"):
            return "local"
        if self.fine_tune_launch_target.currentText() == "RunPod cloud":
            return "runpod"
        return "remote" if self.fine_tune_launch_target.currentText() == "Remote workers" else "local"

    def _architecture_style_config(self) -> dict[str, Any]:
        """Return ModelConfig keyword arguments for the selected block style.

        Returns:
            Architecture style settings.
        """

        if self.architecture_style.currentText() == "Llama-like":
            return {
                "norm_type": "rmsnorm",
                "position_encoding": "rope",
                "mlp_type": "swiglu",
                "rope_theta": 10000.0,
            }
        return {
            "norm_type": "layernorm",
            "position_encoding": "learned",
            "mlp_type": "gelu",
            "rope_theta": 10000.0,
        }

    def _optimizer_value(self) -> str:
        """Return the selected optimizer identifier.

        Returns:
            Stable optimizer name used by the trainer.
        """

        return {
            "AdamW": "adamw",
            "Adam": "adam",
            "Lion": "lion",
            "Adafactor": "adafactor",
        }.get(self.optimizer_name.currentText(), "adamw")

    def _scheduler_value(self) -> str:
        """Return the selected scheduler identifier.

        Returns:
            Stable scheduler name used by the trainer.
        """

        return {
            "Warmup linear": "warmup_linear",
            "Cosine decay": "cosine",
            "Polynomial decay": "polynomial",
            "One-cycle": "one_cycle",
            "Constant": "constant",
        }.get(self.scheduler_name.currentText(), "warmup_linear")

    def _precision_value(self) -> str:
        """Return the selected numeric precision identifier.

        Returns:
            Stable precision name used by the trainer.
        """

        return {
            "FP16": "fp16",
            "BF16": "bf16",
            "FP32": "fp32",
        }.get(self.precision.currentText(), "fp16")

    def _fine_tune_output_path(self) -> Path:
        """Return the selected fine-tune output folder.

        Returns:
            Folder where fine-tuned artifacts should be written.
        """

        text = self.fine_tune_output_dir.text().strip() if hasattr(self, "fine_tune_output_dir") else ""
        if text:
            path = Path(text)
        elif self.current_project_file is not None:
            path = self.current_project_file.parent / "fine_tunes" / "latest"
        else:
            path = Path(self.model_dir.text()) / "fine_tuned"
        try:
            if path.resolve() == Path(self.model_dir.text()).resolve():
                path = Path(self.model_dir.text()) / "fine_tuned"
        except OSError:
            pass
        if hasattr(self, "fine_tune_output_dir"):
            self.fine_tune_output_dir.setText(str(path))
        return path

    def _refresh_fine_tune_default_output(self, *_args: Any) -> None:
        """Keep the fine-tune output folder stage-specific unless a custom folder was chosen."""

        if not hasattr(self, "fine_tune_output_dir") or self.current_project_file is None:
            return
        project_dir = self.current_project_file.parent
        fine_tunes_dir = project_dir / "fine_tunes"
        stage = self._training_stage_value()
        stage_folder = {
            "instruction": "instruction_latest",
            "conversation": "conversation_latest",
            "code": "code_latest",
            "domain": "domain_latest",
        }.get(stage, "fine_tune_latest")
        desired = fine_tunes_dir / stage_folder
        current_text = self.fine_tune_output_dir.text().strip()
        if not current_text:
            self.fine_tune_output_dir.setText(str(desired))
            return
        try:
            current = Path(current_text)
            current_resolved = current.resolve()
            fine_tunes_resolved = fine_tunes_dir.resolve()
        except OSError:
            return
        managed_names = {
            "latest",
            "fine_tune",
            "fine_tuned",
            "instruction",
            "conversation",
            "code",
            "domain",
            "instruction_latest",
            "conversation_latest",
            "code_latest",
            "domain_latest",
            "fine_tune_latest",
        }
        if current_resolved.parent == fine_tunes_resolved and current.name in managed_names:
            self.fine_tune_output_dir.setText(str(desired))

    def _prepare_fine_tune_run_folder(self, training_config: TrainingConfig) -> None:
        """Create fine-tune folders and snapshot the base checkpoint.

        Args:
            training_config: Fine-tune training configuration.
        """

        output_dir = Path(training_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        base_checkpoint = training_config.fine_tune_from_checkpoint
        if base_checkpoint is None:
            return
        base_checkpoint = Path(base_checkpoint)
        if not base_checkpoint.exists():
            return
        try:
            base_resolved = base_checkpoint.resolve()
            output_resolved = output_dir.resolve()
            if base_resolved == (output_resolved / base_checkpoint.name) or output_resolved in base_resolved.parents:
                raise ValueError(
                    "Fine-tune base checkpoint must be outside the selected fine-tune output folder. "
                    "Choose the original pretrained model checkpoint instead."
                )
        except RuntimeError as exc:
            raise ValueError(f"Could not validate fine-tune base checkpoint path: {exc}") from exc
        snapshot_dir = output_dir / "base_model"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        copied_checkpoint = snapshot_dir / base_checkpoint.name
        if not copied_checkpoint.exists() or copied_checkpoint.stat().st_size != base_checkpoint.stat().st_size:
            shutil.copy2(base_checkpoint, copied_checkpoint)
        base_parent = base_checkpoint.parent
        for file_name in ("tokenizer.json", "training_summary.json", "model_lineage.json"):
            source = base_parent / file_name
            if source.exists():
                target = snapshot_dir / file_name
                if not target.exists() or target.stat().st_size != source.stat().st_size:
                    shutil.copy2(source, target)
        manifest = {
            "base_checkpoint": str(base_checkpoint),
            "copied_checkpoint": str(copied_checkpoint),
            "fine_tune_output": str(output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (snapshot_dir / "base_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.fine_tune_log.append(f"Base model snapshot: {copied_checkpoint}")

    def _training_output_dir_for_mode(self, training_mode: Optional[str]) -> Path:
        """Return the output folder for a training mode.

        Args:
            training_mode: Training mode override.

        Returns:
            Base model or fine-tune output folder.
        """

        return self._fine_tune_output_path() if training_mode == "fine_tune" else Path(self.model_dir.text())

    def _training_mode_value(self) -> str:
        """Return the selected training mode identifier.

        Returns:
            Stable training mode used by the trainer.
        """

        return {
            "Pretrain from scratch": "pretrain",
            "Fine-tune checkpoint": "fine_tune",
            "Instruction fine-tune": "fine_tune",
            "Conversation fine-tune": "fine_tune",
            "Code fine-tune": "fine_tune",
        }.get(self.training_mode.currentText(), "pretrain")

    def _training_stage_value(self) -> str:
        """Return the higher-level training stage selected in the UI.

        Returns:
            Training stage identifier.
        """

        return {
            "Pretrain from scratch": "base",
            "Fine-tune checkpoint": "domain",
            "Instruction fine-tune": "instruction",
            "Conversation fine-tune": "conversation",
            "Code fine-tune": "code",
        }.get(self.training_mode.currentText(), "base")

    def _peft_method_value(self) -> str:
        """Return the selected PEFT method identifier.

        Returns:
            Stable PEFT method used by the trainer.
        """

        return {
            "Full fine-tune": "none",
            "LoRA adapters": "lora",
        }.get(self.peft_method.currentText(), "none")

    def _lora_target_value(self) -> str:
        """Return selected LoRA target groups.

        Returns:
            Comma-separated target group string.
        """

        return {
            "Attention projections": "attention",
            "MLP projections": "mlp",
            "Attention + MLP": "attention,mlp",
        }.get(self.lora_targets.currentText(), "attention")

    def _update_training_mode_controls(self) -> None:
        """Enable fine-tune controls only when fine-tuning is selected."""

        enabled = self._training_mode_value() == "fine_tune"
        lora_enabled = enabled and self._peft_method_value() == "lora"
        self.fine_tune_checkpoint.setEnabled(enabled)
        self.peft_method.setEnabled(enabled)
        self.fine_tune_check_button.setEnabled(enabled)
        self.lora_rank.setEnabled(lora_enabled)
        self.lora_alpha.setEnabled(lora_enabled)
        self.lora_dropout.setEnabled(lora_enabled)
        self.lora_targets.setEnabled(lora_enabled)
        self.refresh_fine_tune_workflow()

    def _current_dataset_summary(self) -> dict[str, Any]:
        """Read the active prepared dataset summary.

        Returns:
            Dataset summary dictionary, or an empty dictionary.
        """

        summary_path = Path(self.train_data_dir.text()) / "dataset_summary.json"
        if not summary_path.exists():
            summary_path = Path(self.dataset_dir.text()) / "dataset_summary.json"
        if not summary_path.exists():
            return {}
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            LOGGER.warning("Could not read dataset summary %s: %s", summary_path, exc)
            return {}

    def _fine_tune_dataset_stage_status(self) -> tuple[bool, str]:
        """Check whether the prepared dataset matches the fine-tune type.

        Returns:
            Tuple containing whether the workflow may proceed and a user-facing message.
        """

        expected_stage = self._training_stage_value()
        summary = self._current_dataset_summary()
        if not summary:
            return False, "Dataset: not prepared. Prepare the fine-tune dataset first."
        dataset_stage = str(summary.get("dataset_stage") or self._dataset_stage_value())
        tokens = int(summary.get("token_count", 0) or 0)
        vocab = int(summary.get("tokenizer_vocab_size", 0) or 0)
        stage_name = dataset_stage_label(dataset_stage) if dataset_stage in {"base", "instruction", "conversation", "code"} else dataset_stage
        details = f"{stage_name}, {tokens:,} tokens, vocab {vocab:,}"
        if expected_stage == "instruction" and dataset_stage != "instruction":
            return False, f"Dataset mismatch: selected Instruction fine-tune, but prepared dataset is {details}."
        if expected_stage == "conversation" and dataset_stage != "conversation":
            return False, f"Dataset mismatch: selected Conversation fine-tune, but prepared dataset is {details}."
        if expected_stage == "code" and dataset_stage != "code":
            return False, f"Dataset mismatch: selected Code fine-tune, but prepared dataset is {details}."
        if expected_stage == "domain" and dataset_stage == "base":
            return True, f"Dataset warning: {details}. Base datasets usually belong to pretraining; continue only for domain adaptation."
        return True, f"Dataset ready: {details}."

    def refresh_fine_tune_workflow(self) -> None:
        """Refresh fine-tune workflow guidance in the Fine-Tuning tab."""

        if not hasattr(self, "fine_tune_dataset_status"):
            return
        self._refresh_fine_tune_default_output()
        ok, message = self._fine_tune_dataset_stage_status()
        self.fine_tune_dataset_status.setText(message)
        self.fine_tune_dataset_status.setProperty("state", "ok" if ok else "warning")
        self.fine_tune_dataset_status.style().unpolish(self.fine_tune_dataset_status)
        self.fine_tune_dataset_status.style().polish(self.fine_tune_dataset_status)

    def apply_recommended_fine_tune_settings(self) -> None:
        """Apply conservative fine-tuning defaults for the selected workflow."""

        stage = self._training_stage_value()
        synced = self._sync_architecture_from_fine_tune_base()
        self._set_combo_text(self.peft_method, "LoRA adapters")
        self.lora_dropout.setValue(0.05)
        self._set_combo_text(self.lora_targets, "Attention projections")
        self.max_grad_norm.setValue(0.5)
        self.weight_decay.setValue(0.05)
        self._set_combo_by_data(self.scheduler_name, "cosine", {
            "warmup_linear": "Warmup linear",
            "cosine": "Cosine decay",
            "polynomial": "Polynomial decay",
            "one_cycle": "One-cycle",
            "constant": "Constant",
        })
        if stage == "conversation":
            self.lora_rank.setValue(16)
            self.lora_alpha.setValue(32.0)
            self.learning_rate.setValue(0.00003)
            self.epochs.setValue(max(1, min(self.epochs.value(), 2)))
        elif stage == "code":
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.lora_dropout.setValue(0.05)
            self.learning_rate.setValue(0.00005)
            self.max_grad_norm.setValue(0.5)
            self.epochs.setValue(max(1, min(self.epochs.value(), 3)))
        elif stage == "instruction":
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.learning_rate.setValue(0.00005)
            self.epochs.setValue(max(1, min(self.epochs.value(), 3)))
        else:
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.learning_rate.setValue(0.00005)
        self._update_training_mode_controls()
        message = "Recommended LoRA settings applied."
        if synced:
            message += "\nArchitecture was synced from the selected base checkpoint."
        message += "\nUse Check Fine-tune before starting so checkpoint and tokenizer compatibility are verified."
        self.fine_tune_preview.setText(message)

    def _sync_architecture_from_fine_tune_base(self) -> bool:
        """Sync architecture controls from the selected fine-tune base checkpoint.

        Returns:
            True when a checkpoint was read and architecture controls were updated.
        """

        if not hasattr(self, "fine_tune_checkpoint"):
            return False
        checkpoint_text = self.fine_tune_checkpoint.text().strip()
        if not checkpoint_text:
            return False
        checkpoint_path = Path(checkpoint_text)
        if not checkpoint_path.exists():
            return False
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            LOGGER.warning("Could not read fine-tune base checkpoint %s: %s", checkpoint_path, exc)
            return False
        model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
        if not isinstance(model_config, dict):
            return False
        mappings = {
            "embedding_size": self.n_embd,
            "head_count": self.n_head,
            "layer_count": self.n_layer,
            "context_length": self.context_length,
        }
        for key, widget in mappings.items():
            if key in model_config:
                try:
                    widget.setValue(int(model_config[key]))
                except (TypeError, ValueError):
                    LOGGER.warning("Invalid %s in checkpoint %s: %r", key, checkpoint_path, model_config[key])
        if "dropout" in model_config:
            try:
                self.dropout.setValue(float(model_config["dropout"]))
            except (TypeError, ValueError):
                LOGGER.warning("Invalid dropout in checkpoint %s: %r", checkpoint_path, model_config["dropout"])
        norm_type = str(model_config.get("norm_type", "layernorm")).lower()
        position_encoding = str(model_config.get("position_encoding", "learned")).lower()
        mlp_type = str(model_config.get("mlp_type", "gelu")).lower()
        if norm_type == "rmsnorm" or position_encoding == "rope" or mlp_type == "swiglu":
            self._set_combo_text(self.architecture_style, "Modern LLM")
        else:
            self._set_combo_text(self.architecture_style, "Classic GPT")
        attention_type = str(model_config.get("attention_type", "mha")).lower()
        self._set_combo_by_data(
            self.attention_type,
            attention_type,
            {
                "mha": "Multi-head",
                "mqa": "Multi-query",
                "gqa": "Grouped-query",
            },
        )
        if "kv_head_count" in model_config:
            try:
                self.kv_head_count.setValue(int(model_config["kv_head_count"]))
            except (TypeError, ValueError):
                LOGGER.warning("Invalid kv_head_count in checkpoint %s: %r", checkpoint_path, model_config["kv_head_count"])
        backend = str(model_config.get("attention_backend", "sdpa")).lower()
        self._set_combo_by_data(
            self.attention_backend,
            backend,
            {
                "sdpa": "SDPA / Flash when available",
                "eager": "PyTorch eager",
            },
        )
        if "attention_window" in model_config:
            try:
                self.attention_window.setValue(int(model_config["attention_window"]))
            except (TypeError, ValueError):
                LOGGER.warning("Invalid attention_window in checkpoint %s: %r", checkpoint_path, model_config["attention_window"])
        LOGGER.info("Fine-tune architecture synced from base checkpoint: %s", checkpoint_path)
        return True

    def _attention_type_value(self) -> str:
        """Return the selected attention layout identifier.

        Returns:
            Stable attention type used by the model.
        """

        return {
            "Multi-head": "mha",
            "Grouped-query": "gqa",
            "Multi-query": "mqa",
        }.get(self.attention_type.currentText(), "mha")

    def _attention_backend_value(self) -> str:
        """Return the selected attention backend identifier.

        Returns:
            Stable attention backend used by the model.
        """

        return {
            "SDPA / Flash when available": "sdpa",
            "Manual": "manual",
        }.get(self.attention_backend.currentText(), "sdpa")

    def apply_training_profile(self) -> None:
        """Apply the selected optimizer/scheduler profile."""

        profile = self.training_profile.currentText()
        if profile == "Low-memory":
            self._set_combo_text(self.optimizer_name, "Adafactor")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.0002)
            self.weight_decay.setValue(0.05)
            self.min_lr_ratio.setValue(0.05)
            self.max_grad_norm.setValue(1.0)
            self._set_combo_text(self.precision, "BF16" if torch.cuda.is_available() else "FP32")
            self._set_combo_text(self.attention_type, "Grouped-query")
            self.kv_head_count.setValue(max(1, self.n_head.value() // 2))
            self.activation_checkpointing.setChecked(True)
        elif profile == "Code fine-tune":
            self._set_combo_text(self.optimizer_name, "AdamW")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.00005)
            self.weight_decay.setValue(0.05)
            self.min_lr_ratio.setValue(0.1)
            self.max_grad_norm.setValue(0.5)
            self.dropout.setValue(0.05)
            self._set_combo_text(self.training_mode, "Fine-tune checkpoint")
            self._set_combo_text(self.peft_method, "LoRA adapters")
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.lora_dropout.setValue(0.05)
            self._set_combo_text(self.lora_targets, "Attention projections")
        elif profile == "Experimental Lion":
            self._set_combo_text(self.optimizer_name, "Lion")
            self._set_combo_text(self.scheduler_name, "One-cycle")
            self.learning_rate.setValue(0.0001)
            self.weight_decay.setValue(0.1)
            self.min_lr_ratio.setValue(0.01)
            self.max_grad_norm.setValue(1.0)
        else:
            self._set_combo_text(self.optimizer_name, "AdamW")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.0003)
            self.weight_decay.setValue(0.1)
            self.min_lr_ratio.setValue(0.1)
            self.max_grad_norm.setValue(1.0)
            self._set_combo_text(self.precision, "FP16")
        self._update_training_mode_controls()
        self.refresh_model_estimate()
        self.training_log.append(f"Applied training profile: {profile}")

    def _tokenizer_strategy_reuses(self) -> bool:
        """Return whether current tokenizer strategy ignores vocabulary controls.

        Returns:
            True when an existing tokenizer is selected directly.
        """

        return self.tokenizer_strategy.currentText() in {"Reuse dataset tokenizer", "Import tokenizer.json"}

    def _update_tokenizer_strategy_controls(self) -> None:
        """Enable only the tokenizer inputs relevant to the selected strategy."""

        imports_tokenizer = self.tokenizer_strategy.currentText() == "Import tokenizer.json"
        reuses_tokenizer = self._tokenizer_strategy_reuses()
        if hasattr(self, "tokenizer_path_row"):
            self.tokenizer_path_row.setEnabled(imports_tokenizer)
        self.tokenizer_path.setEnabled(imports_tokenizer)
        self.auto_vocab.setEnabled(not reuses_tokenizer)
        self.manual_vocab_size.setEnabled(not reuses_tokenizer and not self.auto_vocab.isChecked())
        self.min_frequency.setEnabled(not reuses_tokenizer)

    def _update_model_estimate_chips(
        self,
        estimate: dict[str, Any],
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        train_tokens: int = 0,
    ) -> None:
        """Update model and VRAM estimate chips.

        Args:
            estimate: Estimate dictionary from the training planning service.
            model_config: Model architecture used for the estimate.
            training_config: Training options used for the estimate.
            train_tokens: Number of available training tokens.
        """

        params = int(estimate.get("parameters", 0))
        checkpoint_bytes = float(estimate.get("checkpoint_bytes", 0))
        vram_bytes = float(estimate.get("vram_bytes", 0))
        self.model_size_metric.setText(f"Model: {params / 1_000_000:.2f}M, ckpt {format_bytes(checkpoint_bytes)}")
        self.vram_estimate_metric.setText(f"VRAM est: {format_bytes(vram_bytes)}")
        parameter_breakdown = estimate.get("parameter_breakdown", {}) or {}
        memory_breakdown = estimate.get("memory_breakdown", {}) or {}
        embedding_params = int(parameter_breakdown.get("token_embedding", 0)) + int(
            parameter_breakdown.get("position_embedding", 0)
        )
        attention_params = int(parameter_breakdown.get("attention", 0))
        mlp_params = int(parameter_breakdown.get("mlp", 0))
        norm_params = int(parameter_breakdown.get("norms", 0))
        self.parameter_breakdown_metric.setText(
            "Params: "
            f"emb {self._compact_number(embedding_params)}, "
            f"attn {self._compact_number(attention_params)}, "
            f"mlp {self._compact_number(mlp_params)}"
        )
        self._tip(
            self.parameter_breakdown_metric,
            (
                f"Embedding: {embedding_params:,}\n"
                f"Attention: {attention_params:,}\n"
                f"MLP: {mlp_params:,}\n"
                f"Norms/output: {norm_params:,}\n"
                f"Total: {params:,}"
            ),
        )
        weights = float(memory_breakdown.get("weights", 0))
        optimizer = float(memory_breakdown.get("optimizer", 0))
        activations = float(memory_breakdown.get("activations", 0))
        kv_cache = float(memory_breakdown.get("kv_cache", 0))
        self.memory_breakdown_metric.setText(
            f"Memory: w {format_bytes(weights)}, opt {format_bytes(optimizer)}, act {format_bytes(activations)}"
        )
        self._tip(
            self.memory_breakdown_metric,
            (
                f"Weights: {format_bytes(weights)}\n"
                f"Optimizer state: {format_bytes(optimizer)}\n"
                f"Activations: {format_bytes(activations)}\n"
                f"KV cache estimate: {format_bytes(kv_cache)}\n"
                f"Total training estimate: {format_bytes(vram_bytes)}"
            ),
        )
        self._update_architecture_advisor(estimate, model_config, training_config, train_tokens)

    def _update_architecture_advisor(
        self,
        estimate: dict[str, Any],
        model_config: Optional[ModelConfig],
        training_config: Optional[TrainingConfig],
        train_tokens: int,
    ) -> None:
        """Update the compact architecture advisor chip.

        Args:
            estimate: Estimate dictionary from the training planning service.
            model_config: Model architecture used for the estimate.
            training_config: Training options used for the estimate.
            train_tokens: Number of available training tokens.
        """

        params = max(int(estimate.get("parameters", 0) or 0), 1)
        tokens_per_param = float(train_tokens) / float(params) if train_tokens > 0 else 0.0
        vram_bytes = float(estimate.get("vram_bytes", 0) or 0)
        notes: list[str] = []
        if tokens_per_param <= 0:
            label = "Advisor: prepare data"
            notes.append("Prepare a dataset to compare token budget against model size.")
        elif tokens_per_param < 20:
            label = "Advisor: data-light"
            notes.append(
                f"Token budget is about {tokens_per_param:.1f} tokens per parameter. More data or fewer epochs may reduce overfitting."
            )
        elif tokens_per_param > 150:
            label = "Advisor: data-rich"
            notes.append(
                f"Token budget is about {tokens_per_param:.1f} tokens per parameter. The model may be small for this much data."
            )
        else:
            label = "Advisor: balanced"
            notes.append(f"Token budget is about {tokens_per_param:.1f} tokens per parameter.")
        if model_config is not None:
            if model_config.context_length >= 2048 and model_config.embedding_size <= 256:
                notes.append("Long context with a small embedding can be memory-heavy without adding much capacity.")
            if model_config.attention_type in {"grouped_query", "multi_query"}:
                notes.append("Grouped/multi-query attention reduces KV memory and is useful for longer contexts.")
            if model_config.mlp_type == "swiglu" and model_config.norm_type == "rmsnorm":
                notes.append("Llama-like blocks improve modern compatibility but must match checkpoints when resuming.")
        if training_config is not None and training_config.device == "cuda" and vram_bytes > 3.5 * 1024**3:
            notes.append("Estimated VRAM is high for 4 GB GPUs. Try lower batch, context, embedding, or layers.")
            if label == "Advisor: balanced":
                label = "Advisor: memory check"
        self.architecture_advisor_metric.setText(label)
        self._tip(self.architecture_advisor_metric, "\n".join(notes))

    @staticmethod
    def _compact_number(value: int) -> str:
        """Format a large count for tight metric chips.

        Args:
            value: Count to format.

        Returns:
            Compact display string.
        """

        magnitude = abs(value)
        if magnitude >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"
        if magnitude >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if magnitude >= 1_000:
            return f"{value / 1_000:.1f}K"
        return str(value)

    def _current_model_config(self, vocab_size: int = 1) -> ModelConfig:
        """Build a model config from the current AI tab settings.

        Args:
            vocab_size: Tokenizer vocabulary size to use.

        Returns:
            Current model configuration.
        """

        return ModelConfig(
            vocab_size=vocab_size,
            context_length=self.train_context_length.value(),
            embedding_size=self.n_embd.value(),
            head_count=self.n_head.value(),
            layer_count=self.n_layer.value(),
            dropout=self.dropout.value(),
            attention_type=self._attention_type_value(),
            kv_head_count=self.kv_head_count.value(),
            attention_backend=self._attention_backend_value(),
            attention_window=self.attention_window.value(),
            **self._architecture_style_config(),
        )

    def _current_training_config(
        self,
        resume_path: Optional[Path] = None,
        training_mode: Optional[str] = None,
    ) -> TrainingConfig:
        """Build a training config from the current AI tab settings.

        Args:
            resume_path: Optional specific checkpoint to resume from.
            training_mode: Optional explicit training mode override.

        Returns:
            Current training configuration.
        """

        return TrainingConfig(
            output_dir=self._training_output_dir_for_mode(training_mode),
            epochs=self.epochs.value(),
            batch_size=self.batch_size.value(),
            learning_rate=self.learning_rate.value(),
            weight_decay=self.weight_decay.value(),
            optimizer_name=self._optimizer_value(),
            scheduler_name=self._scheduler_value(),
            scheduler_min_lr_ratio=self.min_lr_ratio.value(),
            polynomial_power=self.polynomial_power.value(),
            gradient_accumulation=self.gradient_accumulation.value(),
            sample_stride=self.sample_stride.value(),
            warmup_steps=self.warmup_steps.value(),
            eval_interval=self.eval_interval.value(),
            max_eval_batches=self.max_eval_batches.value(),
            save_interval=self.save_interval.value(),
            data_loader_workers=self.data_loader_workers.value(),
            max_grad_norm=self.max_grad_norm.value(),
            activation_checkpointing=self.activation_checkpointing.isChecked(),
            device=self.device.currentText(),
            use_amp=self.use_amp.isChecked(),
            precision=self._precision_value(),
            seed=self.seed.value(),
            training_mode=training_mode or self._training_mode_value(),
            fine_tune_from_checkpoint=(
                Path(self.fine_tune_checkpoint.text())
                if training_mode != "pretrain" and self.fine_tune_checkpoint.text().strip()
                else None
            ),
            peft_method="none" if training_mode == "pretrain" else self._peft_method_value(),
            lora_rank=self.lora_rank.value(),
            lora_alpha=self.lora_alpha.value(),
            lora_dropout=self.lora_dropout.value(),
            lora_target_modules=self._lora_target_value(),
            resume=self.resume_training.isChecked(),
            resume_from_checkpoint=resume_path if self.resume_training.isChecked() else None,
            require_compatible_resume=self.resume_safety.isChecked(),
            early_stopping=self.early_stopping.isChecked(),
        )

    def _current_training_vocab_size(self, data_dir: Path) -> int:
        """Return the tokenizer vocabulary size for the current training dataset.

        Args:
            data_dir: Prepared dataset folder.

        Returns:
            Vocabulary size, or zero if unavailable.
        """

        summary_path = data_dir / "dataset_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
            if vocab_size > 0:
                return vocab_size
        tokenizer_path = data_dir / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
            vocab = tokenizer_data.get("model", {}).get("vocab", {})
            if isinstance(vocab, dict):
                return len(vocab)
        return 0

    def _checkpoint_vocab_size(self, checkpoint_path: Path) -> int:
        """Return the tokenizer vocabulary size saved in a checkpoint.

        Args:
            checkpoint_path: Checkpoint file to inspect.

        Returns:
            Saved vocabulary size, or zero when unavailable.
        """

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model_config = checkpoint.get("model_config", {})
            if isinstance(model_config, dict):
                return int(model_config.get("vocab_size", 0) or 0)
        except Exception as exc:
            LOGGER.warning("Could not inspect checkpoint vocab size for %s: %s", checkpoint_path, exc)
        return 0

    @staticmethod
    def _tokenizer_mismatch_help(checkpoint_vocab: int, dataset_vocab: int) -> str:
        """Return user-facing help for tokenizer mismatch errors.

        Args:
            checkpoint_vocab: Vocabulary size saved in the checkpoint.
            dataset_vocab: Vocabulary size in the prepared dataset.

        Returns:
            Help text.
        """

        return (
            f"Tokenizer mismatch: base checkpoint vocab is {checkpoint_vocab:,}, "
            f"but prepared dataset vocab is {dataset_vocab:,}.\n"
            "Fix: rebuild the fine-tune dataset using the exact tokenizer from the base model. "
            "In Ingest, set Tokenizer policy to Import tokenizer.json and choose the tokenizer.json "
            "beside the base checkpoint, then prepare the fine-tune dataset again."
        )

    def _training_run_artifacts(self, output_dir: Path) -> list[Path]:
        """Return training-run artifacts in a model output folder.

        Args:
            output_dir: Model output folder to inspect.

        Returns:
            Existing training-run artifact paths.
        """

        candidates = [
            output_dir / "checkpoints",
            output_dir / "final_model.pt",
            output_dir / "final_adapter.pt",
            output_dir / "training_summary.json",
            output_dir / "training_history.json",
            output_dir / "model_lineage.json",
        ]
        return [path for path in candidates if path.exists()]

    def _clear_training_run_artifacts(self, output_dir: Path) -> list[Path]:
        """Delete resumable training artifacts from a model output folder.

        Args:
            output_dir: Model output folder to clean.

        Returns:
            Paths that were removed.
        """

        output_dir = output_dir.resolve()
        removed: list[Path] = []
        candidates = self._training_run_artifacts(output_dir)
        for path in candidates:
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                resolved = path
            if output_dir not in resolved.parents and resolved != output_dir:
                LOGGER.warning("Skipped training cleanup outside model output folder: %s", path)
                continue
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(path)
            elif path.exists():
                path.unlink()
                removed.append(path)
        return removed

    def _selected_resume_path(self) -> Optional[Path]:
        """Return the selected or latest checkpoint path.

        Returns:
            Checkpoint path, or ``None`` when no checkpoint exists.
        """

        if self.resume_checkpoint.text().strip():
            return Path(self.resume_checkpoint.text())
        return latest_checkpoint(Path(self.model_dir.text()) / "checkpoints")

    def preview_resume_compatibility(self) -> None:
        """Preview whether the selected checkpoint can resume safely."""

        if not self.resume_training.isChecked():
            self.resume_training_preview.setText("[INFO] Resume latest is off. Enable resume to continue from a checkpoint.")
            return
        resume_path = self._selected_resume_path()
        if resume_path is None:
            self.resume_training_preview.setText("[INFO] No checkpoint found in the current model folder.")
            return
        if not resume_path.exists():
            self.resume_training_preview.setText(f"[BLOCK] Checkpoint does not exist:\n{resume_path}")
            return
        try:
            vocab_size = self._current_training_vocab_size(Path(self.train_data_dir.text()))
            if vocab_size <= 0:
                self.resume_training_preview.setText("[BLOCK] Could not determine current dataset tokenizer vocabulary size.")
                return
            model_config = self._current_model_config(vocab_size=vocab_size)
            training_config = self._current_training_config(resume_path)
            model_config.validate()
            training_config.validate()
            report = check_resume_compatibility(resume_path, model_config, training_config)
            errors = list(report.errors)
            if training_config.require_compatible_resume:
                if not report.can_load_optimizer_state:
                    errors.append("Safe resume requires matching optimizer state.")
                if not report.can_load_scheduler_state:
                    errors.append("Safe resume requires matching scheduler state.")
                if not report.can_load_scaler_state:
                    errors.append("Safe resume requires matching AMP scaler state.")
            lines: list[str] = []
            if errors:
                lines.append("[BLOCK] Resume is not safe with the current settings.")
            elif report.warnings:
                lines.append("[WARN] Resume is possible, but settings changed.")
            else:
                lines.append("[OK] Checkpoint can resume with the current settings.")
            lines.extend(f"[OK] {line}" for line in report.info)
            lines.extend(f"[WARN] {line}" for line in report.warnings)
            lines.extend(f"[BLOCK] {line}" for line in errors)
            if not training_config.require_compatible_resume and not errors:
                lines.append("[INFO] Safe resume is off. Compatible weights will load; incompatible optimizer state may be skipped.")
            self.resume_training_preview.setText("\n".join(lines))
        except Exception as exc:
            self.resume_training_preview.setText(f"[BLOCK] Could not check resume compatibility:\n{exc}")

    def preview_fine_tune_compatibility(self) -> None:
        """Preview whether the selected checkpoint can be used for fine-tuning."""

        stage_ok, stage_message = self._fine_tune_dataset_stage_status()
        if not stage_ok:
            self.fine_tune_preview.setText(f"[BLOCK] {stage_message}")
            return
        base_path = Path(self.fine_tune_checkpoint.text()) if self.fine_tune_checkpoint.text().strip() else None
        if base_path is None:
            self.fine_tune_preview.setText("[BLOCK] Choose a base checkpoint for fine-tuning.")
            return
        if not base_path.exists():
            self.fine_tune_preview.setText(f"[BLOCK] Fine-tune base checkpoint does not exist:\n{base_path}")
            return
        try:
            vocab_size = self._current_training_vocab_size(Path(self.train_data_dir.text()))
            if vocab_size <= 0:
                self.fine_tune_preview.setText("[BLOCK] Could not determine current dataset tokenizer vocabulary size.")
                return
            model_config = self._current_model_config(vocab_size=vocab_size)
            training_config = self._current_training_config()
            model_config.validate()
            report = check_resume_compatibility(base_path, model_config, training_config)
            lines: list[str] = []
            if report.errors:
                lines.append("[BLOCK] Base checkpoint cannot be fine-tuned with the current model/dataset settings.")
            else:
                lines.append("[OK] Base checkpoint weights can be used for fine-tuning.")
            lines.append(f"[OK] {stage_message}" if stage_ok else f"[BLOCK] {stage_message}")
            lines.extend(self._fine_tune_lineage_advice(base_path))
            lines.extend(f"[OK] {line}" for line in report.info)
            behavior_warnings = [
                warning for warning in report.warnings
                if not warning.startswith("Optimizer changed:") and not warning.startswith("LR scheduler changed:")
            ]
            lines.extend(f"[WARN] {line}" for line in behavior_warnings)
            lines.extend(f"[BLOCK] {line}" for line in report.errors)
            checkpoint_vocab = self._checkpoint_vocab_size(base_path)
            if checkpoint_vocab and checkpoint_vocab != vocab_size:
                lines.append(f"[FIX] {self._tokenizer_mismatch_help(checkpoint_vocab, vocab_size)}")
            if not report.errors:
                lines.append("[INFO] Fine-tuning starts fresh optimizer, scheduler, and scaler state.")
            self.fine_tune_preview.setText("\n".join(lines))
        except Exception as exc:
            self.fine_tune_preview.setText(f"[BLOCK] Could not check fine-tune compatibility:\n{exc}")

    def _fine_tune_lineage_advice(self, base_path: Path) -> list[str]:
        """Return guidance about the selected fine-tune base checkpoint.

        Args:
            base_path: Selected checkpoint path.

        Returns:
            Lines for the fine-tune compatibility report.
        """

        lines: list[str] = []
        try:
            output_dir = self._fine_tune_output_path().resolve()
            base_resolved = base_path.resolve()
            if output_dir == base_resolved.parent or output_dir in base_resolved.parents:
                return [
                    "[BLOCK] Selected base checkpoint is inside the current fine-tune output folder.",
                    "[FIX] Choose the original pretrained model or a completed earlier fine-tune from another folder.",
                ]
        except OSError:
            pass
        lineage_path = base_path.parent / "model_lineage.json"
        summary_path = base_path.parent / "training_summary.json"
        lineage = read_json(lineage_path, default={}) or {}
        summary = read_json(summary_path, default={}) or {}
        training_mode = str(lineage.get("training_mode") or (summary.get("training_config") or {}).get("training_mode") or "")
        stage = str((summary.get("model_lineage") or lineage).get("fine_tune_stage") or "")
        if training_mode == "fine_tune":
            stage_text = f" ({stage})" if stage else ""
            lines.append(f"[INFO] Selected base is a previous fine-tuned checkpoint{stage_text}.")
            lines.append("[INFO] This is correct for cumulative tuning, such as conversation -> instruction -> code.")
        elif training_mode == "pretrain":
            lines.append("[OK] Selected base is the pretrained model checkpoint.")
            lines.append("[INFO] This is correct when starting a new independent fine-tune branch.")
        else:
            lines.append("[INFO] Could not read model lineage; compatibility check will still validate tensor shapes.")
        project_base = self.current_project_file.parent / "models" / "final_model.pt" if self.current_project_file else None
        if project_base and project_base.exists():
            try:
                if base_path.resolve() != project_base.resolve() and training_mode != "fine_tune":
                    lines.append(f"[HINT] Project pretrained model is: {project_base}")
            except OSError:
                pass
        return lines

    def _training_history_path(self) -> Path:
        """Return the training history path for the selected model folder.

        Returns:
            Path to ``training_history.json``.
        """

        output_dir = getattr(self, "active_training_output_dir", None)
        if output_dir is None:
            output_dir = Path(self.model_dir.text())
        return Path(output_dir) / "training_history.json"

    def _load_training_history(self) -> list[dict[str, Any]]:
        """Load training run history.

        Returns:
            List of training run entries.
        """

        path = self._training_history_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def refresh_model_estimate(self) -> None:
        """Refresh model size, rough VRAM, and run history widgets."""

        model_config = self._current_model_config()
        training_config = self._current_training_config()
        data_dir = Path(self.train_data_dir.text())
        train_tokens = max(model_config.context_length * training_config.batch_size, 1)
        try:
            summary_path = data_dir / "dataset_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
                train_tokens = int(summary.get("train_token_count", summary.get("token_count", train_tokens)) or train_tokens)
                if vocab_size > 0:
                    model_config.vocab_size = vocab_size
            elif (data_dir / "tokenizer.json").exists():
                tokenizer_data = json.loads((data_dir / "tokenizer.json").read_text(encoding="utf-8"))
                vocab = tokenizer_data.get("model", {}).get("vocab", {})
                if vocab:
                    model_config.vocab_size = len(vocab)
        except Exception as exc:
            self.training_log.append(f"[WARN] Could not refresh dataset-based estimate: {exc}")
        estimate = estimate_training_resources(model_config, training_config, train_tokens)
        self.last_training_estimate = estimate
        self._update_model_estimate_chips(estimate, model_config, training_config, train_tokens)
        self.history_metric.setText(f"Runs: {len(self._load_training_history())}")
        self.training_log.append(
            "Model estimate refreshed: "
            f"{int(estimate['parameters']):,} params, "
            f"checkpoint {format_bytes(float(estimate['checkpoint_bytes']))}, "
            f"VRAM {format_bytes(float(estimate['vram_bytes']))}."
        )

    def _append_training_history(self, result: Any) -> None:
        """Persist a training run entry to ``training_history.json``.

        Args:
            result: Training result object.
        """

        history_path = self._training_history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = self._load_training_history()
        summary = {}
        try:
            if Path(result.summary_path).exists():
                summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        estimate = getattr(self, "last_training_estimate", {}) or {}
        entry = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_path": str(result.checkpoint_path),
            "summary_path": str(result.summary_path),
            "stopped": bool(getattr(result, "stopped", False)),
            "final_train_loss": result.final_train_loss,
            "final_val_loss": result.final_val_loss,
            "best_val_loss": summary.get("best_val_loss"),
            "recommended_checkpoint_path": summary.get("recommended_checkpoint_path"),
            "best_checkpoint_path": summary.get("best_checkpoint_path"),
            "dataset_dir": self.train_data_dir.text(),
            "dataset_version": (summary.get("model_lineage") or {}).get("dataset_version"),
            "training_run_id": summary.get("training_run_id"),
            "parameters": estimate.get("parameters") or summary.get("parameters"),
            "model_config": summary.get("model_config"),
            "training_config": summary.get("training_config"),
        }
        history.append(entry)
        history_path.write_text(json.dumps(history[-200:], indent=2), encoding="utf-8")
        self.history_metric.setText(f"Runs: {len(history[-200:])}")
        (self.active_training_log or self.training_log).append(f"Training history updated: {history_path}")

    def _run_training_preflight(self, model_config: ModelConfig, training_config: TrainingConfig) -> bool:
        """Run pre-training checklist and disk-space guard.

        Args:
            model_config: Selected model architecture.
            training_config: Selected training settings.

        Returns:
            True when training may continue.
        """

        log = self.active_training_log or self.training_log
        data_dir = Path(self.train_data_dir.text())
        output_dir = training_config.output_dir
        errors: list[str] = []
        warnings: list[str] = []
        info: list[str] = []
        resettable_errors: list[str] = []
        missing: list[str] = []
        if not (data_dir / "tokenizer.json").exists():
            missing.append("tokenizer.json")
        has_npy_tokens = (data_dir / "train_tokens.npy").exists() and (data_dir / "val_tokens.npy").exists()
        has_json_tokens = (data_dir / "train_tokens.json").exists() and (data_dir / "val_tokens.json").exists()
        if not has_npy_tokens and not has_json_tokens:
            missing.append("train_tokens.(npy/json), val_tokens.(npy/json)")
        if not data_dir.exists():
            errors.append(f"Dataset folder does not exist: {data_dir}")
        elif missing:
            errors.append(f"Dataset is not prepared. Missing: {', '.join(missing)}")
        else:
            info.append("Dataset artifacts found.")

        vocab_size = 0
        train_tokens = 0
        val_tokens = 0
        summary = {}
        try:
            summary_path = data_dir / "dataset_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
            if vocab_size <= 0:
                tokenizer_data = json.loads((data_dir / "tokenizer.json").read_text(encoding="utf-8"))
                vocab_size = len(tokenizer_data.get("model", {}).get("vocab", {}))
            train_tokens = int(summary.get("train_token_count", 0) or 0)
            val_tokens = int(summary.get("val_token_count", 0) or 0)
            if train_tokens <= 0 and (data_dir / "train_tokens.json").exists():
                train_tokens = len(json.loads((data_dir / "train_tokens.json").read_text(encoding="utf-8")))
            if val_tokens <= 0 and (data_dir / "val_tokens.json").exists():
                val_tokens = len(json.loads((data_dir / "val_tokens.json").read_text(encoding="utf-8")))
        except Exception as exc:
            warnings.append(f"Could not fully inspect dataset metadata: {exc}")

        if vocab_size > 0:
            model_config.vocab_size = vocab_size
            info.append(f"Tokenizer vocab: {vocab_size:,}.")
        elif not missing:
            errors.append("Could not determine tokenizer vocabulary size.")
        if train_tokens and train_tokens <= model_config.context_length:
            errors.append("Training token count must be larger than context length.")
        elif train_tokens:
            info.append(f"Training tokens: {train_tokens:,}; validation tokens: {val_tokens:,}.")
            if train_tokens < 50_000:
                warnings.append("Training token count is very small; expect smoke-test quality.")

        try:
            model_config.validate()
        except Exception as exc:
            errors.append(f"Model architecture is invalid: {exc}")
        try:
            training_config.validate()
        except Exception as exc:
            errors.append(f"Training options are invalid: {exc}")
        if model_config.attention_backend == "sdpa":
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                if training_config.device == "cuda" and torch.cuda.is_available():
                    flash_enabled = bool(getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)())
                    info.append("Attention backend: SDPA selected; Flash Attention may be used by PyTorch." if flash_enabled else "Attention backend: SDPA selected; CUDA flash kernel is not enabled.")
                else:
                    info.append("Attention backend: SDPA selected; CPU/backend fallback will be used if needed.")
            else:
                warnings.append("SDPA attention selected, but this PyTorch build does not expose scaled_dot_product_attention.")
        else:
            warnings.append("Manual attention backend selected. This is useful for debugging but can be slower.")
        if training_config.peft_method == "lora":
            info.append(
                "PEFT: LoRA adapters enabled. Intermediate checkpoints will save adapter weights; final_model.pt will be merged."
            )

        if training_config.device == "cuda" and not torch.cuda.is_available():
            errors.append("CUDA is selected, but PyTorch cannot use CUDA on this machine.")
        elif training_config.device == "cuda":
            info.append(f"CUDA ready: {torch.cuda.get_device_name(0)}.")
            if training_config.data_loader_workers > 0:
                info.append(f"CPU-assisted batch loading enabled with {training_config.data_loader_workers} worker(s).")
        else:
            warnings.append("CPU training is selected. This can be very slow.")
        if sys.platform.startswith("win") and training_config.data_loader_workers > 4:
            warnings.append("High CPU worker counts can duplicate dataset memory on Windows. Start with 2-4 workers and increase carefully.")

        active_resume_path: Optional[Path] = None
        resume_path = training_config.resume_from_checkpoint if training_config.resume else None
        if resume_path and not Path(resume_path).exists():
            errors.append(f"Selected resume checkpoint does not exist: {resume_path}")
        elif training_config.resume:
            if resume_path is None:
                resume_path = latest_checkpoint(output_dir / "checkpoints")
            if resume_path is None:
                info.append("Resume latest is enabled, but no checkpoint exists yet.")
            else:
                active_resume_path = Path(resume_path)
                try:
                    compatibility = check_resume_compatibility(active_resume_path, model_config, training_config)
                    info.extend(compatibility.info)
                    warnings.extend(compatibility.warnings)
                    errors.extend(compatibility.errors)
                    resettable_errors.extend(compatibility.errors)
                    if training_config.require_compatible_resume:
                        if not compatibility.can_load_optimizer_state:
                            message = "Safe resume requires matching optimizer state."
                            errors.append(message)
                            resettable_errors.append(message)
                        if not compatibility.can_load_scheduler_state:
                            message = "Safe resume requires matching scheduler state."
                            errors.append(message)
                            resettable_errors.append(message)
                        if not compatibility.can_load_scaler_state:
                            message = "Safe resume requires matching AMP scaler state."
                            errors.append(message)
                            resettable_errors.append(message)
                except Exception as exc:
                    errors.append(f"Could not inspect resume checkpoint: {exc}")
        if training_config.training_mode == "fine_tune" and active_resume_path is None:
            base_path = training_config.fine_tune_from_checkpoint
            if base_path is None:
                errors.append("Fine-tune mode requires a base checkpoint.")
            elif not Path(base_path).exists():
                errors.append(f"Fine-tune base checkpoint does not exist: {base_path}")
            else:
                try:
                    compatibility = check_resume_compatibility(Path(base_path), model_config, training_config)
                    info.append(f"Fine-tune base checkpoint: {Path(base_path).name}.")
                    warnings.extend(
                        warning for warning in compatibility.warnings
                        if not warning.startswith("Optimizer changed:") and not warning.startswith("LR scheduler changed:")
                    )
                    errors.extend(compatibility.errors)
                    checkpoint_vocab = self._checkpoint_vocab_size(Path(base_path))
                    if checkpoint_vocab and checkpoint_vocab != vocab_size:
                        errors.append(self._tokenizer_mismatch_help(checkpoint_vocab, vocab_size))
                    if not compatibility.errors:
                        info.append("Fine-tune base weights are compatible. Optimizer state will start fresh.")
                except Exception as exc:
                    errors.append(f"Could not inspect fine-tune base checkpoint: {exc}")
        elif training_config.training_mode == "pretrain" and active_resume_path is None:
            info.append("A fresh pretraining run will start from random weights.")
        elif training_config.training_mode == "fine_tune" and active_resume_path is not None:
            info.append("Existing run checkpoint found; training will resume that run instead of reloading the fine-tune base.")

        output_dir.mkdir(parents=True, exist_ok=True)
        estimate = estimate_training_resources(model_config, training_config, train_tokens)
        self.last_training_estimate = estimate
        self._update_model_estimate_chips(estimate, model_config, training_config, train_tokens)
        params = int(estimate["parameters"])
        checkpoint_bytes = float(estimate["checkpoint_bytes"])
        checkpoint_count = int(estimate["checkpoint_count"])
        estimated_storage = float(estimate["estimated_storage"])
        estimated_vram = float(estimate["vram_bytes"])
        free_bytes = shutil.disk_usage(output_dir).free
        info.append(f"Estimated parameters: {params:,}.")
        info.append(f"Estimated checkpoint size: {format_bytes(checkpoint_bytes)}.")
        info.append(f"Estimated training VRAM: {format_bytes(estimated_vram)}.")
        info.append(f"Estimated training storage need: {format_bytes(estimated_storage)}.")
        info.append(f"Free space on model drive: {format_bytes(free_bytes)}.")
        if training_config.device == "cuda" and torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info()
            info.append(f"GPU free/total VRAM: {format_bytes(free_vram)} / {format_bytes(total_vram)}.")
            if estimated_vram > free_vram * 0.9:
                warnings.append("Estimated VRAM is close to or above currently free GPU memory.")
        if free_bytes < estimated_storage * 1.25:
            errors.append("Not enough free disk space for estimated checkpoints and final model.")
        elif free_bytes < estimated_storage * 2:
            warnings.append("Free disk space is close to the estimated training storage need.")
        if checkpoint_count > 50:
            warnings.append("Save interval may create many checkpoints. Increase Save every or clean old checkpoints.")

        log.clear()
        log.append("Training checklist")
        for line in info:
            log.append(f"[OK] {line}")
        for line in warnings:
            log.append(f"[WARN] {line}")
        for line in errors:
            log.append(f"[ERROR] {line}")

        if errors:
            hard_errors = [error for error in errors if error not in resettable_errors]
            if active_resume_path is not None and resettable_errors and not hard_errors:
                message = (
                    "The existing checkpoint was created with different model settings, so it cannot be resumed.\n\n"
                    "This is expected if you intentionally changed architecture, block style, tokenizer, "
                    "context length, attention layout, or other checkpoint-shaped settings.\n\n"
                    "You can start a fresh training run with the current settings. This will delete old "
                    "checkpoints and training summaries in the selected model output folder.\n\n"
                    f"Model folder:\n{output_dir}\n\n"
                    "Continue and start from scratch?"
                )
                choice = QMessageBox.question(
                    self,
                    "Start From Scratch?",
                    message,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if choice == QMessageBox.Yes:
                    removed = self._clear_training_run_artifacts(output_dir)
                    training_config.resume = False
                    training_config.resume_from_checkpoint = None
                    self.resume_training.setChecked(False)
                    self.resume_checkpoint.clear()
                    log.append("")
                    log.append("Starting fresh run with current settings.")
                    for path in removed:
                        log.append(f"Removed old training artifact: {path}")
                    LOGGER.warning(
                        "User chose to discard incompatible resume checkpoint %s and start fresh in %s",
                        active_resume_path,
                        output_dir,
                    )
                    if training_config.training_mode == "fine_tune":
                        base_path = training_config.fine_tune_from_checkpoint
                        if base_path is None or not Path(base_path).exists():
                            log.append("[ERROR] Fine-tune mode requires an existing base checkpoint after reset.")
                            QMessageBox.warning(self, "Training blocked", "Fine-tune mode still needs a valid base checkpoint.")
                            return False
                        compatibility = check_resume_compatibility(Path(base_path), model_config, training_config)
                        if compatibility.errors:
                            for line in compatibility.errors:
                                log.append(f"[ERROR] {line}")
                            QMessageBox.warning(self, "Training blocked", "The base checkpoint is still incompatible with current settings.")
                            return False
                        log.append("[OK] Old run cleared; fine-tune base checkpoint is compatible.")
                    else:
                        log.append("[OK] Old run cleared; pretraining will start from random weights.")
                    self.project_state.setText("Training reset")
                    self.train_status.setText("Training: starting fresh")
                    return True
            LOGGER.error("Training blocked by preflight checklist.")
            for line in info:
                LOGGER.info("Training preflight OK: %s", line)
            for line in warnings:
                LOGGER.warning("Training preflight warning: %s", line)
            for line in errors:
                LOGGER.error("Training preflight error: %s", line)
            self.project_state.setText("Training blocked")
            self.train_status.setText("Training: blocked")
            QMessageBox.warning(self, "Training blocked", "Fix the checklist errors before starting training.")
            return False
        existing_artifacts = self._training_run_artifacts(output_dir)
        if (
            training_config.training_mode == "pretrain"
            and active_resume_path is None
            and existing_artifacts
        ):
            artifact_text = "\n".join(f"- {path.name}" for path in existing_artifacts)
            message = (
                "This model folder already contains training artifacts from a previous run.\n\n"
                "If you changed architecture or low-memory settings and want a clean start, "
                "the old checkpoints should be removed first.\n\n"
                f"Model folder:\n{output_dir}\n\n"
                f"Artifacts found:\n{artifact_text}\n\n"
                "Delete these artifacts and start from scratch?"
            )
            choice = QMessageBox.question(
                self,
                "Clean Previous Run?",
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                self.project_state.setText("Training cancelled")
                self.train_status.setText("Training: idle")
                log.append("Training cancelled. Previous run artifacts were kept.")
                return False
            removed = self._clear_training_run_artifacts(output_dir)
            training_config.resume = False
            training_config.resume_from_checkpoint = None
            self.resume_training.setChecked(False)
            self.resume_checkpoint.clear()
            log.append("")
            log.append("Previous run artifacts removed. Training will start from scratch with current settings.")
            for path in removed:
                log.append(f"Removed old training artifact: {path}")
            LOGGER.warning("User cleaned previous training artifacts in %s before starting from scratch.", output_dir)
        if warnings:
            message = "Training checklist has warnings. Continue anyway?\n\n" + "\n".join(f"- {warning}" for warning in warnings[:8])
            choice = QMessageBox.question(self, "Training warnings", message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if choice != QMessageBox.Yes:
                self.project_state.setText("Training cancelled")
                self.train_status.setText("Training: idle")
                return False
        return True

    def start_training(self) -> None:
        """Collect training options and start model training."""

        launch_target = self._training_launch_target_value()
        if launch_target == "runpod":
            self.launch_runpod_worker_for_current_training()
            return
        if launch_target == "remote":
            self.publish_remote_training_job()
            return
        self.active_training_log = self.training_log
        self.active_training_progress = self.training_progress
        self.active_training_final_button_text = "Start Training"
        resume_path = Path(self.resume_checkpoint.text()) if self.resume_checkpoint.text().strip() else None
        dataset_dir = Path(self.train_data_dir.text())
        vocab_size = self._current_training_vocab_size(dataset_dir)
        if vocab_size <= 0:
            QMessageBox.warning(self, "Training blocked", "Could not determine tokenizer vocabulary size. Prepare the dataset first.")
            return
        model_config = self._current_model_config(vocab_size=vocab_size)
        training_config = self._current_training_config(resume_path, training_mode="pretrain")
        if not self._run_training_preflight(model_config, training_config):
            return
        self.active_training_output_dir = training_config.output_dir
        self._init_telemetry_store(training_config.output_dir)
        self.training_log.append("")
        self.training_progress.setValue(0)
        self.training_epoch_metric.setText("Epoch: -")
        self.training_step_metric.setText("Step: -")
        self.training_loss_metric.setText("Train loss: -")
        self.training_val_metric.setText("Val loss: -")
        self.training_health_metric.setText("Health: -")
        self.training_health_points = []
        self.training_lr_metric.setText("LR: -")
        self.training_speed_metric.setText("Speed: -")
        self.training_grad_metric.setText("Grad: -")
        self.training_vram_metric.setText("VRAM: -")
        self.training_eta_metric.setText("ETA: -")
        self.loss_chart.clear()
        self.optimization_chart.clear()
        self.stability_chart.clear()
        self.throughput_chart.clear()
        self.memory_chart.clear()
        self.live_prediction_chart.update_distribution(0, None)
        self.live_attention_chart.update_heatmap(0, None)
        self.live_activation_chart.update_histogram(0, None)
        self.live_gradient_chart.update_flow(self.n_layer.value(), None, 0)
        self.live_progress.setValue(0)
        self.live_epoch_metric.setText("Epoch: -")
        self.live_step_metric.setText("Step: -")
        self.live_tokens_metric.setText("Tokens/sec: -")
        self.live_loss_metric.setText("Loss: -")
        self.live_lr_metric.setText("LR: -")
        self.live_data_metric.setText("Data: -")
        self.live_sample_text.setText("Training text: -")
        self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), 0, None)
        self._set_meter(self.live_cpu_bar, "CPU", self._system_cpu_value())
        self._set_meter(self.live_gpu_bar, "GPU memory", None)
        self._set_meter(self.live_vram_bar, "VRAM reserved", None)
        self._set_meter(self.live_ram_bar, "System RAM", self._system_ram_value())
        self.live_worker_status.setText(f"CPU workers: {self.data_loader_workers.value()}")
        self.training_log.append("Training started...")
        self.project_state.setText("Training")
        self.train_status.setText("Training: running")
        self._run_task(
            run_training_job,
            (dataset_dir, model_config, training_config),
            self._training_finished,
            self.training_log,
            self.training_progress,
            with_progress=True,
            button=self.train_button,
            stop_button=self.stop_training_button,
            busy_text="Training",
            task_kind="training",
        )

    def start_fine_tuning(self) -> None:
        """Collect fine-tuning options and start adaptation training."""

        fine_tune_launch = self._fine_tune_launch_target_value()
        if fine_tune_launch in {"remote", "runpod"}:
            stage_ok, stage_message = self._fine_tune_dataset_stage_status()
            self.refresh_fine_tune_workflow()
            if not stage_ok:
                self.fine_tune_log.append(stage_message)
                QMessageBox.warning(self, "Fine-tune blocked", stage_message)
                return
            if fine_tune_launch == "runpod":
                self.launch_runpod_worker_for_current_training(training_mode="fine_tune", stage=self._training_stage_value())
                self.fine_tune_log.append("RunPod fine-tune job launched. Watch Job Manager for worker assignment and progress.")
            else:
                self.publish_remote_training_job(training_mode="fine_tune", stage=self._training_stage_value())
                self.fine_tune_log.append("Remote fine-tune job queued. Watch Job Manager for worker assignment and progress.")
            return
        self.active_training_log = self.fine_tune_log
        self.active_training_progress = self.fine_tune_progress
        self.active_training_final_button_text = "Start Fine-Tune"
        stage_ok, stage_message = self._fine_tune_dataset_stage_status()
        self.refresh_fine_tune_workflow()
        if not stage_ok:
            self.fine_tune_log.append(stage_message)
            QMessageBox.warning(self, "Fine-tune blocked", stage_message)
            return
        resume_path = Path(self.resume_checkpoint.text()) if self.resume_checkpoint.text().strip() else None
        dataset_dir = Path(self.train_data_dir.text())
        vocab_size = self._current_training_vocab_size(dataset_dir)
        if vocab_size <= 0:
            QMessageBox.warning(self, "Fine-tune blocked", "Could not determine tokenizer vocabulary size. Prepare the fine-tuning dataset first.")
            return
        model_config = self._current_model_config(vocab_size=vocab_size)
        training_config = self._current_training_config(resume_path, training_mode="fine_tune")
        if not self._run_training_preflight(model_config, training_config):
            return
        self.active_training_output_dir = training_config.output_dir
        self._prepare_fine_tune_run_folder(training_config)
        self._init_telemetry_store(training_config.output_dir)
        self.fine_tune_log.append("")
        self.fine_tune_progress.setValue(0)
        self.training_progress.setValue(0)
        self.fine_tune_eta_metric.setText("ETA: -")
        self.fine_tune_epoch_metric.setText("Epoch: -")
        self.fine_tune_step_metric.setText("Step: -")
        self.fine_tune_loss_metric.setText("Train loss: -")
        self.fine_tune_val_metric.setText("Val loss: -")
        self.fine_tune_lr_metric.setText("LR: -")
        self.fine_tune_speed_metric.setText("Speed: -")
        self.fine_tune_grad_metric.setText("Grad: -")
        self.training_epoch_metric.setText("Epoch: -")
        self.training_step_metric.setText("Step: -")
        self.training_loss_metric.setText("Train loss: -")
        self.training_val_metric.setText("Val loss: -")
        self.training_health_metric.setText("Health: -")
        self.training_health_points = []
        self.training_lr_metric.setText("LR: -")
        self.training_speed_metric.setText("Speed: -")
        self.training_grad_metric.setText("Grad: -")
        self.training_vram_metric.setText("VRAM: -")
        self.training_eta_metric.setText("ETA: -")
        self.loss_chart.clear()
        self.optimization_chart.clear()
        self.stability_chart.clear()
        self.throughput_chart.clear()
        self.memory_chart.clear()
        self.live_progress.setValue(0)
        self.live_sample_text.setText("Training text: -")
        self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), 0, None)
        self.fine_tune_log.append("Fine-tuning started...")
        self.project_state.setText("Fine-tuning")
        self.train_status.setText("Training: fine-tuning")
        self._run_task(
            run_fine_tuning_job,
            (dataset_dir, model_config, training_config, self._training_stage_value()),
            self._training_finished,
            self.fine_tune_log,
            self.fine_tune_progress,
            with_progress=True,
            button=self.fine_tune_button,
            stop_button=self.stop_fine_tune_button,
            busy_text="Fine-tuning",
            task_kind="fine_tune",
        )

    @Slot(object)
    def _training_finished(self, result: Any) -> None:
        """Update UI after training finishes.

        Args:
            result: Training result.
        """

        log = self.active_training_log or self.training_log
        progress = self.active_training_progress or self.training_progress
        progress.setValue(100)
        if progress is not self.training_progress:
            self.training_progress.setValue(100)
        if hasattr(self, "live_progress"):
            self.live_progress.setValue(100)
        log.append(f"Saved model: {result.checkpoint_path}")
        log.append(f"Final train loss: {result.final_train_loss:.4f}")
        if result.final_val_loss is not None:
            log.append(f"Final validation loss: {result.final_val_loss:.4f}")
        training_summary: dict[str, Any] = {}
        try:
            training_summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        except Exception:
            training_summary = {}
        best_checkpoint = str(training_summary.get("recommended_checkpoint_path") or "")
        best_val_loss = training_summary.get("best_val_loss")
        if best_checkpoint:
            if best_val_loss is not None:
                log.append(f"Recommended checkpoint: {best_checkpoint} (best validation loss {float(best_val_loss):.4f})")
            else:
                log.append(f"Recommended checkpoint: {best_checkpoint}")
        output_dir = self.active_training_output_dir or Path(result.checkpoint_path).parent
        stage_key = self.active_task_kind if self.active_task_kind in {"training", "fine_tune"} else "training"
        self.export_model_dir.setText(str(output_dir))
        try:
            if stage_key != "fine_tune" and Path(output_dir).resolve() == Path(self.model_dir.text()).resolve():
                self.fine_tune_checkpoint.setText(str(result.checkpoint_path))
        except OSError:
            pass
        if getattr(result, "stopped", False):
            self.project_state.setText("Training stopped")
            self.train_status.setText("Training: stopped, checkpoint saved")
            log.append("Training stopped safely. Resume from this checkpoint or the latest checkpoint.")
        else:
            self.project_state.setText("Training complete")
            self.train_status.setText(f"Training: loss {result.final_train_loss:.4f}")
        title = "Fine-tuning complete" if stage_key == "fine_tune" else "Model training complete"
        if getattr(result, "stopped", False):
            title = "Fine-tuning stopped" if stage_key == "fine_tune" else "Model training stopped"
        completion_lines = [
            f"Checkpoint: {result.checkpoint_path}",
            f"Summary: {result.summary_path}",
            f"Final train loss: {result.final_train_loss:.4f}",
        ]
        if result.final_val_loss is not None:
            completion_lines.append(f"Final validation loss: {result.final_val_loss:.4f}")
        if best_checkpoint:
            completion_lines.append(f"Recommended checkpoint: {best_checkpoint}")
        if best_val_loss is not None:
            completion_lines.append(f"Best validation loss: {float(best_val_loss):.4f}")
        completion_lines.append(f"Output: {output_dir}")
        self._notify_complete(stage_key, title, completion_lines)
        self._append_training_history(result)
        self._clear_button_busy(self.active_training_final_button_text)
        self.active_training_log = None
        self.active_training_progress = None
        self.active_training_output_dir = None

    def run_benchmark(self) -> None:
        """Run benchmark prompts against the current trained model."""

        prompts = normalize_prompts(self.benchmark_prompts.toPlainText())
        self.benchmark_log.append(f"Running benchmark with {len(prompts)} prompt(s)...")
        self.benchmark_progress.setValue(0)
        self.project_state.setText("Benchmarking")
        self._run_task(
            evaluate_checkpoint,
            (
                Path(self.model_dir.text()),
                prompts,
                None,
                self.benchmark_tokens.value(),
                self.benchmark_temperature.value(),
                50,
                self.device.currentText(),
                self.benchmark_kv_cache.isChecked(),
            ),
            self._benchmark_finished,
            self.benchmark_log,
            self.benchmark_progress,
            with_progress=True,
            button=self.run_benchmark_button,
            stop_button=self.stop_benchmark_button,
            busy_text="Benchmarking",
        )

    @Slot(object)
    def _benchmark_finished(self, result: Any) -> None:
        """Update UI after benchmark prompts finish.

        Args:
            result: Benchmark result object.
        """

        self.benchmark_progress.setRange(0, 100)
        self.benchmark_progress.setValue(100)
        self.benchmark_log.append(
            f"Benchmark complete: {result.prompt_count} prompt(s), {result.total_seconds:.2f}s, "
            f"{result.total_generated_tokens} generated token(s), {result.tokens_per_second:.2f} tok/s."
        )
        self.benchmark_log.append(f"Benchmark saved: {result.output_path}")
        self.project_state.setText("Benchmark complete")
        self._clear_button_busy("Run Benchmark")

    def toggle_llm_model(self) -> None:
        """Load or unload the selected chat model depending on current state."""

        if self.chat_session is not None:
            self.unload_llm_model()
            return
        self.load_llm_model()

    def load_llm_model(self) -> None:
        """Load a selected model backend for chat testing."""

        backend = self._chat_backend_value()
        path_text = self.microgpt_chat_path.text().strip() if backend == "microgpt" else self.gguf_path.text().strip()
        if not path_text:
            required = "MicroGPT model folder or checkpoint" if backend == "microgpt" else "GGUF model file"
            QMessageBox.information(self, "Model required", f"Choose a {required} first.")
            return
        model_path = Path(path_text)
        self.chat_progress.setValue(0)
        self._render_chat_markdown("**Loading model...**")
        self.chat_stats.setText("Loading model...")
        self.project_state.setText("Loading chat model")
        self.chat_status.setText("Chat: loading model")
        loader = load_microgpt_chat_session if backend == "microgpt" else load_llama_chat_session
        args = (
            (model_path, self.device.currentText())
            if backend == "microgpt"
            else (model_path, self.llama_context.value(), self.llama_threads.value(), self.llama_gpu_layers.value())
        )
        self._run_task(
            loader,
            args,
            self._llm_loaded,
            self.chat_event_log,
            self.chat_progress,
            button=self.load_llm_button,
            busy_text="Loading Model",
        )

    @Slot(object)
    def _llm_loaded(self, session: Any) -> None:
        """Store a loaded GGUF chat session.

        Args:
            session: Loaded ``LlamaChatSession``.
        """

        self.chat_session = session
        self._clear_chat_messages()
        self.chat_markdown = ""
        self._add_chat_message(
            "assistant",
            f"Loaded model: `{session.model_path.name}`\n\n{session.runtime_summary}\n\nSend a message to begin.",
        )
        self.chat_progress.setValue(100)
        self.chat_stats.setText(session.runtime_summary)
        self.project_state.setText("Chat model loaded")
        self.chat_status.setText(f"Chat: {session.runtime_summary}")
        self._clear_button_busy("Unload")
        self._tip(self.load_llm_button, "Unload the currently loaded model from memory.")

    def unload_llm_model(self) -> None:
        """Unload the active chat model and clear chat state."""

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please wait for the current task to finish.")
            return
        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        self._clear_chat_messages()
        self.chat_markdown = ""
        self._add_chat_message("assistant", "Model unloaded.\n\nLoad a model to start testing.")
        self.chat_progress.setRange(0, 100)
        self.chat_progress.setValue(0)
        self.chat_stats.setText("Idle")
        self.project_state.setText("Ready")
        self.chat_status.setText("Chat: no model loaded")
        self.load_llm_button.setText("Load Model")
        self._update_chat_backend_controls()

    def send_chat_message(self) -> None:
        """Send a prompt to the loaded chat model."""

        if self.chat_session is None:
            QMessageBox.information(self, "Load model", "Load a model before sending a message.")
            return
        prompt = self.chat_input.toPlainText().strip()
        if not prompt:
            return
        self.pending_user_message = prompt
        self.chat_input.clear()
        self._add_chat_message("user", prompt, resend_prompt=prompt)
        self.chat_stream_reply = ""
        self._add_chat_message("assistant", "_Thinking..._", resend_prompt=prompt)
        self.chat_progress.setRange(0, 0)
        self.chat_stats.setText("Thinking...")
        self.project_state.setText("Generating")
        self.chat_status.setText("Chat: generating reply")
        streamer = stream_microgpt_chat_reply if self._chat_backend_value() == "microgpt" else stream_chat_reply
        self._run_task(
            streamer,
            (
                self.chat_session,
                prompt,
                self.system_prompt.toPlainText(),
                self.chat_max_tokens.value(),
                self.chat_temperature.value(),
                self.chat_top_p.value(),
                self.chat_repeat_penalty.value(),
                self.reasoning_effort.currentText(),
                self.thinking_enabled.isChecked(),
            ),
            self._chat_reply_finished,
            self.chat_event_log,
            self.chat_progress,
            with_progress=True,
            button=self.send_chat_button,
            stop_button=self.stop_chat_button,
            busy_text="Thinking",
        )

    @Slot(object)
    def _chat_reply_finished(self, reply: Any) -> None:
        """Render the model reply.

        Args:
            reply: Assistant reply text and metrics.
        """

        result = reply if isinstance(reply, dict) else {"reply": str(reply)}
        text = str(result.get("reply", "")).strip()
        if text:
            self.chat_stream_reply = text
        else:
            self.chat_stream_reply = self.chat_stream_reply or "_No reply returned._"
        self._render_chat_markdown(self.chat_stream_reply)
        self.chat_progress.setRange(0, 100)
        self.chat_progress.setValue(100)
        self._set_chat_stats(
            float(result.get("elapsed_seconds", 0.0)),
            int(result.get("token_count", 0)),
            float(result.get("tokens_per_second", 0.0)),
        )
        self.project_state.setText("Ready")
        self.chat_status.setText("Chat: ready")
        self._clear_button_busy("Send")

    def reset_chat(self) -> None:
        """Clear the chat transcript and model conversation memory."""

        if self.chat_session is not None:
            self.chat_session.reset()
        self._clear_chat_messages()
        self.chat_markdown = ""
        self.chat_stream_prefix = ""
        self.chat_stream_reply = ""
        self._add_chat_message("assistant", "Chat reset.")
        self.chat_stats.setText("Idle")
        self.chat_status.setText("Chat: ready")

    def _append_chat_markdown(self, role: str, content: str) -> None:
        """Append one rendered chat message.

        Args:
            role: Display role heading.
            content: Markdown content.
        """

        block = f"### {role}\n{content.strip()}\n"
        self.chat_markdown = f"{self.chat_markdown.rstrip()}\n\n{block}" if self.chat_markdown else block
        self._add_chat_message("user" if role.lower() in {"you", "user"} else "assistant", content)

    def create_bundle(self) -> None:
        """Create a portable model export bundle."""

        self.export_log.append("Creating model bundle...")
        self.export_progress.setValue(15)
        try:
            output = export_project_bundle(Path(self.export_model_dir.text()), Path(self.export_dir.text()))
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"Bundle created: {output}")
        self.export_status.setText("Export: bundle created")

    def quantize_model(self) -> None:
        """Create a quantized FP16 checkpoint when selected."""

        mode = self.quant_mode.currentText()
        if not mode.startswith("FP16"):
            self.export_log.append("This GGUF quantization target is planned. FP16 checkpoint quantization is available now.")
            return
        checkpoint = Path(self.export_model_dir.text()) / "final_model.pt"
        output = Path(self.export_dir.text()) / "final_model_fp16.pt"
        self.export_log.append("Creating FP16 checkpoint...")
        self.export_progress.setValue(20)
        try:
            result = quantize_checkpoint(checkpoint, output, mode="fp16")
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"Quantized checkpoint created: {result}")
        self.export_status.setText("Export: FP16 checkpoint ready")

    def export_hf_package(self) -> None:
        """Create an HF-style MicroGPT package."""

        self.export_log.append("Creating HF-style MicroGPT package...")
        self.export_progress.setValue(20)
        try:
            result = export_hf_microgpt_package(Path(self.export_model_dir.text()))
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"HF package created: {result}")
        self.export_log.append("Note: this package is MicroGPT model_type, not a llama.cpp-supported Llama model.")
        self.export_status.setText("Export: HF package ready")

    def export_llama_adapter(self) -> None:
        """Create a directly loadable Llama-family package when compatible."""

        self.export_log.append("Creating Llama-compatible adapter package...")
        self.export_progress.setValue(20)
        try:
            result = export_llama_adapter_package(Path(self.export_model_dir.text()))
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"Llama adapter package created: {result}")
        self.export_status.setText("Export: Llama adapter ready")

    def convert_hf_to_gguf(self) -> None:
        """Convert an HF-compatible model folder to GGUF through llama.cpp."""

        model_dir_text = self.export_model_dir.text().strip()
        llama_dir_text = self.llama_cpp_dir.text().strip()
        output_text = self.gguf_output_path.text().strip()
        if not model_dir_text:
            QMessageBox.warning(self, "GGUF blocked", "Choose the model core folder first.")
            return
        if not (Path(model_dir_text) / "hf_model").exists():
            QMessageBox.warning(
                self,
                "GGUF blocked",
                "GGUF conversion needs an HF model package first. Use Export HF Package, then convert a llama.cpp-supported model.",
            )
            return
        if not llama_dir_text:
            QMessageBox.warning(self, "GGUF blocked", "Choose your local llama.cpp folder containing convert_hf_to_gguf.py.")
            return
        if not output_text:
            QMessageBox.warning(self, "GGUF blocked", "Choose a GGUF output file path.")
            return
        self.export_log.append("Starting llama.cpp GGUF conversion...")
        self.export_progress.setValue(0)
        self._run_task(
            export_gguf_with_llama_cpp,
            (
                Path(model_dir_text),
                Path(llama_dir_text),
                Path(output_text),
                self.gguf_outtype.currentText(),
            ),
            self._gguf_conversion_finished,
            self.export_log,
            self.export_progress,
            button=self.gguf_convert_button,
            busy_text="Converting GGUF",
        )

    @Slot(object)
    def _gguf_conversion_finished(self, result: Any) -> None:
        """Update UI after GGUF conversion finishes.

        Args:
            result: GGUF output path.
        """

        self.export_progress.setValue(100)
        self.export_log.append(f"GGUF created: {result}")
        self.gguf_path.setText(str(result))
        self.export_status.setText("Export: GGUF ready")
        self._clear_button_busy("Convert HF to GGUF")

    def _apply_preset(self, preset: str) -> None:
        """Apply architecture values for a preset.

        Args:
            preset: Selected preset name.
        """

        if preset == "Tiny":
            self.n_embd.setValue(128)
            self.n_head.setValue(4)
            self.n_layer.setValue(4)
        elif preset == "Small":
            self.n_embd.setValue(512)
            self.n_head.setValue(8)
            self.n_layer.setValue(8)


def main() -> None:
    """Launch the PySide6 desktop application."""

    log_file = setup_logging()
    qInstallMessageHandler(qt_message_handler)
    LOGGER.info("Starting %s. Log file: %s", APP_NAME, log_file)
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
        except Exception:
            LOGGER.exception("Could not set Windows app user model ID")
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    app.setWindowIcon(MainWindow._static_app_icon())
    splash = StartupValidationSplash()
    splash.setWindowIcon(MainWindow._static_app_icon())
    splash.show()
    QTimer.singleShot(0, lambda: _apply_windows_taskbar_icon(splash))
    QApplication.processEvents()
    try:
        _run_startup_validations(splash)
    except Exception as exc:
        LOGGER.exception("Startup validation failed")
        splash.append_log(f"✗ Startup blocked: {exc}")
        splash.close()
        proceed = QMessageBox.question(
            None,
            "Startup validation failed",
            "One or more startup checks failed.\n\n"
            f"{exc}\n\n"
            "Do you want to continue anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if proceed != QMessageBox.Yes:
            return
        LOGGER.warning("User chose to continue after failed startup validation.")
    splash.close()
    window = MainWindow()
    chooser = ProjectChoiceDialog()
    chooser.setWindowIcon(MainWindow._static_app_icon())
    QTimer.singleShot(0, lambda: _apply_windows_taskbar_icon(chooser))
    if chooser.exec() != QDialog.Accepted:
        LOGGER.info("Startup closed at project selection screen")
        return
    try:
        if chooser.choice == "new":
            base_dir = QFileDialog.getExistingDirectory(
                None,
                "Choose folder where the new project will be created",
                str(DEFAULT_PROJECTS_DIR),
            )
            if not base_dir:
                return
            project_name, ok = QInputDialog.getText(None, "Project name", "Enter project name:", text="MicroLLMProject")
            if not ok:
                return
            project_name = project_name.strip() or "MicroLLMProject"
            window._create_project_at(project_name, Path(base_dir))
        elif chooser.choice == "open":
            project_file, _ = QFileDialog.getOpenFileName(
                None,
                "Open Micro LLM project",
                str(DEFAULT_PROJECTS_DIR),
                "Micro LLM project (project.json *.json);;All files (*)",
            )
            if not project_file:
                return
            window._open_project_file(Path(project_file))
        elif chooser.choice == "recent":
            if chooser.selected_project_file is None:
                return
            window._open_project_file(chooser.selected_project_file)
        elif chooser.choice == "test_local_llm":
            window.show_chat_only_mode()
    except Exception as exc:
        LOGGER.exception("Project setup failed during startup")
        QMessageBox.critical(None, "Project setup failed", f"Could not complete project setup.\n\n{exc}")
        return
    window.show()
    QTimer.singleShot(0, window.apply_windows_taskbar_icon)
    interrupt_timer = QTimer()
    interrupt_timer.timeout.connect(lambda: None)
    interrupt_timer.start(200)
    window.interrupt_timer = interrupt_timer
    signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, window.request_shutdown_from_signal))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()