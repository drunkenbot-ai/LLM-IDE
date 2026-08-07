from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import re
import signal
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from typing import Optional, Union

from PySide6.QtCore import (
    QEvent, QObject, QPoint, QSize, QThread, QTimer, Qt, Signal, Slot,
    qInstallMessageHandler,
)
from PySide6.QtGui import (
    QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygon,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QStackedWidget,
    QTextBrowser,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from engine.app_logging import qt_message_handler, setup_logging
from engine.coordinator import JobManager
from engine.coordinator.artifacts import create_job_artifact_bundle
from engine.coordinator.api_server import CoordinatorApiServer
from engine.config import DatasetConfig, ModelConfig, TrainingConfig
from engine.conversation_datasets import CONVERSATION_DATASET_PRESETS
from engine.conversation_presets import dataset_ids_for_stage, dataset_stage_label
from engine.contracts.jobs import BackendKind, RuntimeSpec, TrainingJobSpec
from engine.dataset_preview_scan import scan_dataset_preview
from engine.dataset_preview_health import check_project_health
from engine.dataset_build import build_dataset
from engine.evaluation import DEFAULT_BENCHMARK_PROMPTS, evaluate_checkpoint, normalize_prompts
from engine.export import (
    export_gguf_with_llama_cpp, export_hf_microgpt_package, export_llama_adapter_package,
    export_project_bundle, quantize_checkpoint,
)
from engine.external_dataset import (
    DEFAULT_MANIFEST_URL, download_latest_dataset, is_newer_version, load_manifest,
)
from engine.fine_tuning_service import run_fine_tuning_job
from engine.llama_chat import LlamaChatSession
from engine.llama_chat import load_llama_chat_session, stream_chat_reply
from engine.lineage import read_json
from engine.microgpt_chat import load_microgpt_chat_session, stream_microgpt_chat_reply
from engine.notifier import (
    NotificationManager, default_notifier_config_path, ensure_notifier_config,
)
from engine.runpod_cloud import (
    RunPodClient, default_runpod_config_path, ensure_runpod_config, load_runpod_config,
    public_url_is_cloud_reachable, RunPodConfig, save_runpod_config,
    create_runpod_worker_bundle,
)
from engine.telemetry_store import (
    initialize_store, insert_metric, latest_run, rows_until, telemetry_db_path,
)
from engine.training_planning import estimate_training_resources, format_bytes
from engine.training_resume import check_resume_compatibility, latest_checkpoint
from engine.training_service import run_training_job

from interface.startup_splash import StartupSplash

from engine.license_client import load_stored_license_key
from interface.license_activation_dialog import LicenseActivationDialog, run_license_check_responsively
from interface.chat_widgets import ChatMessageWidget
from interface.markdown_renderer import markdown_to_html
from interface.startup import _register_recent_project
from interface.tabs.benchmark_tab import build_benchmark_tab
from interface.tabs.chat_tab import build_chat_tab
from interface.tabs.dataset_plan_tab import (
    build_dataset_plan_tab, dataset_plan_defaults, default_data_root,
    iter_default_data_files, populate_default_data_tree,
)
from interface.tabs.dataset_tab import build_dataset_tab
from interface.tabs.export_tab import build_export_tab
from interface.tabs.fine_tuning_tab import build_fine_tuning_tab
from interface.tabs.job_manager_tab import build_job_manager_tab, set_table_rows
from interface.tabs.live_tab import build_live_training_tab
from interface.tabs.training_tab import build_training_tab
from interface.workers import ProcessTaskWorker, TaskWorker, WorkerSignalBridge

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


APP_NAME = "DrunkenBot-IDE"
# Bump on every release that should require a version-ceiling check against
# licenses -- this is what license_client.check_license_at_launch compares
# against a license's version_ceiling/grace_period_until.
APP_VERSION = "1.4.0"
# TODO: point at the real deployed cloud-service URL once it has one.
# Overridable via env var so ops can point a build at a different
# deployment (dev/staging/prod) without a code change or rebuild.

LICENSE_SERVER_URL = "https://drunkenbot.store"
WINDOWS_APP_ID = "DrunkenBot.LLMIDE"
LOGGER = logging.getLogger(__name__)
APP_HOME_DIR = Path.home() / ".drunkenbot_ide"
DEFAULT_CACHE_DIR = APP_HOME_DIR / "cache"
DEFAULT_PROJECTS_DIR = APP_HOME_DIR / "projects"
RECENT_PROJECTS_PATH = APP_HOME_DIR / "recent_projects.json"
_WINDOWS_ICON_HANDLES: list[int] = []
_LOGO_FONT_FAMILY: Optional[str] = None


from interface.startup import (
    ProjectChoiceDialog, StartupValidationSplash, _apply_windows_taskbar_icon,
    _run_startup_validations,
)
from interface.startup_validation import _run_startup_tests
from interface.main_window_part1 import MainWindowPart1
from interface.main_window_part2 import MainWindowPart2
from interface.main_window_part3 import MainWindowPart3
from interface.main_window_part4 import MainWindowPart4
from interface.main_window_part5 import MainWindowPart5
from interface.main_window_part6 import MainWindowPart6
from interface.main_window_part7 import MainWindowPart7
from interface.main_window_part8 import MainWindowPart8
from interface.main_window_part9 import MainWindowPart9
from interface.main_window_part10 import MainWindowPart10
from interface.main_window_part11 import MainWindowPart11
from interface.main_window_part12 import MainWindowPart12
from interface.main_window_part13 import MainWindowPart13
from interface.main_window_part14 import MainWindowPart14
from interface.main_window_part15 import MainWindowPart15
from interface.main_window_part16 import MainWindowPart16
from interface.main_window_part17 import MainWindowPart17
from interface.main_window_part18 import MainWindowPart18

class MainWindow(MainWindowPart1, MainWindowPart2, MainWindowPart3, MainWindowPart4, MainWindowPart5, MainWindowPart6, MainWindowPart7, MainWindowPart8, MainWindowPart9, MainWindowPart10, MainWindowPart11, MainWindowPart12, MainWindowPart13, MainWindowPart14, MainWindowPart15, MainWindowPart16, MainWindowPart17, MainWindowPart18, QMainWindow):
    """Main application window composed from focused UI mixins."""

def _ensure_valid_license(splash: "StartupValidationSplash") -> bool:
    """Block app launch until a valid license is confirmed.

    Checks the currently stored license key (if any). On failure, shows
    :class:`LicenseActivationDialog` in a loop -- unlike the general
    startup-validation flow elsewhere in ``main()``, there is deliberately
    no "continue anyway" option here: an unlicensed launch is not a
    degraded-but-usable state, it's the one thing this app must not do.

    Args:
        splash: Startup splash screen, used to show progress.

    Returns:
        True if the app is licensed to proceed, False if the user cancelled
        activation and the app should exit.
    """

    splash.append_log("Checking license...")
    QApplication.processEvents()

    stored_key = load_stored_license_key()
    if stored_key:
        result = run_license_check_responsively(APP_VERSION, LICENSE_SERVER_URL)
        if result.valid:
            splash.append_log(
                "[OK] License valid"
                + (" (offline grace period)" if result.used_offline_grace else "")
            )
            QApplication.instance().setProperty("license_valid", True)
            return True
        initial_message = result.reason
    else:
        initial_message = "No license activated on this machine yet."

    splash.hide()
    try:
        dialog = LicenseActivationDialog(APP_VERSION, LICENSE_SERVER_URL, initial_message)
        dialog.setWindowIcon(MainWindow._static_app_icon())
        if dialog.exec() == QDialog.Accepted and dialog.result_info is not None:
            QApplication.instance().setProperty("license_valid", True)
            splash.append_log("[OK] License activated")
            return True
        if dialog.trial_requested:
            QApplication.instance().setProperty("license_valid", False)
            splash.append_log("[TRIAL] User selected trial version")
            return True
        LOGGER.info("License activation cancelled by user; exiting.")
        QApplication.instance().setProperty("startup_aborted", True)
        return False
    finally:
        splash.show()
        splash.raise_()

def main(app: Optional[QApplication] = None, splash: Optional[StartupSplash] = None) -> None:
    """Launch the PySide6 desktop application."""

    owns_app = app is None
    log_file = setup_logging()
    qInstallMessageHandler(qt_message_handler)
    LOGGER.info("Starting %s. Log file: %s", APP_NAME, log_file)
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
        except Exception:
            LOGGER.exception("Could not set Windows app user model ID")
    app = app or QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    app.setWindowIcon(MainWindow._static_app_icon())
    splash = splash or StartupValidationSplash()
    splash.setWindowIcon(MainWindow._static_app_icon())
    splash.show()
    QTimer.singleShot(0, lambda: _apply_windows_taskbar_icon(splash))
    QApplication.processEvents()
    if not _ensure_valid_license(splash):
        splash.close()
        if not owns_app:
            app.quit()
            app.setProperty("startup_aborted", True)
        return
    try:
        _run_startup_validations(splash)
    except Exception as exc:
        LOGGER.exception("Startup validation failed")
        splash.append_log(f"[FAIL] Startup blocked: {exc}")
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
            if not owns_app:
                app.quit()
                app.setProperty("startup_aborted", True)
            return
        LOGGER.warning("User chose to continue after failed startup validation.")
    splash.close()
    while True:
        chooser = ProjectChoiceDialog()
        chooser.setWindowIcon(MainWindow._static_app_icon())
        QTimer.singleShot(0, lambda dialog=chooser: _apply_windows_taskbar_icon(dialog))
        if chooser.exec() != QDialog.Accepted:
            LOGGER.info("Startup closed at project selection screen")
            if not owns_app:
                app.quit()
                app.setProperty("startup_aborted", True)
            return
        window = MainWindow()
        try:
            if chooser.choice == "new":
                base_dir = QFileDialog.getExistingDirectory(
                    None,
                    "Choose folder where the new project will be created",
                    str(DEFAULT_PROJECTS_DIR),
                )
                if not base_dir:
                    window.deleteLater()
                    continue
                project_name, ok = QInputDialog.getText(None, "Project name", "Enter project name:", text="DrunkenBotProject")
                if not ok:
                    window.deleteLater()
                    continue
                project_name = project_name.strip() or "DrunkenBotProject"
                window._create_project_at(project_name, Path(base_dir))
            elif chooser.choice == "open":
                project_file, _ = QFileDialog.getOpenFileName(
                    None,
                    "Open DrunkenBot LLM project",
                    str(DEFAULT_PROJECTS_DIR),
                    "DrunkenBot LLM project (project.json *.json);;All files (*)",
                )
                if not project_file:
                    window.deleteLater()
                    continue
                window._open_project_file(Path(project_file))
            elif chooser.choice == "recent":
                if chooser.selected_project_file is None:
                    window.deleteLater()
                    continue
                window._open_project_file(chooser.selected_project_file)
            elif chooser.choice == "test_local_llm":
                window.show_chat_only_mode()
        except Exception as exc:
            LOGGER.exception("Project setup failed during startup")
            QMessageBox.critical(None, "Project setup failed", f"Could not complete project setup.\n\n{exc}")
            window.deleteLater()
            continue
        break
    window.show()
    QTimer.singleShot(0, window.apply_windows_taskbar_icon)
    interrupt_timer = QTimer()
    interrupt_timer.timeout.connect(lambda: None)
    interrupt_timer.start(200)
    window.interrupt_timer = interrupt_timer
    signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, window.request_shutdown_from_signal))
    if owns_app:
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
