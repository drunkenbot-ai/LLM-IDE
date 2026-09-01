from __future__ import annotations

# WindowCoreMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface.widgets.app_shell import build_main_shell, update_navigation_icons
from interface.theme import apply_theme, current_theme, normalize_theme
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class WindowCoreMixin:
    def __init__(self) -> None:
        """Create the main application window."""

        super().__init__()
        self.log_file_path = setup_logging()
        LOGGER.info("Creating %s main window", APP_NAME)
        if QApplication.instance():
            QApplication.instance().setFont(QFont("Arial", 10))
        licensed = bool(QApplication.instance().property("license_valid"))
        self.setWindowTitle(
            f"{APP_NAME} {APP_VERSION} "
            f"({'licensed' if licensed else 'Trial Version'})"
        )
        self.setWindowIcon(self._app_icon())
        self._windows_icon_handles: list[int] = []
        self.resize(1240, 820)
        self.thread: Optional[QThread] = None
        self.worker: Optional[TaskWorker] = None
        self.result_bridge: Optional[WorkerSignalBridge] = None
        self.stop_event: Optional[Event] = None
        self.progress_queue: Optional[Queue] = None
        self.active_log: Optional[QTextEdit] = None
        self.active_progress_bar: Optional[QProgressBar] = None
        self.active_button: Optional[QPushButton] = None
        self.active_stop_button: Optional[QPushButton] = None
        self.active_button_text = ""
        self.active_button_restore_text = ""
        self.active_task_kind = ""
        self.active_task_terminal_event: Optional[dict[str, Any]] = None
        self.dataset_diagnostic_sources: set[str] = set()
        self.dataset_result_applied = False
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
        self.active_training_button: Optional[QPushButton] = None
        self.active_training_mode = "pretrain"
        self.active_training_final_button_text = "Start Training"
        self.active_training_output_dir: Optional[Path] = None
        self.persisted_training_process: dict[str, Any] = {}
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

        self.theme_name = current_theme()
        self._apply_theme()

        shell = self._build_shell()
        self.setCentralWidget(shell)
        update_navigation_icons(self)
        self._install_ui_event_logging(shell)
        self._install_wheel_guard(shell)
        self._refresh_notification_manager()
        self._initialize_training_controller()
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

    def edit_focused_widget(self, method_name: str) -> None:
        """Run a standard edit operation on the currently focused editor.

        Args:
            method_name: Qt editor method to invoke, such as ``copy`` or
                ``selectAll``.
        """
        widget = QApplication.focusWidget()
        method = getattr(widget, method_name, None) if widget is not None else None
        if callable(method):
            method()

    def show_about_dialog(self) -> None:
        """Display the application identity and version."""
        QMessageBox.information(
            self,
            f"About {APP_NAME}",
            f"{APP_NAME} {APP_VERSION}\n\nA desktop environment for building, training, and using LLMs.",
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

    def _apply_theme(self) -> None:
        """Apply the window's selected theme application-wide."""
        self.theme_name = apply_theme(self.theme_name)

    def update_theme_actions(self) -> None:
        """Synchronize the theme menu checks with the active theme."""
        if not hasattr(self, "system_theme_action"):
            return
        self.system_theme_action.setChecked(self.theme_name == "system")
        self.dark_theme_action.setChecked(self.theme_name == "dark")

    def set_theme(self, theme: object, persist: bool = True) -> None:
        """Select an application theme and persist it for the active project.

        Args:
            theme: Requested theme identifier.
            persist: Whether to save the selection to the active project.
        """
        self.theme_name = normalize_theme(theme)
        self._apply_theme()
        if hasattr(self, "side_rail"):
            update_navigation_icons(self)
        self.update_theme_actions()
        if persist and self.current_project_file is not None:
            self.save_project()

    def _build_shell(self) -> QWidget:
        """Build the top-level dashboard shell from reusable widgets."""
        return build_main_shell(self, APP_NAME)

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
        if index == self.live_page_index:
            self._render_current_live_snapshot()
        if index == 5:
            QTimer.singleShot(20, self.refresh_job_manager_tab)

    def show_chat_only_mode(self) -> None:
        """Collapse the UI to chat-only view for quick local LLM testing."""

        if hasattr(self, "top_bar"):
            self.top_bar.hide()
        if hasattr(self, "side_rail"):
            self.side_rail.hide()
        self._switch_page(8)
        licensed = bool(QApplication.instance().property("license_valid"))
        self.setWindowTitle(
            f"{APP_NAME} {APP_VERSION} "
            f"({'licensed' if licensed else 'Trial Version'})"
        )
        self.resize(980, 760)

    def resizeEvent(self, event: Any) -> None:
        """Refresh responsive layouts when the main window changes size.

        Args:
            event: Qt resize event.
        """

        super().resizeEvent(event)
        self._refresh_training_layout()

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

        candidates = [
            Path(_app.__file__).resolve().parent / "drunken_bot_logo_small.png",
        ]
        if hasattr(sys, "_MEIPASS"):
            candidates.insert(0, Path(sys._MEIPASS) / "drunken_bot_logo_small.png")
        app_root = os.environ.get("DRUNKENBOT_APP_ROOT")
        if app_root:
            candidates.insert(0, Path(app_root) / "interface" / "drunken_bot_logo_small.png")
        return next((path for path in candidates if path.exists()), candidates[0])

    @staticmethod
    def _app_logo_pixmap(size: int = 64) -> QPixmap:
        """Load the bundled logo as a pixmap.

        Args:
            size: Maximum square size.

        Returns:
            Logo pixmap, or null pixmap when the file is missing.
        """

        pixmap = QPixmap(str(WindowCoreMixin._app_logo_path()))
        if pixmap.isNull():
            return pixmap
        return pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    @staticmethod
    def _static_app_icon() -> QIcon:
        """Create the static app icon.

        Returns:
            Application icon.
        """

        logo_path = WindowCoreMixin._app_logo_path()
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

        return Path(_app.__file__).with_name("drunkenbot_llm_ide.ico")

    @staticmethod
    def _ensure_windows_icon_file() -> Optional[Path]:
        """Ensure the generated Windows ``.ico`` file exists.

        Returns:
            Icon path on Windows, otherwise ``None``.
        """

        if sys.platform != "win32":
            return None
        icon_path = WindowCoreMixin._windows_icon_path()
        if icon_path.exists():
            return icon_path
        icon = WindowCoreMixin._static_app_icon()
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
