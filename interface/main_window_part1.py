from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart1:
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
        self.training_nav = self._nav_button("AI")
        self.training_nav.setText("AI")
        self.fine_tune_nav = self._nav_button("FT")
        self.live_nav = self._nav_button("LIVE")
        self.jobs_nav = self._nav_button("JOB")
        self.benchmark_nav = self._nav_button("Bench")
        self.export_nav = self._nav_button("X")
        self.chat_nav = self._nav_button("Chat")
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
