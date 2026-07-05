from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def build_job_manager_tab(window) -> QWidget:
    """Build the distributed job manager page.

    Args:
        window: Main window instance.

    Returns:
        Job manager page widget.
    """

    page = window._panel()
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(18, 18, 18, 10)
    page_layout.setSpacing(8)

    title_row = QHBoxLayout()
    title_row.addWidget(window._page_title("Job Manager"))
    title_row.addStretch(1)
    window.job_refresh_button = QPushButton("Refresh")
    window.job_refresh_button.clicked.connect(window.refresh_job_manager_tab)
    window._tip(window.job_refresh_button, "Reload worker and job status from the coordinator state store.")
    window.job_stale_button = QPushButton("Mark Stale Offline")
    window.job_stale_button.clicked.connect(window.mark_stale_workers_offline)
    window._tip(window.job_stale_button, "Mark remote workers offline when they have not sent a heartbeat recently.")
    window.job_pause_button = QPushButton("Pause All")
    window.job_pause_button.clicked.connect(window.pause_all_managed_jobs)
    window._tip(window.job_pause_button, "Ask remote workers to pause active jobs and hold queued jobs.")
    window.job_resume_button = QPushButton("Resume All")
    window.job_resume_button.clicked.connect(window.resume_all_managed_jobs)
    window._tip(window.job_resume_button, "Return paused jobs to the queue.")
    window.job_stop_button = QPushButton("Stop All Jobs")
    window.job_stop_button.clicked.connect(window.stop_all_managed_jobs)
    window._tip(window.job_stop_button, "Request cooperative stop for every queued or active managed job.")
    for button in (
        window.job_refresh_button,
        window.job_stale_button,
        window.job_pause_button,
        window.job_resume_button,
        window.job_stop_button,
    ):
        button.setMaximumWidth(170)
        title_row.addWidget(button)
    page_layout.addLayout(title_row)

    summary = QHBoxLayout()
    window.job_worker_count_label = QLabel("Workers: -")
    window.job_worker_count_label.setObjectName("Metric")
    window.job_active_count_label = QLabel("Active jobs: -")
    window.job_active_count_label.setObjectName("Metric")
    window.job_queue_count_label = QLabel("Queued jobs: -")
    window.job_queue_count_label.setObjectName("Metric")
    window.job_db_label = QLabel("State DB: -")
    window.job_db_label.setObjectName("Metric")
    for label in (
        window.job_worker_count_label,
        window.job_active_count_label,
        window.job_queue_count_label,
        window.job_db_label,
    ):
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary.addWidget(label)
    summary.addStretch(1)
    page_layout.addLayout(summary)

    scroll = QScrollArea()
    scroll.setObjectName("PageScroll")
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    content = QWidget()
    root = QVBoxLayout(content)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(10)
    scroll.setWidget(content)
    page_layout.addWidget(scroll, 1)

    connection_form = QFormLayout()
    window._configure_form(connection_form)
    window.coordinator_host = QLineEdit("0.0.0.0")
    window._tip(window.coordinator_host, "Network address used by the coordinator API. Use 0.0.0.0 to accept workers from other machines.")
    window.coordinator_port = window._spin(1, 65535, 8765)
    window._tip(window.coordinator_port, "Coordinator API port. Remote workers connect to this port.")
    window.coordinator_artifact_root = QLineEdit(str(Path.home() / ".micro_llm_creator" / "artifacts"))
    window._tip(window.coordinator_artifact_root, "Folder where job input bundles and uploaded worker result bundles are stored.")
    window.coordinator_public_url = QLineEdit("http://127.0.0.1:8765")
    window._tip(window.coordinator_public_url, "URL workers use to reach this coordinator. Use this machine's LAN/IP address for remote machines.")
    connection_form.addRow("Host", window.coordinator_host)
    connection_form.addRow("Port", window.coordinator_port)
    connection_form.addRow("Artifact root", window._path_row(window.coordinator_artifact_root, directory=True))
    connection_form.addRow("Worker URL", window.coordinator_public_url)

    connection_buttons = QHBoxLayout()
    window.coordinator_start_button = QPushButton("Start Coordinator")
    window.coordinator_start_button.clicked.connect(window.start_coordinator_server)
    window._tip(window.coordinator_start_button, "Start the HTTP coordinator so remote workers can register, claim jobs, download inputs, and upload outputs.")
    window.coordinator_stop_button = QPushButton("Stop Coordinator")
    window.coordinator_stop_button.setEnabled(False)
    window.coordinator_stop_button.clicked.connect(window.stop_coordinator_server)
    window._tip(window.coordinator_stop_button, "Stop the coordinator API. Running remote workers will lose connection until it starts again.")
    window.publish_remote_job_button = QPushButton("Publish Remote Job")
    window.publish_remote_job_button.clicked.connect(window.publish_remote_training_job)
    window._tip(window.publish_remote_job_button, "Bundle the current dataset/checkpoints and queue the current training settings for remote workers.")
    connection_buttons.addWidget(window.coordinator_start_button)
    connection_buttons.addWidget(window.coordinator_stop_button)
    connection_buttons.addWidget(window.publish_remote_job_button)
    connection_buttons.addStretch(1)
    connection_form.addRow("", connection_buttons)
    window.coordinator_status_label = QLabel("Coordinator: stopped")
    window.coordinator_status_label.setObjectName("Metric")
    connection_form.addRow("Status", window.coordinator_status_label)
    coordinator_card = window._card("COORDINATOR API / ARTIFACT SYNC", connection_form)
    coordinator_card.setMinimumHeight(230)
    root.addWidget(coordinator_card, 0)

    runpod_form = QFormLayout()
    window._configure_form(runpod_form)
    window.runpod_api_key = QLineEdit()
    window.runpod_api_key.setEchoMode(QLineEdit.Password)
    window._tip(window.runpod_api_key, "RunPod API key. Stored in runpod_config.json inside the project folder.")
    window.runpod_gpu_type = QComboBox()
    window.runpod_gpu_type.setEditable(True)
    window.runpod_gpu_type.addItems([
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A5000",
        "NVIDIA A40",
        "NVIDIA L40S",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA H100 80GB HBM3",
    ])
    window._tip(window.runpod_gpu_type, "Preferred RunPod GPU type. Availability depends on RunPod capacity.")
    window.runpod_cloud_type = QComboBox()
    window.runpod_cloud_type.addItems(["COMMUNITY", "SECURE"])
    window._tip(window.runpod_cloud_type, "Community is usually cheaper. Secure is more controlled and often more predictable.")
    window.runpod_image = QLineEdit("runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    window._tip(window.runpod_image, "Docker image used for the cloud worker. The default PyTorch image installs this app's worker bundle at startup.")
    window.runpod_container_disk = window._spin(20, 500, 80)
    window._tip(window.runpod_container_disk, "Temporary container disk in GB. Larger values help with dependency install and cache.")
    window.runpod_volume_gb = window._spin(20, 1000, 40)
    window._tip(window.runpod_volume_gb, "Persistent Pod volume in GB mounted at /workspace.")
    window.runpod_min_ram = window._spin(8, 512, 16)
    window._tip(window.runpod_min_ram, "Minimum system RAM per GPU in GB.")
    window.runpod_min_vcpu = window._spin(2, 128, 4)
    window._tip(window.runpod_min_vcpu, "Minimum virtual CPUs per GPU.")
    window.runpod_spot = QCheckBox("Use interruptible cheaper pods")
    window.runpod_spot.setChecked(True)
    window._tip(window.runpod_spot, "Interruptible pods cost less but may be reclaimed. Checkpoints make recovery easier.")
    window.runpod_auto_terminate = QCheckBox("Exit worker after one job")
    window.runpod_auto_terminate.setChecked(True)
    window._tip(window.runpod_auto_terminate, "Worker claims one job and exits. Verify the Pod has stopped in RunPod when the job completes.")
    window.runpod_save_button = QPushButton("Save RunPod Settings")
    window.runpod_save_button.clicked.connect(window.save_runpod_settings)
    window._tip(window.runpod_save_button, "Save RunPod settings to runpod_config.json.")
    window.runpod_launch_button = QPushButton("Launch RunPod Worker")
    window.runpod_launch_button.clicked.connect(window.launch_runpod_worker_for_current_training)
    window._tip(window.runpod_launch_button, "Publish the current training job and create a RunPod cloud worker to claim it.")
    runpod_buttons = QHBoxLayout()
    runpod_buttons.addWidget(window.runpod_save_button)
    runpod_buttons.addWidget(window.runpod_launch_button)
    runpod_buttons.addStretch(1)
    runpod_form.addRow("API key", window.runpod_api_key)
    runpod_form.addRow("GPU", window.runpod_gpu_type)
    runpod_form.addRow("Cloud", window.runpod_cloud_type)
    runpod_form.addRow("Image", window.runpod_image)
    runpod_form.addRow("Disk / volume", _inline_widgets(window.runpod_container_disk, window.runpod_volume_gb))
    runpod_form.addRow("RAM / vCPU", _inline_widgets(window.runpod_min_ram, window.runpod_min_vcpu))
    runpod_form.addRow("", window.runpod_spot)
    runpod_form.addRow("", window.runpod_auto_terminate)
    runpod_form.addRow("", runpod_buttons)
    window.runpod_status_label = QLabel("RunPod: not configured")
    window.runpod_status_label.setObjectName("Metric")
    window.runpod_status_label.setWordWrap(True)
    runpod_form.addRow("Status", window.runpod_status_label)
    runpod_card = window._card("RUNPOD CLOUD GPU", runpod_form)
    runpod_card.setMinimumHeight(330)
    root.addWidget(runpod_card, 0)

    window.job_worker_table = _table(
        ["Worker", "Status", "Backend", "Device", "Last Seen", "Active Job", "CPU/RAM/GPU", "Labels"]
    )
    root.addWidget(window._card("WORKERS / CONNECTIONS", _table_layout(window.job_worker_table)), 0)

    window.job_table = _table(
        ["Job", "Stage", "Status", "Worker", "Backend", "Epoch", "Step", "Batch", "Layers", "Loss", "Speed", "Updated"]
    )
    root.addWidget(window._card("JOBS / TRAINING ASSIGNMENTS", _table_layout(window.job_table)), 0)

    window.job_manager_log = QTextEdit()
    window.job_manager_log.setReadOnly(True)
    window.job_manager_log.setMinimumHeight(120)
    root.addWidget(window._card("COORDINATOR TELEMETRY", _table_layout(window.job_manager_log)), 0)

    window.job_manager_progress = window._thin_progress()
    page_layout.addWidget(window.job_manager_progress)
    window.load_runpod_settings()
    window.refresh_job_manager_tab()
    return page


def _table(headers: list[str]) -> QTableWidget:
    """Create a table widget.

    Args:
        headers: Column labels.

    Returns:
        Table widget.
    """

    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.setMinimumHeight(170)
    return table


def _table_layout(widget: QWidget) -> QVBoxLayout:
    """Wrap a widget in a layout.

    Args:
        widget: Widget to wrap.

    Returns:
        Layout containing the widget.
    """

    layout = QVBoxLayout()
    layout.addWidget(widget)
    return layout


def _inline_widgets(*widgets: QWidget) -> QWidget:
    """Create a compact inline row.

    Args:
        widgets: Widgets to place side by side.

    Returns:
        Container widget.
    """

    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    for widget in widgets:
        row.addWidget(widget)
    row.addStretch(1)
    return holder


def set_table_rows(table: QTableWidget, rows: list[list[str]]) -> None:
    """Replace all table rows.

    Args:
        table: Table widget.
        rows: Row values.
    """

    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            item = QTableWidgetItem(value)
            item.setToolTip(value)
            table.setItem(row_index, column_index, item)
