"""Cluster Fleet Manager Tab for Port-Blocked Distributed Training (Local SGD).

Provides UI controls for:
1. Setting central shared network drive path (SMB/NFS/NAS).
2. Monitoring active cluster worker nodes and hardware telemetry.
3. Configuring Local SGD hyperparameters (sync interval K, straggler timeout).
4. Launching, pausing, resuming, and stopping distributed training jobs.
5. Launching an optional local worker daemon.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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


def _cluster_table(headers: list[str]) -> QTableWidget:
    """Create a styled table for cluster telemetry."""
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    table.setMinimumHeight(150)
    return table


def set_cluster_table_rows(table: QTableWidget, rows: list[list[str]]) -> None:
    """Replace rows in a cluster table widget with diff caching."""
    if getattr(table, "_rendered_row_cache", None) == rows:
        return
    table._rendered_row_cache = [list(r) for r in rows]

    table.setUpdatesEnabled(False)
    try:
        table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            for col_idx, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setToolTip(val)
                table.setItem(row_idx, col_idx, item)
    finally:
        table.setUpdatesEnabled(True)


def build_cluster_card(window) -> QWidget:
    """Build the Port-Blocked Cluster Card for embedding in Job Manager or Standalone."""
    form = QFormLayout()
    window._configure_form(form)

    # Central Shared Drive path
    import os
    default_shared = os.environ.get("LLM_SHARED_PATH") or os.environ.get("LLM_SHARED_DIR") or str(Path.home() / "llm_cluster_shared")
    window.cluster_shared_dir = QLineEdit(str(default_shared))
    window._tip(window.cluster_shared_dir, "Path to centralized network storage folder (SMB/NFS/NAS) shared across all worker machines.")

    shared_dir_row = QHBoxLayout()
    shared_dir_row.addWidget(window.cluster_shared_dir, 1)
    window.cluster_browse_btn = QPushButton("Browse...")
    window.cluster_browse_btn.clicked.connect(window.browse_cluster_shared_dir)
    window.cluster_refresh_btn = QPushButton("Refresh Fleet")
    window.cluster_refresh_btn.clicked.connect(window.refresh_cluster_status)
    shared_dir_row.addWidget(window.cluster_browse_btn)
    shared_dir_row.addWidget(window.cluster_refresh_btn)
    form.addRow("Shared network dir", shared_dir_row)

    # Local SGD Settings
    window.cluster_sync_steps = window._spin(10, 5000, 250)
    window._tip(window.cluster_sync_steps, "Local training steps (K) each worker completes before synchronizing weights.")
    window.cluster_max_rounds = window._spin(1, 1000, 10)
    window._tip(window.cluster_max_rounds, "Total periodic synchronization rounds to execute.")
    window.cluster_sync_timeout = window._spin(15, 3600, 180)
    window._tip(window.cluster_sync_timeout, "Straggler timeout in seconds before proceeding with weight averaging without slow workers.")
    window.cluster_min_workers = window._spin(1, 64, 1)
    window._tip(window.cluster_min_workers, "Minimum number of workers that must deposit weights before round averaging can proceed.")

    hparams_row = QHBoxLayout()
    hparams_row.addWidget(QLabel("Sync interval (K steps):"))
    hparams_row.addWidget(window.cluster_sync_steps)
    hparams_row.addWidget(QLabel("Max rounds:"))
    hparams_row.addWidget(window.cluster_max_rounds)
    hparams_row.addWidget(QLabel("Straggler timeout (s):"))
    hparams_row.addWidget(window.cluster_sync_timeout)
    hparams_row.addWidget(QLabel("Min workers:"))
    hparams_row.addWidget(window.cluster_min_workers)
    hparams_row.addStretch(1)
    form.addRow("Local SGD parameters", hparams_row)

    # Action Buttons
    action_row = QHBoxLayout()
    window.cluster_launch_btn = QPushButton("Launch Cluster Job")
    window.cluster_launch_btn.clicked.connect(window.launch_cluster_training_job)
    window._tip(window.cluster_launch_btn, "Queue a distributed Local SGD training job on the shared network drive.")

    window.cluster_pause_btn = QPushButton("Pause Job")
    window.cluster_pause_btn.clicked.connect(window.pause_cluster_job)

    window.cluster_resume_btn = QPushButton("Resume Job")
    window.cluster_resume_btn.clicked.connect(window.resume_cluster_job)

    window.cluster_stop_btn = QPushButton("Stop Job")
    window.cluster_stop_btn.clicked.connect(window.stop_cluster_job)

    window.cluster_local_worker_btn = QPushButton("Start Local Worker(s)")
    window.cluster_local_worker_btn.clicked.connect(window.toggle_local_cluster_worker)
    window._tip(window.cluster_local_worker_btn, "Run worker daemon processes on this machine for all detected GPUs.")

    window.cluster_restart_local_btn = QPushButton("Restart Local")
    window.cluster_restart_local_btn.clicked.connect(window.restart_local_cluster_workers)
    window._tip(window.cluster_restart_local_btn, "Restart all local cluster worker processes.")

    window.cluster_clean_offline_btn = QPushButton("Clean Offline")
    window.cluster_clean_offline_btn.clicked.connect(window.clean_offline_cluster_workers)
    window._tip(window.cluster_clean_offline_btn, "Remove offline/stale workers from the discovered fleet list.")

    action_row.addWidget(window.cluster_launch_btn)
    action_row.addWidget(window.cluster_pause_btn)
    action_row.addWidget(window.cluster_resume_btn)
    action_row.addWidget(window.cluster_stop_btn)
    action_row.addWidget(window.cluster_local_worker_btn)
    action_row.addWidget(window.cluster_restart_local_btn)
    action_row.addWidget(window.cluster_clean_offline_btn)
    action_row.addStretch(1)
    form.addRow("Cluster control", action_row)

    # Status summary
    status_row = QHBoxLayout()
    status_row.setSpacing(24)
    window.cluster_status_label = QLabel("Status: Idle")
    window.cluster_status_label.setObjectName("Metric")
    window.cluster_workers_label = QLabel("Active Workers: 0")
    window.cluster_workers_label.setObjectName("Metric")
    window.cluster_round_label = QLabel("Round: -")
    window.cluster_round_label.setObjectName("Metric")

    status_row.addWidget(window.cluster_status_label)
    status_row.addWidget(window.cluster_workers_label)
    status_row.addWidget(window.cluster_round_label)
    status_row.addStretch(1)
    form.addRow("Telemetry", status_row)

    # Tables & Logs
    window.cluster_worker_table = _cluster_table(
        ["Worker ID", "Hostname", "GPU / Device", "VRAM Usage", "RAM Usage", "CPU", "Status", "Last Heartbeat"]
    )
    window.cluster_worker_table.setContextMenuPolicy(Qt.CustomContextMenu)
    window.cluster_worker_table.customContextMenuRequested.connect(window.show_cluster_worker_context_menu)
    window.cluster_worker_table.itemSelectionChanged.connect(window.on_cluster_worker_selected)

    # Worker diagnostic log viewer
    worker_log_header = QHBoxLayout()
    window.cluster_selected_worker_label = QLabel("<b>WORKER DIAGNOSTIC LOGS</b> (Select a worker in the table above)")
    worker_log_header.addWidget(window.cluster_selected_worker_label)
    worker_log_header.addStretch(1)
    window.cluster_refresh_worker_logs_btn = QPushButton("Refresh Worker Logs")
    window.cluster_refresh_worker_logs_btn.clicked.connect(window.refresh_selected_worker_logs)
    worker_log_header.addWidget(window.cluster_refresh_worker_logs_btn)

    window.cluster_worker_log = QTextEdit()
    window.cluster_worker_log.setReadOnly(True)
    window.cluster_worker_log.setMinimumHeight(120)
    window.cluster_worker_log.setPlaceholderText("Select a worker row above to inspect its real-time console and error logs from shared database...")

    window.cluster_log = QTextEdit()
    window.cluster_log.setReadOnly(True)
    window.cluster_log.setMinimumHeight(100)

    holder = QWidget()
    holder_layout = QVBoxLayout(holder)
    holder_layout.setContentsMargins(0, 0, 0, 0)
    holder_layout.setSpacing(8)
    holder_layout.addLayout(form)
    holder_layout.addWidget(QLabel("<b>DISCOVERED CLUSTER WORKER FLEET</b>"))
    holder_layout.addWidget(window.cluster_worker_table)
    holder_layout.addLayout(worker_log_header)
    holder_layout.addWidget(window.cluster_worker_log)
    holder_layout.addWidget(QLabel("<b>CLUSTER EVENT LOG</b>"))
    holder_layout.addWidget(window.cluster_log)

    card = window._card("PORT-BLOCKED CLUSTER (LOCAL SGD / SHARED STORAGE)", QVBoxLayout())
    card.layout().addWidget(holder)
    return card


def build_cluster_tab(window) -> QWidget:
    """Build a standalone page widget for the Cluster Fleet Manager."""
    page = window._panel()
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(18, 18, 18, 10)
    page_layout.setSpacing(8)

    title_row = QHBoxLayout()
    title_row.addWidget(window._page_title("Cluster Fleet Manager (Local SGD)"))
    title_row.addStretch(1)
    page_layout.addLayout(title_row)

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

    root.addWidget(build_cluster_card(window), 1)
    return page
