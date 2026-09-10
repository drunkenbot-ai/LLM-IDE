"""Thinkbox Deadline & Fox Renderfarm inspired Cluster Job Monitor.

Layout:
- Left Column (Full Height): Training Jobs Monitor (master job table).
- Right Column Top: Tabbed details widget containing Tasks & Data Shards + Diagnostic Logs.
- Right Column Bottom: Cluster Worker Fleet & Hardware Utilization table.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _styled_table(headers: list[str], min_height: int = 140) -> QTableWidget:
    """Create an industrial-grade table styled like Thinkbox Deadline."""
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setSelectionMode(QTableWidget.SingleSelection)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    table.setMinimumHeight(min_height)
    return table


def set_table_rows(table: QTableWidget, rows: list[list[str]]) -> None:
    """Replace all table rows efficiently with diff caching."""
    if getattr(table, "_rendered_row_cache", None) == rows:
        return
    table._rendered_row_cache = [list(r) for r in rows]

    table.setUpdatesEnabled(False)
    try:
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(row_index, column_index, item)
    finally:
        table.setUpdatesEnabled(True)


def build_job_manager_tab(window) -> QWidget:
    """Build the Deadline / Fox Renderfarm inspired Cluster Job Monitor page."""
    page = window._panel()
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(14, 12, 14, 10)
    page_layout.setSpacing(8)

    # -------------------------------------------------------------------------
    # 1. Top Master Control Header & Storage Config
    # -------------------------------------------------------------------------
    header_box = QFrame()
    header_box.setObjectName("Card")
    header_layout = QVBoxLayout(header_box)
    header_layout.setContentsMargins(12, 10, 12, 10)
    header_layout.setSpacing(8)

    # Top Title & Summary Row
    title_row = QHBoxLayout()
    title_label = window._page_title("Cluster Job Monitor")
    title_row.addWidget(title_label)
    title_row.addSpacing(16)

    window.cluster_status_label = QLabel("Status: Idle")
    window.cluster_status_label.setObjectName("Metric")
    window.cluster_workers_label = QLabel("Active Workers: 0")
    window.cluster_workers_label.setObjectName("Metric")
    window.cluster_round_label = QLabel("Round: -")
    window.cluster_round_label.setObjectName("Metric")

    title_row.addWidget(window.cluster_status_label)
    title_row.addWidget(window.cluster_workers_label)
    title_row.addWidget(window.cluster_round_label)
    title_row.addStretch(1)

    window.job_refresh_button = QPushButton("Refresh Fleet")
    window.job_refresh_button.clicked.connect(window.refresh_job_manager_tab)
    window._tip(window.job_refresh_button, "Query shared storage SQLite bus and reload all jobs, workers, and tasks.")
    window.cluster_refresh_btn = window.job_refresh_button
    title_row.addWidget(window.job_refresh_button)

    header_layout.addLayout(title_row)

    # Storage Path & Action Controls Row
    controls_row = QHBoxLayout()
    controls_row.setSpacing(8)

    default_shared = (
        os.environ.get("LLM_SHARED_PATH")
        or os.environ.get("LLM_SHARED_DIR")
        or str(Path.home() / "llm_cluster_shared")
    )
    window.cluster_shared_dir = QLineEdit(str(default_shared))
    window.cluster_shared_dir.setPlaceholderText("Shared central network storage directory (SMB/NFS/NAS)...")
    window._tip(window.cluster_shared_dir, "Path to centralized network storage folder (SMB/NFS/NAS) shared across all worker machines.")

    window.cluster_browse_btn = QPushButton("Browse...")
    window.cluster_browse_btn.clicked.connect(window.browse_cluster_shared_dir)

    controls_row.addWidget(QLabel("<b>Shared Storage:</b>"))
    controls_row.addWidget(window.cluster_shared_dir, 1)
    controls_row.addWidget(window.cluster_browse_btn)
    controls_row.addSpacing(12)

    # Master Action Buttons
    window.cluster_launch_btn = QPushButton("▶ Launch Job")
    window.cluster_launch_btn.clicked.connect(window.launch_cluster_training_job)
    window._tip(window.cluster_launch_btn, "Queue and dispatch a distributed Local SGD training job across the cluster.")

    window.cluster_pause_btn = QPushButton("⏸ Pause")
    window.cluster_pause_btn.clicked.connect(window.pause_cluster_job)
    window._tip(window.cluster_pause_btn, "Send cooperative PAUSE signal to active cluster job.")

    window.cluster_resume_btn = QPushButton("▶ Resume")
    window.cluster_resume_btn.clicked.connect(window.resume_cluster_job)
    window._tip(window.cluster_resume_btn, "Resume execution of paused cluster job.")

    window.cluster_stop_btn = QPushButton("⏹ Stop")
    window.cluster_stop_btn.clicked.connect(window.stop_cluster_job)
    window._tip(window.cluster_stop_btn, "Cooperatively stop active cluster training job.")

    window.cluster_requeue_btn = QPushButton("🔄 Re-queue")
    window.cluster_requeue_btn.clicked.connect(window.requeue_selected_cluster_job)
    window._tip(window.cluster_requeue_btn, "Re-queue the selected job to allow restarting or resuming.")

    window.cluster_local_worker_btn = QPushButton("⚡ Start Local Worker(s)")
    window.cluster_local_worker_btn.clicked.connect(window.toggle_local_cluster_worker)
    window._tip(window.cluster_local_worker_btn, "Spawn background cluster worker processes on this workstation for all detected GPUs.")

    window.cluster_clean_offline_btn = QPushButton("🧹 Clean Offline")
    window.cluster_clean_offline_btn.clicked.connect(window.clean_offline_cluster_workers)
    window._tip(window.cluster_clean_offline_btn, "Purge offline and stale workers from the fleet directory.")

    controls_row.addWidget(window.cluster_launch_btn)
    controls_row.addWidget(window.cluster_pause_btn)
    controls_row.addWidget(window.cluster_resume_btn)
    controls_row.addWidget(window.cluster_stop_btn)
    controls_row.addWidget(window.cluster_requeue_btn)
    controls_row.addWidget(window.cluster_local_worker_btn)
    controls_row.addWidget(window.cluster_clean_offline_btn)

    header_layout.addLayout(controls_row)
    page_layout.addWidget(header_box)

    # -------------------------------------------------------------------------
    # 2. Main Horizontal Splitter: Left = Jobs Monitor (Full Height), Right = Details & Fleet
    # -------------------------------------------------------------------------
    main_splitter = QSplitter(Qt.Horizontal)
    main_splitter.setChildrenCollapsible(False)

    # LEFT PANE: Full Height Training Jobs Monitor (Green Area)
    jobs_card = QFrame()
    jobs_card.setObjectName("Card")
    jobs_layout = QVBoxLayout(jobs_card)
    jobs_layout.setContentsMargins(10, 8, 10, 8)
    jobs_layout.setSpacing(6)

    jobs_header = QHBoxLayout()
    jobs_header.addWidget(QLabel("<b>TRAINING JOBS MONITOR</b>"))
    jobs_header.addStretch(1)
    window.jobs_summary_label = QLabel("0 total jobs")
    window.jobs_summary_label.setObjectName("Metric")
    jobs_header.addWidget(window.jobs_summary_label)
    jobs_layout.addLayout(jobs_header)

    window.cluster_jobs_table = _styled_table(
        ["Status", "Job ID", "Model / Config", "Progress", "Loss", "Speed", "Rounds", "Created"],
        min_height=260,
    )
    window.cluster_jobs_table.itemSelectionChanged.connect(window.on_cluster_job_selected)
    jobs_layout.addWidget(window.cluster_jobs_table, 1)

    main_splitter.addWidget(jobs_card)

    # RIGHT PANE: Vertical Splitter (Top: Tasks/Logs Tabs, Bottom: Worker Fleet)
    right_splitter = QSplitter(Qt.Vertical)
    right_splitter.setChildrenCollapsible(False)

    # RIGHT TOP: Tab Widget containing Tasks & Shards + Running Diagnostic Logs
    details_card = QFrame()
    details_card.setObjectName("Card")
    details_layout = QVBoxLayout(details_card)
    details_layout.setContentsMargins(10, 8, 10, 8)
    details_layout.setSpacing(6)

    details_header = QHBoxLayout()
    window.selected_job_title_label = QLabel("<b>JOB DETAILS & DIAGNOSTICS</b> (Select a job)")
    details_header.addWidget(window.selected_job_title_label)
    details_header.addStretch(1)

    clear_log_btn = QPushButton("Clear Console")
    clear_log_btn.setMaximumWidth(120)
    clear_log_btn.clicked.connect(lambda: window.clear_active_cluster_log())
    details_header.addWidget(clear_log_btn)
    details_layout.addLayout(details_header)

    window.cluster_details_tabs = QTabWidget()

    # Tab 1: Tasks & Data Shards
    window.cluster_tasks_table = _styled_table(
        ["Shard", "Status", "Assigned Worker", "Device", "Round", "Loss", "Speed"],
        min_height=140,
    )
    window.cluster_details_tabs.addTab(window.cluster_tasks_table, "Tasks & Data Shards")

    # Tab 2: Selected Job Events Log
    window.job_events_log = QTextEdit()
    window.job_events_log.setReadOnly(True)
    window.job_events_log.setPlaceholderText("Logs and progress events for the selected training job will appear here...")
    window.cluster_details_tabs.addTab(window.job_events_log, "Job Event Log")

    # Tab 3: Selected Worker Diagnostics (stdout/stderr from SQLite)
    window.cluster_worker_log = QTextEdit()
    window.cluster_worker_log.setReadOnly(True)
    window.cluster_worker_log.setPlaceholderText("Select a worker row in the fleet table below to inspect its real-time console and error logs...")
    window.cluster_details_tabs.addTab(window.cluster_worker_log, "Worker Diagnostics")

    # Tab 4: Overall Cluster Event Stream
    window.cluster_log = QTextEdit()
    window.cluster_log.setReadOnly(True)
    window.cluster_details_tabs.addTab(window.cluster_log, "Cluster Event Stream")

    details_layout.addWidget(window.cluster_details_tabs, 1)
    right_splitter.addWidget(details_card)

    # RIGHT BOTTOM: Discovered Cluster Worker Fleet Table (Red Area)
    fleet_card = QFrame()
    fleet_card.setObjectName("Card")
    fleet_layout = QVBoxLayout(fleet_card)
    fleet_layout.setContentsMargins(10, 8, 10, 8)
    fleet_layout.setSpacing(6)

    fleet_header = QHBoxLayout()
    fleet_header.addWidget(QLabel("<b>CLUSTER WORKER FLEET & HARDWARE UTILIZATION</b>"))
    fleet_header.addStretch(1)
    window.fleet_utilization_label = QLabel("Nodes: 0 | VRAM: -")
    window.fleet_utilization_label.setObjectName("Metric")
    fleet_header.addWidget(window.fleet_utilization_label)
    fleet_layout.addLayout(fleet_header)

    window.cluster_worker_table = _styled_table(
        ["Worker ID", "Hostname", "GPU / Device", "VRAM Usage", "RAM Usage", "CPU", "Status", "Current Job", "Last Seen"],
        min_height=140,
    )
    window.cluster_worker_table.setContextMenuPolicy(Qt.CustomContextMenu)
    window.cluster_worker_table.customContextMenuRequested.connect(window.show_cluster_worker_context_menu)
    window.cluster_worker_table.itemSelectionChanged.connect(window.on_cluster_worker_selected)
    fleet_layout.addWidget(window.cluster_worker_table, 1)

    right_splitter.addWidget(fleet_card)
    right_splitter.setSizes([320, 320])

    main_splitter.addWidget(right_splitter)
    main_splitter.setSizes([620, 580])

    page_layout.addWidget(main_splitter, 1)

    # -------------------------------------------------------------------------
    # 3. Bottom Progress Bar & Compatibility Aliases
    # -------------------------------------------------------------------------
    window.job_manager_progress = window._thin_progress()
    page_layout.addWidget(window.job_manager_progress)

    # Backward compatibility bindings
    window.cluster_log_tabs = window.cluster_details_tabs
    window.job_worker_table = window.cluster_worker_table
    window.job_table = window.cluster_jobs_table
    window.job_manager_log = window.cluster_log

    return page
