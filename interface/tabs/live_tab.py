"""Neural Observatory & Live Training Flight Deck strictly matching Concept Image 3 with Real Telemetry & Cluster Fleet Support."""

from __future__ import annotations

import os
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from interface.charts import LossChartWidget
from interface.live_widgets import (
    LiveDistributionWidget,
    LiveGradientFlowWidget,
    LiveHeatmapWidget,
    LiveHistogramWidget,
    ModelFlowWidget,
)


def _detect_system_hardware() -> tuple[str, str]:
    """Detect local GPU/CPU hardware capabilities and return (status_text, vram_default)."""
    try:
        import torch

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            total_gb = total_bytes / (1024**3)
            used_gb = (total_bytes - free_bytes) / (1024**3)
            status_text = f"● CUDA:0 Ready ({dev_name} - {total_gb:.1f} GB)"
            vram_default = f"{used_gb:.1f} / {total_gb:.1f} GB ({(used_gb / total_gb * 100):.1f}%)"
            return status_text, vram_default
    except Exception:
        pass
    try:
        import psutil

        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        used_ram_gb = mem.used / (1024**3)
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        status_text = f"● CPU Engine Ready ({cores} Cores, {total_ram_gb:.1f} GB RAM)"
        vram_default = f"RAM: {used_ram_gb:.1f} / {total_ram_gb:.1f} GB ({mem.percent:.1f}%)"
        return status_text, vram_default
    except Exception:
        pass
    return "● Engine Ready", "—"


class FlightDeckMetricLabel(QLabel):
    """Clean metric label that formats and cleans up telemetry prefixes."""

    def __init__(self, text: str = "—", color: str = "#ffffff", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.color = color
        self.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {color};")

    def setText(self, text: str) -> None:  # noqa: N802
        clean = text
        if clean.startswith("Epoch:"):
            clean = clean.replace("Epoch:", "").strip()
            clean = clean.replace("/", " / ")
        elif clean.startswith("Round:"):
            clean = clean.replace("Round:", "Round ").strip()
            clean = clean.replace("/", " / ")
        elif clean.startswith("Step:"):
            clean = clean.replace("Step:", "").strip()
            clean = clean.replace("/", " / ")
        elif clean.startswith("Loss:"):
            clean = clean.replace("Loss:", "").strip()
        elif clean.startswith("Tokens/sec:"):
            val = clean.replace("Tokens/sec:", "").strip()
            clean = f"{val} tok/s"
        elif clean.startswith("Speed:"):
            val = clean.replace("Speed:", "").strip()
            clean = val
        elif clean.startswith("LR:"):
            clean = clean.replace("LR:", "").strip()
        elif clean.startswith("ETA:"):
            clean = clean.replace("ETA:", "").strip()
        super().setText(clean)


def _build_metric_card(title: str, value_label: QLabel, color: str = "#ffffff") -> QFrame:
    """Create one of the 6 top metric cards matching Concept Image 3."""
    card = QFrame()
    card.setObjectName("MetricCard")
    card.setStyleSheet(
        "QFrame#MetricCard {"
        "  background: #131722;"
        "  border: 1px solid #222738;"
        "  border-radius: 8px;"
        "  padding: 8px 12px;"
        "}"
    )
    vbox = QVBoxLayout(card)
    vbox.setContentsMargins(4, 4, 4, 4)
    vbox.setSpacing(4)

    title_label = QLabel(title)
    title_label.setStyleSheet("color: #64748b; font-size: 10px; font-weight: 800; letter-spacing: 0.8px;")
    vbox.addWidget(title_label)
    vbox.addWidget(value_label)
    return card


def build_live_training_tab(window: Any) -> QWidget:
    """Build the Neural Observatory & Live Training Flight Deck page strictly matching Concept Image 3."""

    hw_status, vram_init_val = _detect_system_hardware()

    page = window._panel()
    outer = QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    scroll = QScrollArea()
    scroll.setObjectName("PageScroll")
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    content = QWidget()
    content.setObjectName("Panel")
    content.setStyleSheet("background-color: #0b0e17;")
    layout = QVBoxLayout(content)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(14)
    scroll.setWidget(content)
    outer.addWidget(scroll, 1)

    # =========================================================================
    # Header: Title + Mode Switcher + Dynamic Status Pill
    # =========================================================================
    header = QHBoxLayout()
    header.setContentsMargins(0, 0, 0, 0)
    header.setSpacing(12)

    title = QLabel("Neural Observatory & Live Training Flight Deck")
    title.setObjectName("PageTitle")
    title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc;")
    header.addWidget(title)

    header.addStretch(1)

    # Perspective Toggle: Local Process vs Cluster Fleet
    mode_container = QFrame()
    mode_container.setObjectName("LiveModeContainer")
    mode_container.setStyleSheet(
        "QFrame#LiveModeContainer {"
        "  background: #141722;"
        "  border: 1px solid #282e42;"
        "  border-radius: 8px;"
        "  padding: 2px;"
        "}"
    )
    mode_layout = QHBoxLayout(mode_container)
    mode_layout.setContentsMargins(2, 2, 2, 2)
    mode_layout.setSpacing(4)

    window.live_mode_local_btn = QPushButton("💻 Local Process")
    window.live_mode_local_btn.setCheckable(True)
    window.live_mode_local_btn.setChecked(True)

    window.live_mode_cluster_btn = QPushButton("🌐 Cluster Fleet")
    window.live_mode_cluster_btn.setCheckable(True)
    window.live_mode_cluster_btn.setChecked(False)

    btn_qss = (
        "QPushButton {"
        "  background: transparent;"
        "  color: #94a3b8;"
        "  border: none;"
        "  border-radius: 6px;"
        "  padding: 4px 12px;"
        "  font-size: 11px;"
        "  font-weight: 700;"
        "}"
        "QPushButton:hover { color: #f8fafc; background: #1e2433; }"
        "QPushButton:checked {"
        "  background: #252b3d;"
        "  color: #f59e0b;"
        "  border: 1px solid #5a3c22;"
        "}"
    )
    window.live_mode_local_btn.setStyleSheet(btn_qss)
    window.live_mode_cluster_btn.setStyleSheet(btn_qss)
    mode_layout.addWidget(window.live_mode_local_btn)
    mode_layout.addWidget(window.live_mode_cluster_btn)
    header.addWidget(mode_container)

    # Dynamic Status Pill (Initialized truthfully to Standby)
    window.live_training_badge = QLabel("○ STANDBY - READY TO TRAIN")
    window.live_training_badge.setObjectName("LiveTrainingBadge")
    window.live_training_badge.setStyleSheet(
        "QLabel {"
        "  background: rgba(148, 163, 184, 0.12);"
        "  color: #94a3b8;"
        "  border: 1px solid #475569;"
        "  border-radius: 12px;"
        "  padding: 4px 12px;"
        "  font-size: 11px;"
        "  font-weight: 800;"
        "}"
    )
    header.addWidget(window.live_training_badge)
    layout.addLayout(header)

    # =========================================================================
    # Top Metrics Strip (6 Distinct Cards matching Concept Image 3)
    # =========================================================================
    metrics_row = QHBoxLayout()
    metrics_row.setSpacing(12)

    # 1. EPOCH / ROUND
    window.live_epoch_metric = FlightDeckMetricLabel("—", "#ffffff")
    card_epoch = _build_metric_card("EPOCH / ROUND", window.live_epoch_metric, "#ffffff")
    metrics_row.addWidget(card_epoch, 1)

    # 2. GLOBAL STEP
    window.live_step_metric = FlightDeckMetricLabel("—", "#38bdf8")
    card_step = _build_metric_card("GLOBAL STEP", window.live_step_metric, "#38bdf8")
    metrics_row.addWidget(card_step, 1)

    # 3. TRAIN LOSS
    window.live_loss_metric = FlightDeckMetricLabel("—", "#f59e0b")
    card_train_loss = _build_metric_card("TRAIN LOSS", window.live_loss_metric, "#f59e0b")
    metrics_row.addWidget(card_train_loss, 1)

    # 4. VAL LOSS
    window.live_val_loss_metric = FlightDeckMetricLabel("—", "#38bdf8")
    card_val_loss = _build_metric_card("VAL LOSS", window.live_val_loss_metric, "#38bdf8")
    metrics_row.addWidget(card_val_loss, 1)

    # 5. THROUGHPUT
    window.live_tokens_metric = FlightDeckMetricLabel("0 tok/s", "#10b981")
    card_throughput = _build_metric_card("THROUGHPUT", window.live_tokens_metric, "#10b981")
    metrics_row.addWidget(card_throughput, 1)

    # 6. ETA REMAINING
    window.live_eta_metric = FlightDeckMetricLabel("—", "#f8fafc")
    card_eta = _build_metric_card("ETA REMAINING", window.live_eta_metric, "#f8fafc")
    metrics_row.addWidget(card_eta, 1)

    layout.addLayout(metrics_row)

    # =========================================================================
    # Main Body: Dual-Column Layout (68% Chart | 32% Health & Diagnostics)
    # =========================================================================
    body = QHBoxLayout()
    body.setSpacing(14)

    # -------------------------------------------------------------------------
    # Left: DYNAMIC LOSS CONVERGENCE (— Train Loss | — Validation Loss)
    # -------------------------------------------------------------------------
    chart_card = QFrame()
    chart_card.setObjectName("Card")
    chart_card.setStyleSheet(
        "QFrame#Card { background: #131722; border: 1px solid #222738; border-radius: 10px; }"
    )
    chart_layout = QVBoxLayout(chart_card)
    chart_layout.setContentsMargins(14, 12, 14, 12)
    chart_layout.setSpacing(8)

    # Title row with color legends matching Concept Image 3
    chart_title_row = QHBoxLayout()
    chart_title_label = QLabel("DYNAMIC LOSS CONVERGENCE")
    chart_title_label.setStyleSheet("font-size: 11px; font-weight: 800; color: #cbd5e1; letter-spacing: 0.8px;")
    chart_legend_label = QLabel(
        "(<span style='color:#f59e0b; font-weight:bold;'>— Train / Global Loss</span> | "
        "<span style='color:#38bdf8; font-weight:bold;'>— Validation Loss</span>)"
    )
    chart_legend_label.setTextFormat(Qt.RichText)
    chart_legend_label.setStyleSheet("font-size: 11px;")
    chart_title_row.addWidget(chart_title_label)
    chart_title_row.addWidget(chart_legend_label)
    chart_title_row.addStretch(1)
    chart_layout.addLayout(chart_title_row)

    # Loss Chart Widget
    window.loss_chart = LossChartWidget(
        primary_label="Train Loss",
        secondary_label="Validation Loss",
        empty_text="Loss convergence telemetry will stream here during training",
        title="",
        y_label="Loss",
    )
    window.loss_chart.setMinimumHeight(380)
    try:
        import pyqtgraph as pg

        window.loss_chart.primary_curve.setPen(pg.mkPen("#f59e0b", width=2.4))
        window.loss_chart.secondary_curve.setPen(pg.mkPen("#38bdf8", width=2.4))
        window.loss_chart.plot.setBackground("#0b0e17")
        window.loss_chart.plot.showGrid(x=True, y=True, alpha=0.18)
    except Exception:
        pass
    chart_layout.addWidget(window.loss_chart, 1)

    body.addWidget(chart_card, 68)

    # -------------------------------------------------------------------------
    # Right: Dual Perspective Diagnostics (Local Health vs Cluster Fleet)
    # -------------------------------------------------------------------------
    right_column = QVBoxLayout()
    right_column.setSpacing(14)

    def _add_health_row(grid: QGridLayout, row_idx: int, label_text: str, default_val: str, color: str) -> QLabel:
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        val = QLabel(default_val)
        val.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 700;")
        grid.addWidget(lbl, row_idx, 0, Qt.AlignLeft)
        grid.addWidget(val, row_idx, 1, Qt.AlignRight)
        return val

    # ----------------- LOCAL PERSPECTIVE: HEALTH CARD ------------------------
    window.live_local_health_card = QFrame()
    window.live_local_health_card.setObjectName("Card")
    window.live_local_health_card.setStyleSheet(
        "QFrame#Card { background: #131722; border: 1px solid #222738; border-radius: 10px; }"
    )
    local_health_layout = QVBoxLayout(window.live_local_health_card)
    local_health_layout.setContentsMargins(14, 12, 14, 12)
    local_health_layout.setSpacing(10)

    local_health_title = QLabel("LOCAL TRAINING HEALTH & STABILITY")
    local_health_title.setStyleSheet("font-size: 11px; font-weight: 800; color: #94a3b8; letter-spacing: 0.8px;")
    local_health_layout.addWidget(local_health_title)

    local_health_grid = QGridLayout()
    local_health_grid.setSpacing(8)
    local_health_grid.setContentsMargins(2, 4, 2, 4)

    window.live_lr_metric = _add_health_row(local_health_grid, 0, "Learning Rate:", "—", "#38bdf8")
    window.live_grad_norm_metric = _add_health_row(local_health_grid, 1, "Gradient Norm:", "—", "#10b981")
    window.live_vram_metric = _add_health_row(local_health_grid, 2, "VRAM Consumption:", vram_init_val, "#f59e0b")
    window.live_early_stop_metric = _add_health_row(local_health_grid, 3, "Early Stopping:", "Standby (Awaiting run)", "#a855f7")

    local_health_layout.addLayout(local_health_grid)
    right_column.addWidget(window.live_local_health_card, 0)

    # ----------------- LOCAL PERSPECTIVE: PROBE CARD -------------------------
    window.live_local_probe_card = QFrame()
    window.live_local_probe_card.setObjectName("Card")
    window.live_local_probe_card.setStyleSheet(
        "QFrame#Card { background: #11141e; border: 1px solid #1e2638; border-radius: 8px; }"
    )
    probe_layout = QVBoxLayout(window.live_local_probe_card)
    probe_layout.setContentsMargins(12, 10, 12, 10)
    probe_layout.setSpacing(6)

    probe_header = QLabel("Checkpoint Probe (Sample Generation):")
    probe_header.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 700;")
    probe_layout.addWidget(probe_header)

    window.live_probe_prompt = QLabel("Prompt: Awaiting training checkpoint sample generation...")
    window.live_probe_prompt.setStyleSheet("color: #f1f5f9; font-size: 11px; font-family: 'Consolas', monospace;")
    window.live_probe_prompt.setWordWrap(True)
    probe_layout.addWidget(window.live_probe_prompt)

    window.live_probe_output = QLabel("Output: (No checkpoint evaluated yet)")
    window.live_probe_output.setStyleSheet("color: #38bdf8; font-size: 11px; font-family: 'Consolas', monospace;")
    window.live_probe_output.setWordWrap(True)
    probe_layout.addWidget(window.live_probe_output)

    probe_layout.addStretch(1)
    right_column.addWidget(window.live_local_probe_card, 1)

    # ----------------- CLUSTER PERSPECTIVE: HEALTH CARD ----------------------
    window.live_cluster_health_card = QFrame()
    window.live_cluster_health_card.setObjectName("Card")
    window.live_cluster_health_card.setStyleSheet(
        "QFrame#Card { background: #131722; border: 1px solid #222738; border-radius: 10px; }"
    )
    cluster_health_layout = QVBoxLayout(window.live_cluster_health_card)
    cluster_health_layout.setContentsMargins(14, 12, 14, 12)
    cluster_health_layout.setSpacing(10)

    cluster_health_title = QLabel("CLUSTER FLEET HEALTH & SYNC")
    cluster_health_title.setStyleSheet("font-size: 11px; font-weight: 800; color: #38bdf8; letter-spacing: 0.8px;")
    cluster_health_layout.addWidget(cluster_health_title)

    cluster_health_grid = QGridLayout()
    cluster_health_grid.setSpacing(8)
    cluster_health_grid.setContentsMargins(2, 4, 2, 4)

    window.live_cluster_nodes_metric = _add_health_row(cluster_health_grid, 0, "Online Fleet Nodes:", "Waiting for workers...", "#38bdf8")
    window.live_cluster_vram_metric = _add_health_row(cluster_health_grid, 1, "Fleet Aggregate VRAM:", "—", "#f59e0b")
    window.live_cluster_sync_metric = _add_health_row(cluster_health_grid, 2, "Sync Mechanism:", "Local SGD (250 steps/round)", "#10b981")
    window.live_cluster_consensus_metric = _add_health_row(cluster_health_grid, 3, "Consensus Health:", "Standby (Idle)", "#a855f7")

    cluster_health_layout.addLayout(cluster_health_grid)
    right_column.addWidget(window.live_cluster_health_card, 0)
    window.live_cluster_health_card.setVisible(False)

    # ----------------- CLUSTER PERSPECTIVE: WORKER FLEET CARD ----------------
    window.live_cluster_fleet_card = QFrame()
    window.live_cluster_fleet_card.setObjectName("Card")
    window.live_cluster_fleet_card.setStyleSheet(
        "QFrame#Card { background: #11141e; border: 1px solid #1e2638; border-radius: 8px; }"
    )
    fleet_card_layout = QVBoxLayout(window.live_cluster_fleet_card)
    fleet_card_layout.setContentsMargins(12, 10, 12, 10)
    fleet_card_layout.setSpacing(6)

    fleet_card_header = QLabel("Cluster Worker Fleet Telemetry:")
    fleet_card_header.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 700;")
    fleet_card_layout.addWidget(fleet_card_header)

    fleet_scroll = QScrollArea()
    fleet_scroll.setWidgetResizable(True)
    fleet_scroll.setStyleSheet("background: transparent; border: none;")
    fleet_scroll_content = QWidget()
    fleet_scroll_content.setStyleSheet("background: transparent;")
    window.live_cluster_workers_layout = QVBoxLayout(fleet_scroll_content)
    window.live_cluster_workers_layout.setContentsMargins(0, 0, 0, 0)
    window.live_cluster_workers_layout.setSpacing(4)

    window.live_cluster_workers_placeholder = QLabel("No active workers connected yet. Start worker nodes to view distributed telemetry.")
    window.live_cluster_workers_placeholder.setStyleSheet("color: #64748b; font-size: 11px; font-style: italic;")
    window.live_cluster_workers_placeholder.setWordWrap(True)
    window.live_cluster_workers_layout.addWidget(window.live_cluster_workers_placeholder)
    window.live_cluster_workers_layout.addStretch(1)

    fleet_scroll.setWidget(fleet_scroll_content)
    fleet_card_layout.addWidget(fleet_scroll, 1)

    right_column.addWidget(window.live_cluster_fleet_card, 1)
    window.live_cluster_fleet_card.setVisible(False)

    def update_live_cluster_workers(worker_losses: dict[str, Any], workers_count: int) -> None:
        """Dynamically render the cluster worker fleet telemetry rows."""
        # Clear previous items
        while window.live_cluster_workers_layout.count() > 0:
            item = window.live_cluster_workers_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not worker_losses and workers_count == 0:
            lbl = QLabel("No active workers connected yet. Start worker nodes to view distributed telemetry.")
            lbl.setStyleSheet("color: #64748b; font-size: 11px; font-style: italic;")
            lbl.setWordWrap(True)
            window.live_cluster_workers_layout.addWidget(lbl)
            window.live_cluster_workers_layout.addStretch(1)
            return

        for w_id, loss_val in sorted(worker_losses.items()):
            row = QFrame()
            row.setStyleSheet(
                "QFrame { background: #161b26; border: 1px solid #252e42; border-radius: 6px; padding: 4px 8px; }"
            )
            r_box = QHBoxLayout(row)
            r_box.setContentsMargins(6, 4, 6, 4)
            r_box.setSpacing(8)

            w_lbl = QLabel(f"Node: {w_id}")
            w_lbl.setStyleSheet("color: #f1f5f9; font-weight: 700; font-size: 11px;")
            r_box.addWidget(w_lbl)
            r_box.addStretch(1)

            l_lbl = QLabel(f"Loss: {float(loss_val):.4f}")
            l_lbl.setStyleSheet("color: #f59e0b; font-weight: 700; font-size: 11px;")
            r_box.addWidget(l_lbl)

            status_dot = QLabel("● Online")
            status_dot.setStyleSheet("color: #10b981; font-weight: 700; font-size: 10px;")
            r_box.addWidget(status_dot)

            window.live_cluster_workers_layout.addWidget(row)

        window.live_cluster_workers_layout.addStretch(1)

    window.update_live_cluster_workers = update_live_cluster_workers

    body.addLayout(right_column, 32)
    layout.addLayout(body, 1)

    # =========================================================================
    # Bottom Action Bar & Status matching Concept Image 3
    # =========================================================================
    bottom_row = QHBoxLayout()
    bottom_row.setContentsMargins(0, 4, 0, 0)
    bottom_row.setSpacing(10)

    window.pause_training_button = QPushButton("Pause Training")
    window.pause_training_button.setStyleSheet(
        "QPushButton {"
        "  background: #151821;"
        "  color: #f8fafc;"
        "  border: 1px solid #282e42;"
        "  border-radius: 6px;"
        "  padding: 8px 16px;"
        "  font-weight: 700;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover { background: #222738; border-color: #8b5cf6; }"
    )
    bottom_row.addWidget(window.pause_training_button)

    window.save_resumable_checkpoint_button = QPushButton("Save Resumable Checkpoint")
    window.save_resumable_checkpoint_button.setStyleSheet(
        "QPushButton {"
        "  background: #151821;"
        "  color: #f8fafc;"
        "  border: 1px solid #282e42;"
        "  border-radius: 6px;"
        "  padding: 8px 16px;"
        "  font-weight: 700;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover { background: #222738; border-color: #38bdf8; }"
    )
    bottom_row.addWidget(window.save_resumable_checkpoint_button)

    bottom_row.addStretch(1)

    # Emergency Graceful Stop (Vibrant red pill button)
    window.stop_training_button = QPushButton("Emergency Graceful Stop")
    window.stop_training_button.setObjectName("EmergencyStopButton")
    window.stop_training_button.setStyleSheet(
        "QPushButton {"
        "  background: #ef4444;"
        "  color: #ffffff;"
        "  border: none;"
        "  border-radius: 8px;"
        "  padding: 9px 20px;"
        "  font-weight: 800;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover { background: #dc2626; }"
        "QPushButton:pressed { background: #b91c1c; }"
    )
    bottom_row.addWidget(window.stop_training_button)
    layout.addLayout(bottom_row)

    # Footer status (Shows real detected hardware)
    footer_row = QHBoxLayout()
    footer_row.setContentsMargins(0, 0, 0, 0)
    window.live_cuda_status = QLabel(hw_status)
    window.live_cuda_status.setStyleSheet("color: #10b981; font-size: 11px; font-weight: 700;")
    footer_row.addWidget(window.live_cuda_status)
    footer_row.addStretch(1)
    layout.addLayout(footer_row)

    # Mode switcher clicks
    window.live_mode_local_btn.clicked.connect(lambda: window._switch_live_telemetry_mode("local") if hasattr(window, "_switch_live_telemetry_mode") else None)
    window.live_mode_cluster_btn.clicked.connect(lambda: window._switch_live_telemetry_mode("cluster") if hasattr(window, "_switch_live_telemetry_mode") else None)

    # Wire bottom actions
    def _on_pause_clicked() -> None:
        active_mode = getattr(window, "_active_live_mode", "local")
        if active_mode == "cluster":
            if hasattr(window, "pause_cluster_job"):
                window.pause_cluster_job()
        else:
            if hasattr(window, "pause_training"):
                window.pause_training()
            elif hasattr(window, "training_controller"):
                window.training_controller.pause()

    def _on_save_checkpoint_clicked() -> None:
        if hasattr(window, "save_resumable_checkpoint"):
            window.save_resumable_checkpoint()
        elif hasattr(window, "training_controller"):
            window.training_controller.checkpoint()

    def _on_stop_clicked() -> None:
        active_mode = getattr(window, "_active_live_mode", "local")
        if active_mode == "cluster":
            if hasattr(window, "stop_cluster_job"):
                window.stop_cluster_job()
            elif hasattr(window, "stop_training"):
                window.stop_training()
        else:
            if hasattr(window, "stop_training"):
                window.stop_training()
            elif hasattr(window, "training_controller"):
                window.training_controller.stop()

    window.pause_training_button.clicked.connect(_on_pause_clicked)
    window.save_resumable_checkpoint_button.clicked.connect(_on_save_checkpoint_clicked)
    window.stop_training_button.clicked.connect(_on_stop_clicked)

    # =========================================================================
    # Hidden / Auxiliary compatibility widgets
    # (Keeps all existing telemetry calls working with zero attribute errors)
    # =========================================================================
    aux_frame = QWidget()
    aux_frame.setVisible(False)
    aux_layout = QVBoxLayout(aux_frame)

    window.live_data_metric = QLabel("Data: 0%")
    window.live_sample_text = window.live_probe_prompt
    window.live_flow = ModelFlowWidget()
    window.live_model_status = QLabel("Model: Transformer")
    window.live_layer_status = QLabel("Layers: -")
    window.live_head_status = QLabel("Heads: -")
    window.live_hidden_status = QLabel("Hidden: -")
    window.live_batch_status = QLabel("Batch: -")
    window.live_context_status = QLabel("Context: -")
    window.live_device_status = QLabel("Device: CUDA:0")
    window.live_worker_status = QLabel("Workers: 4")

    window.live_cpu_label = QLabel("CPU: -")
    window.live_cpu_bar = window._hardware_meter("System CPU")
    window.live_training_cpu_label = QLabel("Train CPU: -")
    window.live_training_cpu_bar = window._hardware_meter("Train CPU")
    window.live_ui_cpu_label = QLabel("UI CPU: -")
    window.live_ui_cpu_bar = window._hardware_meter("UI CPU")
    window.live_gpu_label = QLabel("GPU: -")
    window.live_gpu_bar = window._hardware_meter("GPU")
    window.live_vram_label = window.live_vram_metric
    window.live_vram_bar = window._hardware_meter("VRAM")
    window.live_ram_label = QLabel("RAM: -")
    window.live_ram_bar = window._hardware_meter("RAM")

    window.optimization_chart = LossChartWidget("LR", "Grad Norm")
    window.stability_chart = LossChartWidget("Weight", "Update")
    window.throughput_chart = LossChartWidget("Tokens/s", "Samples/s")
    window.memory_chart = LossChartWidget("Alloc", "Reserved")

    window.live_prediction_chart = LiveDistributionWidget()
    window.live_attention_chart = LiveHeatmapWidget()
    window.live_activation_chart = LiveHistogramWidget()
    window.live_gradient_chart = LiveGradientFlowWidget()

    window.live_time_slider = QSlider(Qt.Horizontal)
    window.live_time_slider.setRange(0, 0)
    window.live_time_slider.setValue(0)
    window.live_timeline_label = QLabel("Timeline: live")
    window.live_progress = QProgressBar()

    window.resume_training_button = window.pause_training_button
    window.save_checkpoint_button = window.save_resumable_checkpoint_button

    for w in (
        window.live_flow,
        window.optimization_chart,
        window.stability_chart,
        window.throughput_chart,
        window.memory_chart,
        window.live_prediction_chart,
        window.live_attention_chart,
        window.live_activation_chart,
        window.live_gradient_chart,
    ):
        aux_layout.addWidget(w)
    layout.addWidget(aux_frame)

    return page
