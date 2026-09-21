"""Neural Observatory & Live Training Flight Deck strictly matching Concept Image 3."""

from __future__ import annotations

import re
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


class FlightDeckMetricLabel(QLabel):
    """Clean metric label that formats and cleans up telemetry prefixes."""

    def __init__(self, text: str = "-", color: str = "#ffffff", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.color = color
        self.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {color};")

    def setText(self, text: str) -> None:  # noqa: N802
        clean = text
        if clean.startswith("Epoch:"):
            clean = clean.replace("Epoch:", "").strip()
            clean = clean.replace("/", " / ")
        elif clean.startswith("Step:"):
            clean = clean.replace("Step:", "").strip()
            clean = clean.replace("/", " / ")
        elif clean.startswith("Loss:"):
            clean = clean.replace("Loss:", "").strip()
        elif clean.startswith("Tokens/sec:"):
            val = clean.replace("Tokens/sec:", "").strip()
            clean = f"{val} tok/s"
        elif clean.startswith("LR:"):
            clean = clean.replace("LR:", "").strip()
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
    # Header: Title + Status Pill
    # =========================================================================
    header = QHBoxLayout()
    header.setContentsMargins(0, 0, 0, 0)
    header.setSpacing(12)

    title = QLabel("Neural Observatory & Live Training Flight Deck")
    title.setObjectName("PageTitle")
    title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc;")
    header.addWidget(title)

    header.addStretch(1)

    # Status Pill matching Concept Image 3
    window.live_training_badge = QLabel("● TRAINING ACTIVE (PID: 18492)")
    window.live_training_badge.setObjectName("LiveTrainingBadge")
    window.live_training_badge.setStyleSheet(
        "QLabel {"
        "  background: rgba(16, 185, 129, 0.15);"
        "  color: #10b981;"
        "  border: 1px solid #10b981;"
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

    # 1. EPOCH
    window.live_epoch_metric = FlightDeckMetricLabel("3 / 10", "#ffffff")
    card_epoch = _build_metric_card("EPOCH", window.live_epoch_metric, "#ffffff")
    metrics_row.addWidget(card_epoch, 1)

    # 2. GLOBAL STEP
    window.live_step_metric = FlightDeckMetricLabel("4,520 / 12,000", "#38bdf8")
    card_step = _build_metric_card("GLOBAL STEP", window.live_step_metric, "#38bdf8")
    metrics_row.addWidget(card_step, 1)

    # 3. TRAIN LOSS
    window.live_loss_metric = FlightDeckMetricLabel("1.842", "#f59e0b")
    card_train_loss = _build_metric_card("TRAIN LOSS", window.live_loss_metric, "#f59e0b")
    metrics_row.addWidget(card_train_loss, 1)

    # 4. VAL LOSS
    window.live_val_loss_metric = FlightDeckMetricLabel("1.905", "#38bdf8")
    card_val_loss = _build_metric_card("VAL LOSS", window.live_val_loss_metric, "#38bdf8")
    metrics_row.addWidget(card_val_loss, 1)

    # 5. THROUGHPUT
    window.live_tokens_metric = FlightDeckMetricLabel("14,200 tok/s", "#10b981")
    card_throughput = _build_metric_card("THROUGHPUT", window.live_tokens_metric, "#10b981")
    metrics_row.addWidget(card_throughput, 1)

    # 6. ETA REMAINING
    window.live_eta_metric = FlightDeckMetricLabel("1h 42m", "#f8fafc")
    card_eta = _build_metric_card("ETA REMAINING", window.live_eta_metric, "#f8fafc")
    metrics_row.addWidget(card_eta, 1)

    layout.addLayout(metrics_row)

    # =========================================================================
    # Main Body: Dual-Column Layout (68% Chart | 32% Health & Checkpoint Probe)
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
    chart_legend_label = QLabel("(<span style='color:#f59e0b; font-weight:bold;'>— Train Loss</span> | <span style='color:#38bdf8; font-weight:bold;'>— Validation Loss</span>)")
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
    # Right: TRAINING HEALTH & STABILITY + Checkpoint Probe
    # -------------------------------------------------------------------------
    right_column = QVBoxLayout()
    right_column.setSpacing(14)

    # Card 1: TRAINING HEALTH & STABILITY
    health_card = QFrame()
    health_card.setObjectName("Card")
    health_card.setStyleSheet(
        "QFrame#Card { background: #131722; border: 1px solid #222738; border-radius: 10px; }"
    )
    health_layout = QVBoxLayout(health_card)
    health_layout.setContentsMargins(14, 12, 14, 12)
    health_layout.setSpacing(10)

    health_title = QLabel("TRAINING HEALTH & STABILITY")
    health_title.setStyleSheet("font-size: 11px; font-weight: 800; color: #94a3b8; letter-spacing: 0.8px;")
    health_layout.addWidget(health_title)

    health_grid = QGridLayout()
    health_grid.setSpacing(8)
    health_grid.setContentsMargins(2, 4, 2, 4)

    def _add_health_row(row_idx: int, label_text: str, default_val: str, color: str) -> QLabel:
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        val = QLabel(default_val)
        val.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 700;")
        health_grid.addWidget(lbl, row_idx, 0, Qt.AlignLeft)
        health_grid.addWidget(val, row_idx, 1, Qt.AlignRight)
        return val

    window.live_lr_metric = _add_health_row(0, "Learning Rate:", "0.000284 (Warmup-Cosine)", "#38bdf8")
    window.live_grad_norm_metric = _add_health_row(1, "Gradient Norm:", "0.842 (Optimal clipping < 1.0)", "#10b981")
    window.live_vram_metric = _add_health_row(2, "VRAM Consumption:", "11.8 / 24.0 GB (49.1%)", "#f59e0b")
    window.live_early_stop_metric = _add_health_row(3, "Early Stopping:", "Healthy (Patience 3/5)", "#a855f7")

    health_layout.addLayout(health_grid)
    right_column.addWidget(health_card, 0)

    # Card 2: Checkpoint Probe (Step 4,500 Sample Generation)
    probe_card = QFrame()
    probe_card.setObjectName("Card")
    probe_card.setStyleSheet(
        "QFrame#Card { background: #11141e; border: 1px solid #1e2638; border-radius: 8px; }"
    )
    probe_layout = QVBoxLayout(probe_card)
    probe_layout.setContentsMargins(12, 10, 12, 10)
    probe_layout.setSpacing(6)

    probe_header = QLabel("Checkpoint Probe (Step 4,500 Sample Generation):")
    probe_header.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 700;")
    probe_layout.addWidget(probe_header)

    window.live_probe_prompt = QLabel("Prompt: 'The derivative of sin(x) with respect to x is'")
    window.live_probe_prompt.setStyleSheet("color: #f1f5f9; font-size: 11px; font-family: 'Consolas', monospace;")
    window.live_probe_prompt.setWordWrap(True)
    probe_layout.addWidget(window.live_probe_prompt)

    window.live_probe_output = QLabel("Output: 'cos(x). Proof: by the definition of the derivative...'")
    window.live_probe_output.setStyleSheet("color: #38bdf8; font-size: 11px; font-family: 'Consolas', monospace;")
    window.live_probe_output.setWordWrap(True)
    probe_layout.addWidget(window.live_probe_output)

    probe_layout.addStretch(1)
    right_column.addWidget(probe_card, 1)

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

    # Footer status
    footer_row = QHBoxLayout()
    footer_row.setContentsMargins(0, 0, 0, 0)
    window.live_cuda_status = QLabel("● CUDA:0 Ready (24GB)")
    window.live_cuda_status.setStyleSheet("color: #10b981; font-size: 11px; font-weight: 700;")
    footer_row.addWidget(window.live_cuda_status)
    footer_row.addStretch(1)
    layout.addLayout(footer_row)

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
