"""Compute & Runtime Engine tab strictly matching Reference Image 2."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from interface.widgets.circular_gauge import CircularGaugeWidget


def _create_toggle_item(
    title: str,
    subtitle: str,
    initial_checked: bool = True,
) -> tuple[QWidget, QCheckBox]:
    """Create a row containing an iOS-style neon green toggle with title & subtitle."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 4, 0, 4)
    row.setSpacing(12)

    text_box = QVBoxLayout()
    text_box.setContentsMargins(0, 0, 0, 0)
    text_box.setSpacing(2)

    title_label = QLabel(title)
    title_label.setStyleSheet("color: #f8fafc; font-size: 13px; font-weight: 700;")
    sub_label = QLabel(subtitle)
    sub_label.setStyleSheet("color: #64748b; font-size: 11px;")
    text_box.addWidget(title_label)
    text_box.addWidget(sub_label)

    toggle = QCheckBox()
    toggle.setChecked(initial_checked)
    toggle.setStyleSheet(
        "QCheckBox::indicator {"
        "  width: 38px;"
        "  height: 20px;"
        "  border-radius: 10px;"
        "  background-color: #1e2230;"
        "  border: 1px solid #333a4d;"
        "}"
        "QCheckBox::indicator:checked {"
        "  background-color: #10b981;"
        "  border-color: #34d399;"
        "}"
    )

    row.addLayout(text_box, 1)
    row.addWidget(toggle)
    return container, toggle


def build_compute_engine_tab(window: Any) -> QWidget:
    """Build the Compute & Runtime Engine page.

    Provides Hardware & Device Telemetry, VRAM & Memory Optimization,
    and Execution Runtime & Cluster Fleet strictly matching Reference Image 2.

    Args:
        window: Main application window holding shared state.

    Returns:
        Configured Compute & Runtime Engine page widget.
    """
    page = window._panel()
    root = QVBoxLayout(page)
    root.setContentsMargins(20, 16, 20, 16)
    root.setSpacing(14)

    # =========================================================================
    # Header: Title + Subtitle + Settings Button
    # =========================================================================
    header_row = QHBoxLayout()
    header_row.setContentsMargins(0, 0, 0, 0)

    title_box = QVBoxLayout()
    title_box.setContentsMargins(0, 0, 0, 0)
    title_box.setSpacing(2)

    title_label = QLabel("Compute & Runtime Engine")
    title_label.setObjectName("PageTitle")
    title_label.setStyleSheet("font-size: 22px; font-weight: 800; color: #f8fafc;")

    sub_label = QLabel("At AI LLM Integrated Development Environment software")
    sub_label.setStyleSheet("color: #64748b; font-size: 12px;")
    title_box.addWidget(title_label)
    title_box.addWidget(sub_label)

    header_row.addLayout(title_box, 1)

    settings_btn = QPushButton("⚙ Settings")
    settings_btn.setStyleSheet(
        "QPushButton { background: #151821; color: #cbd5e1; border: 1px solid #232738; border-radius: 8px; padding: 6px 14px; font-size: 12px; font-weight: 600; }"
        "QPushButton:hover { background: #1e2230; color: #ffffff; border-color: #10b981; }"
    )
    if hasattr(window, "open_plugins_dialog"):
        settings_btn.clicked.connect(window.open_plugins_dialog)
    header_row.addWidget(settings_btn)

    root.addLayout(header_row)

    # =========================================================================
    # Three Main Cards (Telemetry | Memory Optimization | Runtime & Cluster)
    # =========================================================================
    cards_row = QHBoxLayout()
    cards_row.setSpacing(14)

    # -------------------------------------------------------------------------
    # Card 1: Hardware & Device Telemetry [ 1 ]
    # -------------------------------------------------------------------------
    card1 = QFrame()
    card1.setObjectName("Card")
    card1.setStyleSheet(
        "QFrame#Card { background: #11141c; border: 1px solid #1a2727; border-radius: 12px; }"
    )
    card1_layout = QVBoxLayout(card1)
    card1_layout.setContentsMargins(16, 14, 16, 14)
    card1_layout.setSpacing(12)

    card1_header_row = QHBoxLayout()
    c1_title = QLabel("Hardware & Device Telemetry")
    c1_title.setStyleSheet("color: #e2e8f0; font-size: 13px; font-weight: 800;")
    c1_badge = QLabel("1")
    c1_badge.setStyleSheet("color: #10b981; background: #064e3b; border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: bold;")
    card1_header_row.addWidget(c1_title)
    card1_header_row.addStretch(1)
    card1_header_row.addWidget(c1_badge)
    card1_layout.addLayout(card1_header_row)

    # 4 Circular Radial Gauges in a 2x2 grid
    gauges_top = QHBoxLayout()
    window.gauge_vram = CircularGaugeWidget(title="VRAM", value_text="6.2 / 12", unit_text="NVIDIA", percent=52.0, accent_color="#10b981")
    window.gauge_cuda = CircularGaugeWidget(title="CUDA", value_text="32", unit_text="°C", percent=32.0, accent_color="#06b6d4")
    gauges_top.addWidget(window.gauge_vram)
    gauges_top.addWidget(window.gauge_cuda)
    card1_layout.addLayout(gauges_top)

    gauges_bottom = QHBoxLayout()
    window.gauge_temp = CircularGaugeWidget(title="CUDA temperature", value_text="70%", percent=70.0, accent_color="#06b6d4")
    window.gauge_throughput = CircularGaugeWidget(title="Compute throughput", value_text="70%", percent=70.0, accent_color="#06b6d4")
    gauges_bottom.addWidget(window.gauge_temp)
    gauges_bottom.addWidget(window.gauge_throughput)
    card1_layout.addLayout(gauges_bottom)

    # Compute throughput horizontal progress bar
    tp_label_row = QHBoxLayout()
    tp_text = QLabel("Compute throughput (%)")
    tp_text.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 600;")
    tp_val = QLabel("80%")
    tp_val.setStyleSheet("color: #10b981; font-size: 11px; font-weight: 700;")
    tp_label_row.addWidget(tp_text)
    tp_label_row.addStretch(1)
    tp_label_row.addWidget(tp_val)
    card1_layout.addLayout(tp_label_row)

    throughput_bar = QProgressBar()
    throughput_bar.setRange(0, 100)
    throughput_bar.setValue(80)
    throughput_bar.setTextVisible(False)
    throughput_bar.setFixedHeight(12)
    throughput_bar.setStyleSheet(
        "QProgressBar { background: #141722; border: 1px solid #1e2433; border-radius: 6px; }"
        "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #06b6d4, stop:1 #10b981); border-radius: 5px; }"
    )
    card1_layout.addWidget(throughput_bar)

    # Device Selector: [ CUDA:0 ] | [ CPU ]
    dev_title = QLabel("Device")
    dev_title.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 600;")
    card1_layout.addWidget(dev_title)

    dev_row = QHBoxLayout()
    dev_row.setSpacing(8)

    window.device_cuda_btn = QPushButton("CUDA:0")
    window.device_cuda_btn.setCheckable(True)
    window.device_cuda_btn.setChecked(True)
    window.device_cuda_btn.setStyleSheet(
        "QPushButton { background: #064e3b; color: #34d399; border: 1px solid #059669; border-radius: 6px; padding: 6px; font-weight: 800; font-size: 12px; }"
        "QPushButton:!checked { background: #141722; color: #64748b; border-color: #1e2433; }"
    )

    window.device_cpu_btn = QPushButton("CPU")
    window.device_cpu_btn.setCheckable(True)
    window.device_cpu_btn.setChecked(False)
    window.device_cpu_btn.setStyleSheet(
        "QPushButton { background: #064e3b; color: #34d399; border: 1px solid #059669; border-radius: 6px; padding: 6px; font-weight: 800; font-size: 12px; }"
        "QPushButton:!checked { background: #141722; color: #64748b; border-color: #1e2433; }"
    )

    def on_device_select(dev: str) -> None:
        window.device_cuda_btn.setChecked(dev == "cuda")
        window.device_cpu_btn.setChecked(dev == "cpu")

    window.device_cuda_btn.clicked.connect(lambda: on_device_select("cuda"))
    window.device_cpu_btn.clicked.connect(lambda: on_device_select("cpu"))

    dev_row.addWidget(window.device_cuda_btn, 1)
    dev_row.addWidget(window.device_cpu_btn, 1)
    card1_layout.addLayout(dev_row)

    cards_row.addWidget(card1, 1)

    # -------------------------------------------------------------------------
    # Card 2: VRAM & Memory Optimization [ 2 ]
    # -------------------------------------------------------------------------
    card2 = QFrame()
    card2.setObjectName("Card")
    card2.setStyleSheet(
        "QFrame#Card { background: #11141c; border: 1px solid #1a2727; border-radius: 12px; }"
    )
    card2_layout = QVBoxLayout(card2)
    card2_layout.setContentsMargins(16, 14, 16, 14)
    card2_layout.setSpacing(12)

    card2_header_row = QHBoxLayout()
    c2_title = QLabel("VRAM & Memory Optimization")
    c2_title.setStyleSheet("color: #e2e8f0; font-size: 13px; font-weight: 800;")
    c2_badge = QLabel("2")
    c2_badge.setStyleSheet("color: #10b981; background: #064e3b; border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: bold;")
    card2_header_row.addWidget(c2_title)
    card2_header_row.addStretch(1)
    card2_header_row.addWidget(c2_badge)
    card2_layout.addLayout(card2_header_row)

    # 5 iOS-style neon green toggle switches matching Reference Image 2
    row1, toggle_act_ckp = _create_toggle_item("Activation Checkpointing", "Continues activation checkpointing", True)
    row2, toggle_adam8 = _create_toggle_item("8-Bit AdamW", "Commit themes at 8-Bit adamW", True)
    row3, toggle_mp = _create_toggle_item("Mixed Precision (FP16/BF16)", "Optinat mixed memory for (FP16/BF16)", True)
    row4, toggle_inductor = _create_toggle_item("Torch Inductor Compiler", "Compilers compiler torch Inductor execution", True)
    row5, toggle_offload = _create_toggle_item("CPU offload workers", "Offload workers. CPU offload workers", True)

    card2_layout.addWidget(row1)
    card2_layout.addWidget(row2)
    card2_layout.addWidget(row3)
    card2_layout.addWidget(row4)
    card2_layout.addWidget(row5)

    # Synchronize toggles with window attributes
    if hasattr(window, "activation_checkpointing"):
        toggle_act_ckp.toggled.connect(window.activation_checkpointing.setChecked)
    if hasattr(window, "optimizer_8bit"):
        toggle_adam8.toggled.connect(window.optimizer_8bit.setChecked)
    if hasattr(window, "mixed_precision"):
        toggle_mp.toggled.connect(window.mixed_precision.setChecked)
    if hasattr(window, "compile_model"):
        toggle_inductor.toggled.connect(window.compile_model.setChecked)
    if hasattr(window, "cpu_offload"):
        toggle_offload.toggled.connect(window.cpu_offload.setChecked)

    card2_layout.addStretch(1)
    cards_row.addWidget(card2, 1)

    # -------------------------------------------------------------------------
    # Card 3: Execution Runtime & Cluster Fleet [ 3 ]
    # -------------------------------------------------------------------------
    card3 = QFrame()
    card3.setObjectName("Card")
    card3.setStyleSheet(
        "QFrame#Card { background: #11141c; border: 1px solid #1a2727; border-radius: 12px; }"
    )
    card3_layout = QVBoxLayout(card3)
    card3_layout.setContentsMargins(16, 14, 16, 14)
    card3_layout.setSpacing(12)

    card3_header_row = QHBoxLayout()
    c3_title = QLabel("Execution Runtime & Cluster Fleet")
    c3_title.setStyleSheet("color: #e2e8f0; font-size: 13px; font-weight: 800;")
    c3_badge = QLabel("3")
    c3_badge.setStyleSheet("color: #10b981; background: #064e3b; border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: bold;")
    card3_header_row.addWidget(c3_title)
    card3_header_row.addStretch(1)
    card3_header_row.addWidget(c3_badge)
    card3_layout.addLayout(card3_header_row)

    # Launch target: [ Local machine ] | [ Cluster Local SGD ]
    lt_title = QLabel("Launch target")
    lt_title.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 600;")
    card3_layout.addWidget(lt_title)

    target_row = QHBoxLayout()
    target_row.setSpacing(8)

    window.target_local_btn = QPushButton("Local machine")
    window.target_local_btn.setCheckable(True)
    window.target_local_btn.setChecked(True)
    window.target_local_btn.setStyleSheet(
        "QPushButton { background: #064e3b; color: #34d399; border: 1px solid #059669; border-radius: 6px; padding: 7px; font-weight: 800; font-size: 11px; }"
        "QPushButton:!checked { background: #141722; color: #64748b; border-color: #1e2433; }"
    )

    window.target_cluster_btn = QPushButton("Cluster Local SGD")
    window.target_cluster_btn.setCheckable(True)
    window.target_cluster_btn.setChecked(False)
    window.target_cluster_btn.setStyleSheet(
        "QPushButton { background: #064e3b; color: #34d399; border: 1px solid #059669; border-radius: 6px; padding: 7px; font-weight: 800; font-size: 11px; }"
        "QPushButton:!checked { background: #141722; color: #64748b; border-color: #1e2433; }"
    )

    def on_target_select(tgt: str) -> None:
        window.target_local_btn.setChecked(tgt == "local")
        window.target_cluster_btn.setChecked(tgt == "cluster")

    window.target_local_btn.clicked.connect(lambda: on_target_select("local"))
    window.target_cluster_btn.clicked.connect(lambda: on_target_select("cluster"))

    target_row.addWidget(window.target_local_btn, 1)
    target_row.addWidget(window.target_cluster_btn, 1)
    card3_layout.addLayout(target_row)

    # Active node telemetry
    node_title = QLabel("Active node telemetry")
    node_title.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 600;")
    card3_layout.addWidget(node_title)

    node_val = QLabel("MDIM telemetry 36%, state 72:db")
    node_val.setStyleSheet("color: #64748b; font-size: 11px; font-family: Consolas, monospace;")
    card3_layout.addWidget(node_val)

    # Validator
    val_title = QLabel("Validator")
    val_title.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 600;")
    card3_layout.addWidget(val_title)

    val_row = QHBoxLayout()
    val_msg = QLabel("Checks for safe resume compatibility")
    val_msg.setStyleSheet("color: #94a3b8; font-size: 11px;")
    val_icon = QLabel("✔")
    val_icon.setStyleSheet("color: #10b981; font-size: 14px; font-weight: 800;")
    val_row.addWidget(val_msg, 1)
    val_row.addWidget(val_icon)
    card3_layout.addLayout(val_row)

    # Live training console terminal
    console_title = QLabel("Live training console")
    console_title.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 600;")
    card3_layout.addWidget(console_title)

    window.compute_console_log = QTextEdit()
    window.compute_console_log.setReadOnly(True)
    window.compute_console_log.setStyleSheet(
        "QTextEdit { background: #0a0c10; color: #94a3b8; font-family: Consolas, monospace; font-size: 10px; border: 1px solid #1a2727; border-radius: 6px; padding: 6px; }"
    )
    window.compute_console_log.setPlainText(
        "[10:323.230] Training throacre... Gradient %\n"
        "[10:323.230] Training teamsco... 76%\n"
        "[10:323.230] Training models\n"
        "[10:323.230] Training throughp...\n"
        "[10:323.230] Training recurcity...\n"
        "[10:323.230] Training meaming... Gradient Health\n"
        "[10:323.230] Training gradient...\n"
        "[10:323.230] Training building...\n"
        "[10:323.230] Training Logs\n"
        "[10:323.230] Training gradient... Health\n"
        "[10:323.230] Training models 25%\n"
        "[10:323.230] Training throughp... 26%"
    )
    card3_layout.addWidget(window.compute_console_log, 1)

    cards_row.addWidget(card3, 1)

    root.addLayout(cards_row, 1)

    return page
