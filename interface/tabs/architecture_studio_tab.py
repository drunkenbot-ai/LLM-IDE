"""Model Architecture Studio unified pre-training and fine-tuning screen."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from interface.widgets.parameter_donut_chart import ParameterDonutChartWidget
from interface.widgets.transformer_visualizer import TransformerVisualizerWidget


def _create_slider_spin_row(
    slider: QSlider,
    spin: QSpinBox | QDoubleSpinBox,
) -> QWidget:
    """Create a paired slider and spinbox layout matching Reference Image 1."""
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(10)
    slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    spin.setFixedWidth(86)
    spin.setStyleSheet(
        "QSpinBox, QDoubleSpinBox {"
        "  background: #141722;"
        "  color: #f8fafc;"
        "  border: 1px solid #282e42;"
        "  border-radius: 6px;"
        "  padding: 2px 4px;"
        "  font-weight: 600;"
        "}"
    )
    h.addWidget(slider, 1)
    h.addWidget(spin)
    return row


def build_architecture_studio_tab(window: Any) -> QWidget:
    """Build the unified Model Architecture Studio page.

    Combines Base Model Pre-training and LoRA Fine-Tuning into a unified 3-card layout
    strictly matching Reference Image 1.

    Args:
        window: Main application window holding shared state.

    Returns:
        Configured Model Architecture Studio page widget.
    """
    page = window._panel()
    root = QVBoxLayout(page)
    root.setContentsMargins(20, 16, 20, 16)
    root.setSpacing(12)

    # =========================================================================
    # Header: Title + Status + Project
    # =========================================================================
    header_row = QHBoxLayout()
    header_row.setContentsMargins(0, 0, 0, 0)
    header_row.setSpacing(12)

    title_label = QLabel("Model Architecture Studio")
    title_label.setObjectName("PageTitle")
    title_label.setStyleSheet("font-size: 22px; font-weight: 800; color: #f8fafc;")
    header_row.addWidget(title_label)

    header_row.addStretch(1)

    compute_status_label = QLabel("Compute: IDLE")
    compute_status_label.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 600;")
    header_row.addWidget(compute_status_label)

    amber_dot = QLabel("● Amber")
    amber_dot.setStyleSheet("color: #f59e0b; font-size: 12px; font-weight: 700;")
    header_row.addWidget(amber_dot)

    root.addLayout(header_row)

    # =========================================================================
    # Segmented Mode Switcher (Base Model vs Fine-Tuning)
    # =========================================================================
    switcher_frame = QFrame()
    switcher_frame.setObjectName("SegmentedSwitcher")
    switcher_layout = QHBoxLayout(switcher_frame)
    switcher_layout.setContentsMargins(0, 0, 0, 0)
    switcher_layout.setSpacing(8)

    window.base_mode_btn = QPushButton("Base Model (Pre-training)")
    window.base_mode_btn.setCheckable(True)
    window.base_mode_btn.setChecked(True)
    window.base_mode_btn.setStyleSheet(
        "QPushButton {"
        "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(139, 92, 246, 0.25), stop:1 rgba(139, 92, 246, 0.1));"
        "  color: #f8fafc;"
        "  border: 2px solid #8b5cf6;"
        "  border-radius: 8px;"
        "  padding: 10px;"
        "  font-weight: 800;"
        "  font-size: 13px;"
        "}"
    )

    window.finetune_mode_btn = QPushButton("Fine-Tuning (LoRA / Adapters)")
    window.finetune_mode_btn.setCheckable(True)
    window.finetune_mode_btn.setChecked(False)
    window.finetune_mode_btn.setStyleSheet(
        "QPushButton {"
        "  background: #151821;"
        "  color: #fbbf24;"
        "  border: 1px solid #78350f;"
        "  border-radius: 8px;"
        "  padding: 10px;"
        "  font-weight: 700;"
        "  font-size: 13px;"
        "}"
        "QPushButton:hover { background: #1c1813; border-color: #f59e0b; }"
    )

    def on_switch_mode(mode: str) -> None:
        if mode == "base":
            window.base_mode_btn.setChecked(True)
            window.finetune_mode_btn.setChecked(False)
            window.base_mode_btn.setStyleSheet(
                "QPushButton {"
                "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(139, 92, 246, 0.25), stop:1 rgba(139, 92, 246, 0.1));"
                "  color: #f8fafc;"
                "  border: 2px solid #8b5cf6;"
                "  border-radius: 8px;"
                "  padding: 10px;"
                "  font-weight: 800;"
                "  font-size: 13px;"
                "}"
            )
            window.finetune_mode_btn.setStyleSheet(
                "QPushButton {"
                "  background: #151821;"
                "  color: #fbbf24;"
                "  border: 1px solid #78350f;"
                "  border-radius: 8px;"
                "  padding: 10px;"
                "  font-weight: 700;"
                "  font-size: 13px;"
                "}"
                "QPushButton:hover { background: #1c1813; border-color: #f59e0b; }"
            )
            window.active_training_mode = "pretrain"
            if hasattr(window, "training_mode"):
                window.training_mode.setCurrentText("Pretrain from scratch")
        else:
            window.base_mode_btn.setChecked(False)
            window.finetune_mode_btn.setChecked(True)
            window.finetune_mode_btn.setStyleSheet(
                "QPushButton {"
                "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(245, 158, 11, 0.25), stop:1 rgba(245, 158, 11, 0.1));"
                "  color: #f8fafc;"
                "  border: 2px solid #f59e0b;"
                "  border-radius: 8px;"
                "  padding: 10px;"
                "  font-weight: 800;"
                "  font-size: 13px;"
                "}"
            )
            window.base_mode_btn.setStyleSheet(
                "QPushButton {"
                "  background: #151821;"
                "  color: #cbd5e1;"
                "  border: 1px solid #232738;"
                "  border-radius: 8px;"
                "  padding: 10px;"
                "  font-weight: 700;"
                "  font-size: 13px;"
                "}"
                "QPushButton:hover { background: #1a1d29; border-color: #8b5cf6; }"
            )
            window.active_training_mode = "fine_tune"
            if hasattr(window, "training_mode"):
                window.training_mode.setCurrentText("Fine-tune checkpoint")

        if hasattr(window, "update_train_button_state"):
            window.update_train_button_state()

    window.base_mode_btn.clicked.connect(lambda: on_switch_mode("base"))
    window.finetune_mode_btn.clicked.connect(lambda: on_switch_mode("finetune"))

    switcher_layout.addWidget(window.base_mode_btn, 1)
    switcher_layout.addWidget(window.finetune_mode_btn, 1)
    root.addWidget(switcher_frame)

    # Sub-header config row
    config_subrow = QHBoxLayout()
    config_subrow.setContentsMargins(0, 0, 0, 0)
    window.model_config_label = QLabel("Model Config: GPT-NeoX-20B")
    window.model_config_label.setStyleSheet("color: #cbd5e1; font-size: 12px; font-weight: 600;")
    config_subrow.addWidget(window.model_config_label)
    config_subrow.addStretch(1)

    project_btn = QPushButton("⚙ Project Alpha ▾")
    project_btn.setStyleSheet(
        "QPushButton { background: #141722; color: #94a3b8; border: 1px solid #282e42; border-radius: 6px; padding: 4px 10px; font-size: 11px; }"
        "QPushButton:hover { color: #f8fafc; border-color: #f59e0b; }"
    )
    config_subrow.addWidget(project_btn)
    root.addLayout(config_subrow)

    # =========================================================================
    # Three Main Cards (Visualizer | Parameters | LoRA)
    # =========================================================================
    cards_row = QHBoxLayout()
    cards_row.setSpacing(14)

    # -------------------------------------------------------------------------
    # Card 1: Architecture Visualizer
    # -------------------------------------------------------------------------
    vis_card = QFrame()
    vis_card.setObjectName("Card")
    vis_card.setStyleSheet(
        "QFrame#Card { background: #151821; border: 1px solid #232738; border-radius: 12px; }"
    )
    vis_layout = QVBoxLayout(vis_card)
    vis_layout.setContentsMargins(14, 12, 14, 12)
    vis_layout.setSpacing(8)

    vis_header = QLabel("Architecture Visualizer")
    vis_header.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 800; letter-spacing: 0.8px;")
    vis_layout.addWidget(vis_header)

    window.transformer_visualizer = TransformerVisualizerWidget()
    vis_layout.addWidget(window.transformer_visualizer, 1)

    cards_row.addWidget(vis_card, 1)

    # -------------------------------------------------------------------------
    # Card 2: Architecture Parameters
    # -------------------------------------------------------------------------
    params_card = QFrame()
    params_card.setObjectName("Card")
    params_card.setStyleSheet(
        "QFrame#Card { background: #151821; border: 1px solid #232738; border-radius: 12px; }"
    )
    params_layout = QVBoxLayout(params_card)
    params_layout.setContentsMargins(16, 12, 16, 12)
    params_layout.setSpacing(10)

    params_header_row = QHBoxLayout()
    params_title = QLabel("Architecture Parameters")
    params_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 800; letter-spacing: 0.8px;")
    params_info = QLabel("ⓘ")
    params_info.setStyleSheet("color: #64748b; font-size: 12px; font-weight: bold;")
    params_header_row.addWidget(params_title)
    params_header_row.addStretch(1)
    params_header_row.addWidget(params_info)
    params_layout.addLayout(params_header_row)

    base_sub = QLabel("Base Model")
    base_sub.setStyleSheet("color: #f8fafc; font-size: 13px; font-weight: 700;")
    params_layout.addWidget(base_sub)

    form_layout = QFormLayout()
    form_layout.setSpacing(6)
    form_layout.setLabelAlignment(Qt.AlignLeft)

    # Model Name Combo
    window.preset = QComboBox()
    window.preset.addItems(["GPT-NeoX-20B", "Llama-3-8B", "Mistral-7B", "MicroLLM-250M", "Custom"])
    window.preset.setStyleSheet(
        "QComboBox { background: #141722; color: #f8fafc; border: 1px solid #282e42; border-radius: 6px; padding: 4px 10px; font-size: 11px; }"
    )
    window._tip(window.preset, "Predefined model configuration presets with battle-tested hyperparameter profiles.")
    form_layout.addRow("Model Name", window.preset)

    # 1. Hidden Size
    window.hidden_size_slider = QSlider(Qt.Horizontal)
    window.hidden_size_slider.setRange(256, 16384)
    window.hidden_size_slider.setValue(4096)
    window.hidden_size = window._spin(256, 16384, 4096)
    window.hidden_size_slider.valueChanged.connect(window.hidden_size.setValue)
    window.hidden_size.valueChanged.connect(window.hidden_size_slider.setValue)
    window._tip(window.hidden_size, "Hidden dimension size (n_embd). Model embedding width; higher values increase capacity and expressiveness.")
    window._tip(window.hidden_size_slider, "Slide to adjust hidden embedding dimension (256 to 16,384).")
    form_layout.addRow("Hidden Size (n_embd)", _create_slider_spin_row(window.hidden_size_slider, window.hidden_size))

    # 2. Heads
    window.num_heads_slider = QSlider(Qt.Horizontal)
    window.num_heads_slider.setRange(1, 128)
    window.num_heads_slider.setValue(32)
    window.num_heads = window._spin(1, 128, 32)
    window.num_heads_slider.valueChanged.connect(window.num_heads.setValue)
    window.num_heads.valueChanged.connect(window.num_heads_slider.setValue)
    window._tip(window.num_heads, "Number of attention query heads (n_head). Must divide hidden size evenly.")
    window._tip(window.num_heads_slider, "Slide to adjust attention head count (1 to 128).")
    form_layout.addRow("Heads (n_head)", _create_slider_spin_row(window.num_heads_slider, window.num_heads))

    # 3. Layers
    window.num_layers_slider = QSlider(Qt.Horizontal)
    window.num_layers_slider.setRange(1, 128)
    window.num_layers_slider.setValue(44)
    window.num_layers = window._spin(1, 128, 44)
    window.num_layers_slider.valueChanged.connect(window.num_layers.setValue)
    window.num_layers.valueChanged.connect(window.num_layers_slider.setValue)
    window._tip(window.num_layers, "Number of stacked transformer layers (n_layer / depth).")
    window._tip(window.num_layers_slider, "Slide to adjust transformer layer count (1 to 128).")
    form_layout.addRow("Layers (n_layer)", _create_slider_spin_row(window.num_layers_slider, window.num_layers))

    # 4. Vocab Size
    window.vocab_size_slider = QSlider(Qt.Horizontal)
    window.vocab_size_slider.setRange(256, 128000)
    window.vocab_size_slider.setValue(50257)
    window.vocab_size = window._spin(256, 128000, 50257)
    window.vocab_size_slider.valueChanged.connect(window.vocab_size.setValue)
    window.vocab_size.valueChanged.connect(window.vocab_size_slider.setValue)
    window._tip(window.vocab_size, "Vocabulary size: Total token count in embedding and unembedding projection tables.")
    window._tip(window.vocab_size_slider, "Slide to adjust model vocabulary size (256 to 128,000).")
    form_layout.addRow("Vocab Size", _create_slider_spin_row(window.vocab_size_slider, window.vocab_size))

    # 5. RoPE Theta
    window.rope_theta_slider = QSlider(Qt.Horizontal)
    window.rope_theta_slider.setRange(1000, 1_000_000)
    window.rope_theta_slider.setValue(10000)
    window.rope_theta = window._double_spin(1.0, 10_000_000.0, 10000.0, 1000.0, 1)
    window.rope_theta_slider.valueChanged.connect(lambda v: window.rope_theta.setValue(float(v)))
    window.rope_theta.valueChanged.connect(lambda v: window.rope_theta_slider.setValue(int(v)))
    window._tip(window.rope_theta, "Rotary Position Embedding base frequency theta. Standard is 10000.0; extended context models use 500000.0+.")
    window._tip(window.rope_theta_slider, "Slide to adjust RoPE theta base frequency.")
    form_layout.addRow("RoPE Theta", _create_slider_spin_row(window.rope_theta_slider, window.rope_theta))

    # 6. Context Len
    window.context_length_slider = QSlider(Qt.Horizontal)
    window.context_length_slider.setRange(128, 32768)
    window.context_length_slider.setValue(2048)
    window.context_length = window._spin(16, 1_000_000, 2048)
    window.context_length_slider.valueChanged.connect(window.context_length.setValue)
    window.context_length.valueChanged.connect(window.context_length_slider.setValue)
    window._tip(window.context_length, "Context length / sequence length: Maximum number of consecutive tokens processed in each training sequence.")
    window._tip(window.context_length_slider, "Slide to adjust sequence context window length.")
    form_layout.addRow("Context Len", _create_slider_spin_row(window.context_length_slider, window.context_length))

    # 7. Intermediate Size (MLP Dim)
    window.intermediate_size_slider = QSlider(Qt.Horizontal)
    window.intermediate_size_slider.setRange(256, 65536)
    window.intermediate_size_slider.setValue(11008)
    window.intermediate_size = window._spin(256, 131072, 11008)
    window.intermediate_size_slider.valueChanged.connect(window.intermediate_size.setValue)
    window.intermediate_size.valueChanged.connect(window.intermediate_size_slider.setValue)
    window._tip(window.intermediate_size, "Intermediate dimension of MLP/FFN block. Typically 2.5x to 4x of hidden dimension.")
    window._tip(window.intermediate_size_slider, "Slide to adjust feedforward intermediate dimension.")
    form_layout.addRow("Intermediate Size ⓘ", _create_slider_spin_row(window.intermediate_size_slider, window.intermediate_size))

    # 8. Attention Type
    window.attention_type = QComboBox()
    window.attention_type.addItems(["Multi-Head (MHA)", "Grouped-Query (GQA)", "Multi-Query (MQA)"])
    window.attention_type.setStyleSheet(
        "QComboBox { background: #141722; color: #f8fafc; border: 1px solid #282e42; border-radius: 6px; padding: 4px 10px; font-size: 11px; }"
    )
    window._tip(window.attention_type, "Attention mechanism: Multi-Head (MHA), Grouped-Query (GQA) for memory savings, or Multi-Query (MQA).")
    form_layout.addRow("Attention Type ⓘ", window.attention_type)

    # 9. KV Heads (for GQA)
    window.kv_head_count_slider = QSlider(Qt.Horizontal)
    window.kv_head_count_slider.setRange(1, 128)
    window.kv_head_count_slider.setValue(8)
    window.kv_head_count = window._spin(1, 128, 8)
    window.kv_head_count_slider.valueChanged.connect(window.kv_head_count.setValue)
    window.kv_head_count.valueChanged.connect(window.kv_head_count_slider.setValue)
    window._tip(window.kv_head_count, "Key/Value head count for GQA. e.g. 8 KV heads with 32 Query heads reduces KV cache memory 4x.")
    window._tip(window.kv_head_count_slider, "Slide to adjust KV head count.")
    form_layout.addRow("KV Heads (n_kv_head) ⓘ", _create_slider_spin_row(window.kv_head_count_slider, window.kv_head_count))

    # 10. Activation Fn & Normalization
    arch_row = QHBoxLayout()
    arch_row.setSpacing(6)
    window.activation_fn = QComboBox()
    window.activation_fn.addItems(["SwiGLU", "GELU", "SiLU"])
    window.activation_fn.setStyleSheet(
        "QComboBox { background: #141722; color: #f8fafc; border: 1px solid #282e42; border-radius: 6px; padding: 4px 8px; font-size: 11px; }"
    )
    window._tip(window.activation_fn, "Non-linear feedforward activation function (SwiGLU used in Llama/Mistral, GELU in GPT-NeoX).")
    window.norm_type = QComboBox()
    window.norm_type.addItems(["RMSNorm", "LayerNorm"])
    window.norm_type.setStyleSheet(
        "QComboBox { background: #141722; color: #f8fafc; border: 1px solid #282e42; border-radius: 6px; padding: 4px 8px; font-size: 11px; }"
    )
    window._tip(window.norm_type, "Normalization layer: RMSNorm (faster, modern standard) or LayerNorm (classic post/pre-norm).")
    arch_row.addWidget(window.activation_fn, 1)
    arch_row.addWidget(window.norm_type, 1)
    form_layout.addRow("Activation & Norm ⓘ", arch_row)

    # 11. Sliding Window & Toggles
    toggles_row = QHBoxLayout()
    toggles_row.setSpacing(10)
    window.use_bias = QCheckBox("Use Bias")
    window.use_bias.setChecked(False)
    window.use_bias.setStyleSheet("color: #cbd5e1; font-size: 11px;")
    window._tip(window.use_bias, "Include learned bias parameters in linear layers. Disabled in modern LLMs for cleaner scaling.")
    window.tie_embeddings = QCheckBox("Tie Embeddings")
    window.tie_embeddings.setChecked(True)
    window.tie_embeddings.setStyleSheet("color: #cbd5e1; font-size: 11px;")
    window._tip(window.tie_embeddings, "Share input embedding and output projection weight matrices to reduce parameter footprint.")
    window.attention_window = window._spin(0, 65536, 0)
    window.attention_window.setToolTip("Sliding window size (0 = Full Context attention)")
    window.attention_window.setFixedWidth(70)
    toggles_row.addWidget(window.use_bias)
    toggles_row.addWidget(window.tie_embeddings)
    toggles_row.addStretch(1)
    toggles_row.addWidget(QLabel("Window:"))
    toggles_row.addWidget(window.attention_window)
    form_layout.addRow("Features & Window ⓘ", toggles_row)

    # 12. Window Stride (sample_stride)
    window.sample_stride_slider = QSlider(Qt.Horizontal)
    window.sample_stride_slider.setRange(1, 512)
    window.sample_stride_slider.setValue(128)
    window.sample_stride = window._spin(1, 4096, 128)
    window.sample_stride_slider.valueChanged.connect(window.sample_stride.setValue)
    window.sample_stride.valueChanged.connect(window.sample_stride_slider.setValue)
    window._tip(
        window.sample_stride,
        "Stride window (sample_stride): Token advance between consecutive sliding training sequences. Stride=1 yields dense overlapping windows; higher values reduce total sample count and speed up training passes.",
    )
    window._tip(window.sample_stride_slider, "Slide to adjust training sample window stride.")
    form_layout.addRow("Stride Window (stride) ⓘ", _create_slider_spin_row(window.sample_stride_slider, window.sample_stride))

    params_layout.addLayout(form_layout)

    # Donut Chart for Parameter Weight Breakdown
    donut_box = QFrame()
    donut_box.setStyleSheet("background: #11131c; border: 1px solid #1e2230; border-radius: 8px; padding: 6px;")
    donut_box_layout = QVBoxLayout(donut_box)
    donut_box_layout.setContentsMargins(8, 6, 8, 6)
    donut_box_layout.setSpacing(4)

    donut_title = QLabel("Parameter Weight Breakdown")
    donut_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 700;")
    donut_box_layout.addWidget(donut_title)

    window.donut_chart = ParameterDonutChartWidget(
        attention_pct=38.0,
        mlp_pct=41.0,
        embed_pct=21.0,
        total_params_str="20.3B",
    )
    donut_box_layout.addWidget(window.donut_chart)
    params_layout.addWidget(donut_box)

    cards_row.addWidget(params_card, 1)

    # -------------------------------------------------------------------------
    # Card 3: LoRA & Training Optimization Engine
    # -------------------------------------------------------------------------
    lora_card = QFrame()
    lora_card.setObjectName("Card")
    lora_card.setStyleSheet(
        "QFrame#Card { background: #151821; border: 1px solid #232738; border-radius: 12px; }"
    )
    lora_layout = QVBoxLayout(lora_card)
    lora_layout.setContentsMargins(16, 12, 16, 12)
    lora_layout.setSpacing(10)

    lora_title = QLabel("LoRA")
    lora_title.setStyleSheet("color: #f8fafc; font-size: 13px; font-weight: 700;")
    lora_layout.addWidget(lora_title)

    lora_form = QFormLayout()
    lora_form.setSpacing(6)
    lora_form.setLabelAlignment(Qt.AlignLeft)

    # 1. Rank
    window.fine_tune_lora_rank_slider = QSlider(Qt.Horizontal)
    window.fine_tune_lora_rank_slider.setRange(1, 128)
    window.fine_tune_lora_rank_slider.setValue(8)
    window.fine_tune_lora_rank = window._spin(1, 128, 8)
    window.fine_tune_lora_rank_slider.valueChanged.connect(window.fine_tune_lora_rank.setValue)
    window.fine_tune_lora_rank.valueChanged.connect(window.fine_tune_lora_rank_slider.setValue)
    window._tip(window.fine_tune_lora_rank, "LoRA rank (r): Low-rank bottleneck dimension for adapter matrices (typically 8, 16, or 32).")
    window._tip(window.fine_tune_lora_rank_slider, "Slide to adjust LoRA rank dimension.")
    lora_form.addRow("Rank", _create_slider_spin_row(window.fine_tune_lora_rank_slider, window.fine_tune_lora_rank))

    # 2. Alpha
    window.fine_tune_lora_alpha_slider = QSlider(Qt.Horizontal)
    window.fine_tune_lora_alpha_slider.setRange(1, 256)
    window.fine_tune_lora_alpha_slider.setValue(16)
    window.fine_tune_lora_alpha = window._spin(1, 256, 16)
    window.fine_tune_lora_alpha_slider.valueChanged.connect(window.fine_tune_lora_alpha.setValue)
    window.fine_tune_lora_alpha.valueChanged.connect(window.fine_tune_lora_alpha_slider.setValue)
    window._tip(window.fine_tune_lora_alpha, "LoRA alpha: Scaling factor for LoRA updates (typically 2x of rank).")
    window._tip(window.fine_tune_lora_alpha_slider, "Slide to adjust LoRA alpha scaling factor.")
    lora_form.addRow("Alpha", _create_slider_spin_row(window.fine_tune_lora_alpha_slider, window.fine_tune_lora_alpha))

    # 3. Dropout
    window.fine_tune_lora_dropout_slider = QSlider(Qt.Horizontal)
    window.fine_tune_lora_dropout_slider.setRange(0, 50)
    window.fine_tune_lora_dropout_slider.setValue(5)
    window.fine_tune_lora_dropout = window._double_spin(0.0, 0.5, 0.05, 0.01, 2)
    window.fine_tune_lora_dropout_slider.valueChanged.connect(lambda v: window.fine_tune_lora_dropout.setValue(v / 100.0))
    window.fine_tune_lora_dropout.valueChanged.connect(lambda v: window.fine_tune_lora_dropout_slider.setValue(int(v * 100)))
    window._tip(window.fine_tune_lora_dropout, "LoRA dropout: Dropout probability applied to LoRA adapter layers during fine-tuning.")
    window._tip(window.fine_tune_lora_dropout_slider, "Slide to adjust LoRA dropout rate.")
    lora_form.addRow("Dropout ⓘ", _create_slider_spin_row(window.fine_tune_lora_dropout_slider, window.fine_tune_lora_dropout))

    # Aliases for project state and training controller
    window.lora_rank = window.fine_tune_lora_rank
    window.lora_alpha = window.fine_tune_lora_alpha
    window.lora_dropout = window.fine_tune_lora_dropout

    lora_layout.addLayout(lora_form)

    # Adapter Injection Target Selector
    target_sec_label = QLabel("Adapter Injection Target Selector")
    target_sec_label.setStyleSheet("color: #f8fafc; font-size: 13px; font-weight: 700; margin-top: 6px;")
    lora_layout.addWidget(target_sec_label)

    target_pills_row = QHBoxLayout()
    target_pills_row.setSpacing(8)

    window.target_self_attn = QPushButton("● Self-Attn")
    window.target_self_attn.setCheckable(True)
    window.target_self_attn.setChecked(True)
    window.target_self_attn.setStyleSheet(
        "QPushButton { background: #2b1b44; color: #c084fc; border: 1px solid #9333ea; border-radius: 12px; padding: 5px 10px; font-size: 11px; font-weight: 700; }"
        "QPushButton:!checked { background: #141722; color: #64748b; border-color: #282e42; }"
    )
    window._tip(window.target_self_attn, "Inject LoRA adapters into Self-Attention query, key, value, and output projection matrices.")

    window.target_mlp = QPushButton("● MLP")
    window.target_mlp.setCheckable(True)
    window.target_mlp.setChecked(True)
    window.target_mlp.setStyleSheet(
        "QPushButton { background: #2b1b44; color: #c084fc; border: 1px solid #9333ea; border-radius: 12px; padding: 5px 10px; font-size: 11px; font-weight: 700; }"
        "QPushButton:!checked { background: #141722; color: #64748b; border-color: #282e42; }"
    )
    window._tip(window.target_mlp, "Inject LoRA adapters into Feed-Forward Network / MLP intermediate projections.")

    window.target_layernorm = QPushButton("○ LayerNorm")
    window.target_layernorm.setCheckable(True)
    window.target_layernorm.setChecked(False)
    window.target_layernorm.setStyleSheet(
        "QPushButton { background: #2b1b44; color: #c084fc; border: 1px solid #9333ea; border-radius: 12px; padding: 5px 10px; font-size: 11px; font-weight: 700; }"
        "QPushButton:!checked { background: #141722; color: #64748b; border-color: #282e42; }"
    )
    window._tip(window.target_layernorm, "Inject LoRA trainable scale/bias adapters into LayerNorm/RMSNorm blocks.")

    target_pills_row.addWidget(window.target_self_attn)
    target_pills_row.addWidget(window.target_mlp)
    target_pills_row.addWidget(window.target_layernorm)
    lora_layout.addLayout(target_pills_row)

    # Optimization Engine Parameters
    opt_sec_label = QLabel("Optimization & Hyperparameters")
    opt_sec_label.setStyleSheet("color: #f8fafc; font-size: 13px; font-weight: 700; margin-top: 8px;")
    lora_layout.addWidget(opt_sec_label)

    opt_form = QFormLayout()
    opt_form.setSpacing(6)
    opt_form.setLabelAlignment(Qt.AlignLeft)

    window.learning_rate = window._double_spin(0.00001, 0.1, 0.0003, 0.00005, 5)
    window._tip(window.learning_rate, "Optimizer learning rate. Recommended: 3e-4 for pre-training from scratch, 1e-4 for LoRA fine-tuning.")
    window.batch_size = window._spin(1, 1024, 16)
    window._tip(window.batch_size, "Micro-batch size per optimization step.")
    window.epochs = window._spin(1, 1000, 5)
    window._tip(window.epochs, "Number of complete dataset training epochs.")
    window.optimizer_name = QComboBox()
    window.optimizer_name.addItems(["AdamW", "AdamW (8-bit)", "Lion", "Adafactor", "Adam"])
    window.optimizer_name.setStyleSheet("QComboBox { background: #141722; color: #f8fafc; border: 1px solid #282e42; border-radius: 6px; padding: 3px 6px; font-size: 11px; }")
    window._tip(window.optimizer_name, "Optimizer algorithm. AdamW is the standard; 8-bit AdamW saves significant GPU memory.")
    window.precision = QComboBox()
    window.precision.addItems(["BF16 (Mixed)", "FP16 (Mixed)", "FP32"])
    window.precision.setStyleSheet("QComboBox { background: #141722; color: #f8fafc; border: 1px solid #282e42; border-radius: 6px; padding: 3px 6px; font-size: 11px; }")
    window._tip(window.precision, "Compute precision. BF16 (Mixed) is recommended for modern GPUs; FP16 or FP32 fallback.")
    window.scheduler_name = QComboBox()
    window.scheduler_name.addItems(["Cosine decay", "Warmup linear", "Polynomial decay", "One-cycle", "Constant"])
    window.scheduler_name.setStyleSheet("QComboBox { background: #141722; color: #f8fafc; border: 1px solid #282e42; border-radius: 6px; padding: 3px 6px; font-size: 11px; }")
    window._tip(window.scheduler_name, "Learning rate schedule. Cosine decay with warmup is recommended for stable convergence.")
    window.warmup_steps = window._spin(0, 100000, 100)
    window._tip(window.warmup_steps, "Warmup steps: Initial steps with linearly increasing learning rate to prevent early divergence.")
    window.gradient_accumulation = window._spin(1, 256, 1)
    window._tip(window.gradient_accumulation, "Gradient accumulation steps: Accumulates gradients across N micro-steps to simulate a larger effective batch size.")
    window.weight_decay = window._double_spin(0.0, 1.0, 0.1, 0.01, 3)
    window._tip(window.weight_decay, "L2 weight decay penalty coefficient to regularize weights and prevent overfitting.")
    window.max_grad_norm = window._double_spin(0.1, 100.0, 1.0, 0.1, 2)
    window._tip(window.max_grad_norm, "Maximum gradient norm clipping threshold to prevent exploding gradients and loss spikes.")
    window.save_interval = window._spin(1, 100000, 500)
    window._tip(window.save_interval, "Checkpoint save interval (in optimization steps).")
    window.eval_interval = window._spin(0, 100000, 100)
    window._tip(window.eval_interval, "Validation loss evaluation interval (in optimization steps).")

    opt_form.addRow("Learning Rate", window.learning_rate)
    opt_form.addRow("Batch Size", window.batch_size)
    opt_form.addRow("Epochs", window.epochs)
    opt_form.addRow("Optimizer", window.optimizer_name)
    opt_form.addRow("Scheduler", window.scheduler_name)
    opt_form.addRow("Warmup / Accum", _create_slider_spin_row(window.gradient_accumulation, window.warmup_steps) if hasattr(window, "_create_paired_row") else window._paired_row(window.warmup_steps, "Accum", window.gradient_accumulation) if hasattr(window, "_paired_row") else window.warmup_steps)
    opt_form.addRow("Decay / Grad Norm", window._paired_row(window.weight_decay, "Max grad", window.max_grad_norm) if hasattr(window, "_paired_row") else window.weight_decay)
    opt_form.addRow("Save / Eval Step", window._paired_row(window.save_interval, "Eval", window.eval_interval) if hasattr(window, "_paired_row") else window.save_interval)
    opt_form.addRow("Precision", window.precision)

    opt_toggles = QHBoxLayout()
    window.activation_checkpointing = QCheckBox("Act Ckpt")
    window.activation_checkpointing.setChecked(True)
    window.activation_checkpointing.setStyleSheet("color: #cbd5e1; font-size: 11px;")
    window._tip(window.activation_checkpointing, "Activation checkpointing: Recomputes activations during backward pass to drastically reduce VRAM usage.")
    window.compile_model = QCheckBox("Torch Compile")
    window.compile_model.setChecked(False)
    window.compile_model.setStyleSheet("color: #cbd5e1; font-size: 11px;")
    window._tip(window.compile_model, "PyTorch 2.x Inductor compiler: Fuses kernel operations to accelerate GPU throughput.")
    opt_toggles.addWidget(window.activation_checkpointing)
    opt_toggles.addWidget(window.compile_model)
    opt_form.addRow("Acceleration", opt_toggles)
    lora_layout.addLayout(opt_form)

    # Launch Target Selector (Local machine vs Cluster)
    launch_target_row = QHBoxLayout()
    launch_target_row.setSpacing(6)
    launch_target_label = QLabel("Launch Target:")
    launch_target_label.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 700;")
    window.training_launch_target = QComboBox()
    window.training_launch_target.addItems(["Local machine", "Cluster (Local SGD)"])
    window.training_launch_target.setStyleSheet(
        "QComboBox { background: #141722; color: #38bdf8; border: 1px solid #0284c7; border-radius: 6px; padding: 4px 8px; font-size: 11px; font-weight: bold; }"
    )
    window._tip(
        window.training_launch_target,
        "Select training execution target: 'Local machine' trains on local GPU/CPU. 'Cluster (Local SGD)' dispatches a distributed job across the cluster fleet.",
    )
    launch_target_row.addWidget(launch_target_label)
    launch_target_row.addWidget(window.training_launch_target, 1)
    lora_layout.addLayout(launch_target_row)

    # Action buttons matching modern IDE workflow
    actions_row = QHBoxLayout()
    actions_row.setSpacing(8)
    window.train_button = QPushButton("Start Base Pre-Training")
    window.train_button.setStyleSheet(
        "QPushButton { background: #10b981; color: white; border: none; border-radius: 6px; padding: 8px 14px; font-size: 12px; font-weight: 800; }"
        "QPushButton:hover { background: #059669; }"
    )
    window._tip(window.train_button, "Start model training using the active mode (Base Pre-training or LoRA Fine-tuning) and launch target.")

    window.dry_run_button = QPushButton("Dry Run")
    window.dry_run_button.setStyleSheet(
        "QPushButton { background: #151821; color: #cbd5e1; border: 1px solid #282e42; border-radius: 6px; padding: 8px 10px; font-size: 11px; font-weight: 700; }"
        "QPushButton:hover { border-color: #f59e0b; color: white; }"
    )
    window._tip(window.dry_run_button, "Execute a test forward/backward pass with dummy data to verify GPU memory, gradient flow, and execution speed.")
    actions_row.addWidget(window.train_button, 2)
    actions_row.addWidget(window.dry_run_button, 1)
    lora_layout.addLayout(actions_row)

    def update_train_button_state() -> None:
        mode = getattr(window, "active_training_mode", "pretrain")
        target = window.training_launch_target.currentText()
        is_cluster = "Cluster" in target
        if mode == "fine_tune":
            if is_cluster:
                window.train_button.setText("Launch Cluster Fine-Tuning")
                window.train_button.setToolTip("Submit distributed LoRA fine-tuning job (Local SGD) to the cluster worker fleet.")
            else:
                window.train_button.setText("Start LoRA Fine-Tuning")
                window.train_button.setToolTip("Start local LoRA adapter fine-tuning using configured targets and hyperparameters.")
        else:
            if is_cluster:
                window.train_button.setText("Launch Cluster Pre-Training")
                window.train_button.setToolTip("Submit distributed pre-training job (Local SGD) to the cluster worker fleet.")
            else:
                window.train_button.setText("Start Base Pre-Training")
                window.train_button.setToolTip("Start local model pre-training from scratch using configured architecture and hyperparameters.")

    window.update_train_button_state = update_train_button_state
    window.training_launch_target.currentTextChanged.connect(lambda _: update_train_button_state())
    update_train_button_state()

    def on_train_button_clicked() -> None:
        mode = getattr(window, "active_training_mode", "pretrain")
        target = window.training_launch_target.currentText()
        if "Cluster" in target:
            if hasattr(window, "launch_cluster_training_job"):
                window.launch_cluster_training_job(training_mode=mode)
            if hasattr(window, "_switch_page") and hasattr(window, "job_manager_page_index"):
                window._switch_page(window.job_manager_page_index)
        else:
            if mode == "fine_tune":
                if hasattr(window, "start_fine_tuning"):
                    window.start_fine_tuning()
                elif hasattr(window, "start_training"):
                    window.start_training()
            else:
                if hasattr(window, "start_training"):
                    window.start_training()
            if hasattr(window, "_switch_page") and hasattr(window, "live_page_index"):
                window._switch_page(window.live_page_index)

    window.train_button.clicked.connect(on_train_button_clicked)

    lora_layout.addStretch(1)

    cards_row.addWidget(lora_card, 1)

    root.addLayout(cards_row, 1)

    # =========================================================================
    # Bottom Status Bar
    # =========================================================================
    status_bar_row = QHBoxLayout()
    status_bar_row.setContentsMargins(4, 4, 4, 4)

    left_status = QLabel("Compute: IDLE   ● Amber")
    left_status.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
    status_bar_row.addWidget(left_status)

    status_bar_row.addStretch(1)

    right_status = QLabel("Project Saved | GPU: CUDA Enabled")
    right_status.setStyleSheet("color: #10b981; font-size: 11px; font-weight: 700;")
    status_bar_row.addWidget(right_status)

    root.addLayout(status_bar_row)

    # Dynamic parameter recalculation routine
    def recalculate_parameters(*args: Any) -> None:
        try:
            h = float(window.hidden_size.value())
            l = float(window.num_layers.value())
            v = float(window.vocab_size.value())
            inter = float(window.intermediate_size.value()) if hasattr(window, "intermediate_size") else (4.0 * h)
            is_swiglu = "swiglu" in window.activation_fn.currentText().lower() if hasattr(window, "activation_fn") else True

            # Embedding params (tied: v * h, untied: 2 * v * h)
            embed_params = v * h

            # Attention params per layer: Q, K, V, O projections (4 * h^2)
            attn_layer_params = 4.0 * h * h

            # MLP params per layer:
            # SwiGLU: gate + up + down projections (3 * h * inter)
            # GELU: up + down projections (2 * h * inter)
            mlp_layer_params = (3.0 if is_swiglu else 2.0) * h * inter

            total_attn = l * attn_layer_params
            total_mlp = l * mlp_layer_params
            total_params = embed_params + total_attn + total_mlp

            if total_params > 0:
                attn_pct = (total_attn / total_params) * 100.0
                mlp_pct = (total_mlp / total_params) * 100.0
                embed_pct = (embed_params / total_params) * 100.0
            else:
                attn_pct, mlp_pct, embed_pct = 38.0, 41.0, 21.0

            if total_params >= 1e9:
                t_str = f"{total_params / 1e9:.1f}B"
            elif total_params >= 1e6:
                t_str = f"{total_params / 1e6:.1f}M"
            else:
                t_str = f"{total_params / 1e3:.1f}K"

            window.donut_chart.set_breakdown(attn_pct, mlp_pct, embed_pct, t_str)
        except Exception:
            pass

    window.hidden_size.valueChanged.connect(recalculate_parameters)
    window.num_layers.valueChanged.connect(recalculate_parameters)
    window.vocab_size.valueChanged.connect(recalculate_parameters)
    window.intermediate_size.valueChanged.connect(recalculate_parameters)
    window.activation_fn.currentTextChanged.connect(recalculate_parameters)
    recalculate_parameters()

    # Visualizer synchronization routine
    def update_visualizer_from_ui() -> None:
        if not hasattr(window, "transformer_visualizer"):
            return
        model_name = window.preset.currentText() if hasattr(window, "preset") else "GPT-NeoX-20B"
        norm_type = window.norm_type.currentText() if hasattr(window, "norm_type") else "RMSNorm"
        activation = window.activation_fn.currentText() if hasattr(window, "activation_fn") else "SwiGLU"
        attention_type = window.attention_type.currentText() if hasattr(window, "attention_type") else "Multi-Head (MHA)"
        pos_encoding = "RoPE"
        use_bias = window.use_bias.isChecked() if hasattr(window, "use_bias") else False
        window.transformer_visualizer.set_architecture(
            model_name=model_name,
            norm_type=norm_type,
            activation=activation,
            attention_type=attention_type,
            pos_encoding=pos_encoding,
            use_bias=use_bias,
        )

    window.update_visualizer_from_ui = update_visualizer_from_ui

    # Preset selection synchronization
    def on_preset_changed(text: str) -> None:
        window.model_config_label.setText(f"Model Config: {text}")
        if "NeoX" in text:
            window.hidden_size.setValue(4096)
            window.num_heads.setValue(32)
            window.num_layers.setValue(44)
            window.vocab_size.setValue(50257)
            window.intermediate_size.setValue(16384)
            window.attention_type.setCurrentText("Multi-Head (MHA)")
            window.activation_fn.setCurrentText("GELU")
            window.norm_type.setCurrentText("LayerNorm")
        elif "Llama-3" in text:
            window.hidden_size.setValue(4096)
            window.num_heads.setValue(32)
            window.num_layers.setValue(32)
            window.vocab_size.setValue(128256)
            window.intermediate_size.setValue(14336)
            window.kv_head_count.setValue(8)
            window.attention_type.setCurrentText("Grouped-Query (GQA)")
            window.activation_fn.setCurrentText("SwiGLU")
            window.norm_type.setCurrentText("RMSNorm")
        elif "Mistral" in text:
            window.hidden_size.setValue(4096)
            window.num_heads.setValue(32)
            window.num_layers.setValue(32)
            window.vocab_size.setValue(32000)
            window.intermediate_size.setValue(14336)
            window.kv_head_count.setValue(8)
            window.attention_type.setCurrentText("Grouped-Query (GQA)")
            window.activation_fn.setCurrentText("SwiGLU")
            window.norm_type.setCurrentText("RMSNorm")
        elif "MicroLLM" in text:
            window.hidden_size.setValue(768)
            window.num_heads.setValue(12)
            window.num_layers.setValue(12)
            window.vocab_size.setValue(8000)
            window.intermediate_size.setValue(2816)
            window.attention_type.setCurrentText("Multi-Head (MHA)")
            window.activation_fn.setCurrentText("SwiGLU")
            window.norm_type.setCurrentText("RMSNorm")
        update_visualizer_from_ui()

    window.preset.currentTextChanged.connect(on_preset_changed)
    window.norm_type.currentTextChanged.connect(lambda _: update_visualizer_from_ui())
    window.activation_fn.currentTextChanged.connect(lambda _: update_visualizer_from_ui())
    window.attention_type.currentTextChanged.connect(lambda _: update_visualizer_from_ui())
    window.use_bias.toggled.connect(lambda _: update_visualizer_from_ui())
    update_visualizer_from_ui()

    # Legacy attributes maintained for compatibility with training mixins and runners
    if not hasattr(window, "train_data_dir"):
        window.train_data_dir = QLineEdit(str(Path.cwd() / "runs" / "dataset"))
    if not hasattr(window, "model_dir"):
        window.model_dir = QLineEdit(str(Path.cwd() / "runs" / "model"))
    window.architecture_style = getattr(window, "norm_type", None) or QComboBox()
    if not hasattr(window, "use_bias"):
        window.use_bias = QCheckBox()
        window.use_bias.setChecked(True)
    if not hasattr(window, "batch_size"):
        window.batch_size = window._spin(1, 1024, 16)
    if not hasattr(window, "learning_rate"):
        window.learning_rate = window._double_spin(0.00001, 0.1, 0.0003, 0.00005, 5)
    if not hasattr(window, "epochs"):
        window.epochs = window._spin(1, 100, 5)
    if not hasattr(window, "warmup_steps"):
        window.warmup_steps = window._spin(0, 10000, 100)
    if not hasattr(window, "weight_decay"):
        window.weight_decay = window._double_spin(0.0, 1.0, 0.1, 0.01, 3)
    if not hasattr(window, "max_grad_norm"):
        window.max_grad_norm = window._double_spin(0.0, 10.0, 1.0, 0.1, 2)
    window.checkpoint_interval = window._spin(10, 100000, 500)
    if not hasattr(window, "mixed_precision"):
        window.mixed_precision = QCheckBox()
        window.mixed_precision.setChecked(True)
    if not hasattr(window, "compile_model"):
        window.compile_model = QCheckBox()
    if not hasattr(window, "activation_checkpointing"):
        window.activation_checkpointing = QCheckBox()
        window.activation_checkpointing.setChecked(True)
    if not hasattr(window, "optimizer_8bit"):
        window.optimizer_8bit = QCheckBox()
        window.optimizer_8bit.setChecked(True)
    if not hasattr(window, "cpu_offload"):
        window.cpu_offload = QCheckBox()
    if not hasattr(window, "stop_training_button"):
        window.stop_training_button = QPushButton("Stop")
    window.stop_train_button = window.stop_training_button
    window.fine_tune_button = QPushButton("Start Fine-Tune")
    window.stop_fine_tune_button = QPushButton("Stop")
    window.training_process_status = QLabel("Worker: idle")
    window.fine_tune_process_status = QLabel("Worker: idle")
    window.fine_tune_adapter_type = QComboBox()
    window.fine_tune_model_path = QLineEdit()
    window.fine_tune_data_dir = QLineEdit()
    window.fine_tune_output_dir = QLineEdit()
    window.fine_tune_log = QTextEdit()
    window.training_log = QTextEdit()
    window.fine_tune_preview = QLabel("No compatibility check has been run.")

    # Telemetry metrics & progress tracking expected by training controllers
    window.training_step_metric = QLabel("Step: 0/0")
    window.training_epoch_metric = QLabel("Epoch: -")
    window.training_loss_metric = QLabel("Train loss: -")
    window.training_val_metric = QLabel("Val loss: -")
    window.training_lr_metric = QLabel("LR: -")
    window.training_grad_metric = QLabel("Grad: -")
    window.training_speed_metric = QLabel("Speed: -")
    window.training_vram_metric = QLabel("VRAM: -")
    window.training_eta_metric = QLabel("ETA: -")
    window.training_elapsed_metric = QLabel("Total time: -")
    window.training_health_metric = QLabel("Health: -")
    window.fine_tune_health_metric = QLabel("Health: -")
    window.model_size_metric = QLabel("Model: -")
    window.vram_estimate_metric = QLabel("VRAM est: -")
    window.parameter_breakdown_metric = QLabel("Params: -")
    window.memory_breakdown_metric = QLabel("Memory: -")
    window.architecture_advisor_metric = QLabel("Advisor: -")
    window.history_metric = QLabel("Runs: -")
    window.refresh_estimate_button = QPushButton("Refresh Estimate")
    if hasattr(window, "refresh_model_estimate"):
        window.refresh_estimate_button.clicked.connect(window.refresh_model_estimate)
    window.resume_check_button = QPushButton("Check Resume")
    if hasattr(window, "preview_resume_compatibility"):
        window.resume_check_button.clicked.connect(window.preview_resume_compatibility)
    window.resume_training_preview = QTextEdit()
    window.resume_training_preview.setReadOnly(True)
    window.resume_training_preview.setText("No compatibility check has been run.")
    window.resume_preview = window.resume_training_preview
    if not hasattr(window, "train_button"):
        window.train_button = QPushButton("Start Training")
    if hasattr(window, "start_training"):
        window.train_button.clicked.connect(window.start_training)
    if hasattr(window, "stop_training_process"):
        window.stop_training_button.clicked.connect(window.stop_training_process)
    window.training_progress = window._thin_progress() if hasattr(window, "_thin_progress") else QProgressBar()

    window.fine_tune_step_metric = QLabel("Step: 0/0")
    window.fine_tune_loss_metric = QLabel("Train loss: -")
    window.fine_tune_val_metric = QLabel("Val loss: -")
    window.fine_tune_lr_metric = QLabel("LR: -")
    window.fine_tune_grad_metric = QLabel("Grad: -")
    window.fine_tune_speed_metric = QLabel("Speed: -")
    window.fine_tune_eta_metric = QLabel("ETA: -")
    window.fine_tune_progress = window._thin_progress() if hasattr(window, "_thin_progress") else QProgressBar()

    # Model dimension aliases for direct compatibility with project serializer
    window.n_embd = window.hidden_size
    window.n_head = window.num_heads
    window.n_layer = window.num_layers
    window.context_len = window.context_length
    window.train_context_length = window.context_length

    # Additional project state attributes
    if not hasattr(window, "resume_checkpoint"):
        window.resume_checkpoint = QLineEdit("")
    if not hasattr(window, "fine_tune_checkpoint"):
        window.fine_tune_checkpoint = QLineEdit("")
    if not hasattr(window, "training_launch_target"):
        window.training_launch_target = QComboBox()
        window.training_launch_target.addItems(["Local machine", "Cluster (Local SGD)"])
    if not hasattr(window, "fine_tune_launch_target"):
        window.fine_tune_launch_target = QComboBox()
        window.fine_tune_launch_target.addItems(["Local machine", "Cluster (Local SGD)"])
    if not hasattr(window, "attention_type"):
        window.attention_type = QComboBox()
        window.attention_type.addItems(["Multi-head", "Grouped-query", "Multi-query"])
    if not hasattr(window, "kv_head_count"):
        window.kv_head_count = window._spin(1, 128, 4)
    if not hasattr(window, "attention_backend"):
        window.attention_backend = QComboBox()
        window.attention_backend.addItems(["SDPA / Flash when available", "Manual"])
    if not hasattr(window, "attention_window"):
        window.attention_window = window._spin(0, 32768, 0)
    if not hasattr(window, "training_mode"):
        window.training_mode = QComboBox()
        window.training_mode.addItems(["Pretrain from scratch", "Fine-tune checkpoint", "Instruction fine-tune", "Conversation fine-tune", "Code fine-tune"])
    if not hasattr(window, "peft_method"):
        window.peft_method = QComboBox()
        window.peft_method.addItems(["Full fine-tune", "LoRA adapters"])
    if not hasattr(window, "lora_targets"):
        window.lora_targets = QComboBox()
        window.lora_targets.addItems(["Attention projections", "MLP projections", "Attention + MLP"])
    if not hasattr(window, "lora_rank"):
        window.lora_rank = getattr(window, "fine_tune_lora_rank", None) or window._spin(1, 256, 8)
    if not hasattr(window, "lora_alpha"):
        window.lora_alpha = getattr(window, "fine_tune_lora_alpha", None) or window._double_spin(1.0, 512.0, 16.0, 1.0, 1)
    if not hasattr(window, "lora_dropout"):
        window.lora_dropout = getattr(window, "fine_tune_lora_dropout", None) or window._double_spin(0.0, 0.9, 0.05, 0.01, 3)
    if not hasattr(window, "fine_tune_check_button"):
        window.fine_tune_check_button = QPushButton("Check Fine-tune")
    if not hasattr(window, "fine_tune_dataset_status"):
        window.fine_tune_dataset_status = QLabel("Dataset: not checked")
    if not hasattr(window, "fine_tune_refresh_button"):
        window.fine_tune_refresh_button = QPushButton("Refresh Dataset Fit")
    if not hasattr(window, "apply_lora_preset_button"):
        window.apply_lora_preset_button = QPushButton("Apply Recommended LoRA")
    if not hasattr(window, "fine_tune_epoch_metric"):
        window.fine_tune_epoch_metric = QLabel("Epoch: -")
    if not hasattr(window, "fine_tune_runtime_hint"):
        window.fine_tune_runtime_hint = QLabel("Uses AI tab device, precision, resume, and checkpoint settings.")
    if not hasattr(window, "fine_tune_dataset_builder_stage"):
        window.fine_tune_dataset_builder_stage = QComboBox()
        window.fine_tune_dataset_builder_stage.addItems([
            "Instruction fine-tune",
            "Conversation fine-tune",
            "Tool-call fine-tune",
            "Code fine-tune",
            "Thinking fine-tune",
        ])
    if not hasattr(window, "fine_tune_button"):
        window.fine_tune_button = QPushButton("Start Fine-Tune")
    if not hasattr(window, "stop_fine_tune_button"):
        window.stop_fine_tune_button = QPushButton("Stop Fine-Tune")
    if not hasattr(window, "fine_tune_process_status"):
        window.fine_tune_process_status = QLabel("Worker: detached | Run: - | PID: -")

    if hasattr(window, "preview_fine_tune_compatibility"):
        window.fine_tune_check_button.clicked.connect(window.preview_fine_tune_compatibility)
    if hasattr(window, "refresh_fine_tune_workflow"):
        window.fine_tune_refresh_button.clicked.connect(window.refresh_fine_tune_workflow)
    if hasattr(window, "apply_recommended_fine_tune_settings"):
        window.apply_lora_preset_button.clicked.connect(window.apply_recommended_fine_tune_settings)
    if hasattr(window, "_update_training_mode_controls"):
        window.training_mode.currentTextChanged.connect(window._update_training_mode_controls)
        window.peft_method.currentTextChanged.connect(window._update_training_mode_controls)
    if hasattr(window, "refresh_fine_tune_workflow"):
        window.training_mode.currentTextChanged.connect(window.refresh_fine_tune_workflow)
    if hasattr(window, "_refresh_fine_tune_default_output"):
        window.training_mode.currentTextChanged.connect(window._refresh_fine_tune_default_output)
        window.fine_tune_dataset_builder_stage.currentTextChanged.connect(window._refresh_fine_tune_default_output)

    if not hasattr(window, "dropout"):
        window.dropout = window._double_spin(0.0, 1.0, 0.0, 0.01, 2)
    if not hasattr(window, "training_profile"):
        window.training_profile = QComboBox()
        window.training_profile.addItems(["Stable LLM", "Aggressive", "Conservative"])
    if not hasattr(window, "optimizer_name"):
        window.optimizer_name = QComboBox()
        window.optimizer_name.addItems(["AdamW", "AdamW (8-bit)", "Adam", "Lion", "Adafactor"])
    if not hasattr(window, "scheduler_name"):
        window.scheduler_name = QComboBox()
        window.scheduler_name.addItems(["Warmup linear", "Cosine decay", "Polynomial decay", "One-cycle", "Constant"])
    if not hasattr(window, "min_lr_ratio"):
        window.min_lr_ratio = window._double_spin(0.0, 1.0, 0.1, 0.01, 2)
    if not hasattr(window, "polynomial_power"):
        window.polynomial_power = window._double_spin(0.1, 5.0, 1.0, 0.1, 2)
    if not hasattr(window, "gradient_accumulation"):
        window.gradient_accumulation = window._spin(1, 128, 1)
    if not hasattr(window, "sample_stride"):
        window.sample_stride = window._spin(1, 128, 1)
    if not hasattr(window, "eval_interval"):
        window.eval_interval = window._spin(1, 10000, 100)
    if not hasattr(window, "max_eval_batches"):
        window.max_eval_batches = window._spin(1, 1000, 10)
    if not hasattr(window, "save_interval"):
        window.save_interval = window._spin(1, 10000, 500)
    if not hasattr(window, "data_loader_workers"):
        window.data_loader_workers = window._spin(0, 32, 2)
    if not hasattr(window, "seed"):
        window.seed = window._spin(0, 999999, 42)
    if not hasattr(window, "device"):
        window.device = QComboBox()
        window.device.addItems(["cuda:0", "cpu"])
    if not hasattr(window, "use_amp"):
        window.use_amp = QCheckBox()
        window.use_amp.setChecked(True)
        window.use_amp_default = True
    if not hasattr(window, "device_info"):
        window.device_info = QLabel("CUDA Enabled")
    if not hasattr(window, "precision"):
        window.precision = QComboBox()
        window.precision.addItems(["FP16", "BF16", "FP32"])
    if not hasattr(window, "resume_training"):
        window.resume_training = QCheckBox()
        window.resume_training.setChecked(True)
    if not hasattr(window, "resume_safety"):
        window.resume_safety = QCheckBox()
        window.resume_safety.setChecked(True)
    window.early_stopping = QCheckBox()
    window.early_stopping.setChecked(True)
    window.early_stopping_patience = window._spin(1, 50, 5)
    window.benchmark_prompts = QTextEdit()
    window.benchmark_tokens = window._spin(1, 1000, 100)
    window.benchmark_temperature = window._double_spin(0.0, 2.0, 0.7, 0.05, 2)
    window.benchmark_kv_cache = QCheckBox()
    window.benchmark_kv_cache.setChecked(True)

    return page
