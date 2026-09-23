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
    spin.setFixedWidth(92)
    h.addWidget(slider, 1)
    h.addWidget(spin)
    return row


def _paired_row(
    w1: QWidget,
    lbl2_text: str = "",
    w2: QWidget | None = None,
    stretch1: int = 1,
    stretch2: int = 1,
) -> QWidget:
    """Pack two form controls into a single row to save vertical space."""
    row = QWidget()
    row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    h.addWidget(w1, stretch1)
    if lbl2_text and w2 is not None:
        lbl2 = QLabel(lbl2_text)
        lbl2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl2.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        h.addWidget(lbl2)
    if w2 is not None:
        h.addWidget(w2, stretch2)
    return row


def _form_row(
    label_text: str,
    widget: QWidget,
    tip_text: str = "",
    label_width: int = 155,
    window: Any = None,
) -> QWidget:
    """Create an amber-bordered parameter container row with label and value controls."""
    row = QWidget()
    row.setObjectName("FormRow")
    h = QHBoxLayout(row)
    h.setContentsMargins(10, 5, 10, 5)
    h.setSpacing(10)
    if label_text:
        lbl = QLabel(label_text)
        lbl.setObjectName("IngestionFieldLabel")
        if label_width > 0:
            lbl.setFixedWidth(label_width)
        if window and tip_text:
            window._tip(lbl, tip_text)
        h.addWidget(lbl)
    if window and tip_text and widget:
        window._tip(widget, tip_text)
    if widget:
        h.addWidget(widget, 1)
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
    title_label.setStyleSheet("font-size: 22px; font-weight: 800;")
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
    window.base_mode_btn.setObjectName("BaseModeBtn")
    window.base_mode_btn.setCheckable(True)
    window.base_mode_btn.setChecked(True)

    window.finetune_mode_btn = QPushButton("Fine-Tuning (LoRA / Adapters)")
    window.finetune_mode_btn.setObjectName("FinetuneModeBtn")
    window.finetune_mode_btn.setCheckable(True)
    window.finetune_mode_btn.setChecked(False)

    def on_switch_mode(mode: str) -> None:
        is_base = (mode == "base")
        window.base_mode_btn.setChecked(is_base)
        window.finetune_mode_btn.setChecked(not is_base)
        window.active_training_mode = "pretrain" if is_base else "fine_tune"
        if hasattr(window, "training_mode"):
            window.training_mode.setCurrentText("Pretrain from scratch" if is_base else "Fine-tune checkpoint")
        if hasattr(window, "peft_method"):
            window._set_combo_text(window.peft_method, "Full fine-tune" if is_base else "LoRA adapters")
        if not is_base and hasattr(window, "fine_tune_checkpoint") and hasattr(window, "resume_checkpoint"):
            if not window.fine_tune_checkpoint.text().strip() and window.resume_checkpoint.text().strip():
                window.fine_tune_checkpoint.setText(window.resume_checkpoint.text().strip())
        if hasattr(window, "lora_section_widget"):
            window.lora_section_widget.setVisible(not is_base)

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
    window.model_config_label.setStyleSheet("font-size: 12px; font-weight: 600;")
    config_subrow.addWidget(window.model_config_label)
    config_subrow.addStretch(1)

    project_btn = QPushButton("⚙ Project Alpha ▾")
    project_btn.setObjectName("SecondaryAction")
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
    params_card_outer = QVBoxLayout(params_card)
    params_card_outer.setContentsMargins(14, 12, 14, 12)
    params_card_outer.setSpacing(8)

    params_scroll = QScrollArea()
    params_scroll.setWidgetResizable(True)
    params_scroll.setFrameShape(QFrame.NoFrame)
    params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    params_content = QWidget()
    params_layout = QVBoxLayout(params_content)
    params_layout.setContentsMargins(0, 0, 4, 0)
    params_layout.setSpacing(10)
    params_scroll.setWidget(params_content)
    params_card_outer.addWidget(params_scroll, 1)

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
    base_sub.setStyleSheet("font-size: 13px; font-weight: 700;")
    params_layout.addWidget(base_sub)

    form_layout = QVBoxLayout()
    form_layout.setSpacing(6)

    # Model Name Combo
    window.preset = QComboBox()
    window.preset.addItems(["GPT-NeoX-20B", "Llama-3-8B", "Mistral-7B", "MicroLLM-250M", "Custom"])
    window._tip(window.preset, "Predefined model configuration presets with battle-tested hyperparameter profiles.")
    form_layout.addWidget(_form_row("Model Name", window.preset, label_width=155, window=window))

    # 1. Hidden Size
    window.hidden_size_slider = QSlider(Qt.Horizontal)
    window.hidden_size_slider.setRange(256, 16384)
    window.hidden_size_slider.setValue(4096)
    window.hidden_size = window._spin(256, 16384, 4096)
    window.hidden_size_slider.valueChanged.connect(window.hidden_size.setValue)
    window.hidden_size.valueChanged.connect(window.hidden_size_slider.setValue)
    window._tip(window.hidden_size, "Hidden dimension size (n_embd). Model embedding width; higher values increase capacity and expressiveness.")
    window._tip(window.hidden_size_slider, "Slide to adjust hidden embedding dimension (256 to 16,384).")
    form_layout.addWidget(_form_row("Hidden Size (n_embd)", _create_slider_spin_row(window.hidden_size_slider, window.hidden_size), label_width=155, window=window))

    # 2. Heads
    window.num_heads_slider = QSlider(Qt.Horizontal)
    window.num_heads_slider.setRange(1, 128)
    window.num_heads_slider.setValue(32)
    window.num_heads = window._spin(1, 128, 32)
    window.num_heads_slider.valueChanged.connect(window.num_heads.setValue)
    window.num_heads.valueChanged.connect(window.num_heads_slider.setValue)
    window._tip(window.num_heads, "Number of attention query heads (n_head). Must divide hidden size evenly.")
    window._tip(window.num_heads_slider, "Slide to adjust attention head count (1 to 128).")
    form_layout.addWidget(_form_row("Heads (n_head)", _create_slider_spin_row(window.num_heads_slider, window.num_heads), label_width=155, window=window))

    # 3. Layers
    window.num_layers_slider = QSlider(Qt.Horizontal)
    window.num_layers_slider.setRange(1, 128)
    window.num_layers_slider.setValue(44)
    window.num_layers = window._spin(1, 128, 44)
    window.num_layers_slider.valueChanged.connect(window.num_layers.setValue)
    window.num_layers.valueChanged.connect(window.num_layers_slider.setValue)
    window._tip(window.num_layers, "Number of stacked transformer layers (n_layer / depth).")
    window._tip(window.num_layers_slider, "Slide to adjust transformer layer count (1 to 128).")
    form_layout.addWidget(_form_row("Layers (n_layer)", _create_slider_spin_row(window.num_layers_slider, window.num_layers), label_width=155, window=window))

    # 4. Vocab Size
    window.vocab_size_slider = QSlider(Qt.Horizontal)
    window.vocab_size_slider.setRange(256, 128000)
    window.vocab_size_slider.setValue(50257)
    window.vocab_size = window._spin(256, 128000, 50257)
    window.vocab_size_slider.valueChanged.connect(window.vocab_size.setValue)
    window.vocab_size.valueChanged.connect(window.vocab_size_slider.setValue)
    window._tip(window.vocab_size, "Vocabulary size: Total token count in embedding and unembedding projection tables.")
    window._tip(window.vocab_size_slider, "Slide to adjust model vocabulary size (256 to 128,000).")
    form_layout.addWidget(_form_row("Vocab Size", _create_slider_spin_row(window.vocab_size_slider, window.vocab_size), label_width=155, window=window))

    # 5. RoPE Theta
    window.rope_theta_slider = QSlider(Qt.Horizontal)
    window.rope_theta_slider.setRange(1000, 1_000_000)
    window.rope_theta_slider.setValue(10000)
    window.rope_theta = window._double_spin(1.0, 10_000_000.0, 10000.0, 1000.0, 1)
    window.rope_theta_slider.valueChanged.connect(lambda v: window.rope_theta.setValue(float(v)))
    window.rope_theta.valueChanged.connect(lambda v: window.rope_theta_slider.setValue(int(v)))
    window._tip(window.rope_theta, "Rotary Position Embedding base frequency theta. Standard is 10000.0; extended context models use 500000.0+.")
    window._tip(window.rope_theta_slider, "Slide to adjust RoPE theta base frequency.")
    form_layout.addWidget(_form_row("RoPE Theta", _create_slider_spin_row(window.rope_theta_slider, window.rope_theta), label_width=155, window=window))

    # 6. Context Len
    window.context_length_slider = QSlider(Qt.Horizontal)
    window.context_length_slider.setRange(128, 32768)
    window.context_length_slider.setValue(2048)
    window.context_length = window._spin(16, 1_000_000, 2048)
    window.context_length_slider.valueChanged.connect(window.context_length.setValue)
    window.context_length.valueChanged.connect(window.context_length_slider.setValue)
    window._tip(window.context_length, "Context length / sequence length: Maximum number of consecutive tokens processed in each training sequence.")
    window._tip(window.context_length_slider, "Slide to adjust sequence context window length.")
    form_layout.addWidget(_form_row("Context Len", _create_slider_spin_row(window.context_length_slider, window.context_length), label_width=155, window=window))

    # 7. Intermediate Size (MLP Dim)
    window.intermediate_size_slider = QSlider(Qt.Horizontal)
    window.intermediate_size_slider.setRange(256, 65536)
    window.intermediate_size_slider.setValue(11008)
    window.intermediate_size = window._spin(256, 131072, 11008)
    window.intermediate_size_slider.valueChanged.connect(window.intermediate_size.setValue)
    window.intermediate_size.valueChanged.connect(window.intermediate_size_slider.setValue)
    window._tip(window.intermediate_size, "Intermediate dimension of MLP/FFN block. Typically 2.5x to 4x of hidden dimension.")
    window._tip(window.intermediate_size_slider, "Slide to adjust feedforward intermediate dimension.")
    form_layout.addWidget(_form_row("Intermediate Size ⓘ", _create_slider_spin_row(window.intermediate_size_slider, window.intermediate_size), label_width=155, window=window))

    # 8. Attention Type
    window.attention_type = QComboBox()
    window.attention_type.addItems(["Multi-Head (MHA)", "Grouped-Query (GQA)", "Multi-Query (MQA)"])
    window._tip(window.attention_type, "Attention mechanism: Multi-Head (MHA), Grouped-Query (GQA) for memory savings, or Multi-Query (MQA).")
    form_layout.addWidget(_form_row("Attention Type ⓘ", window.attention_type, label_width=155, window=window))

    # 9. KV Heads (for GQA)
    window.kv_head_count_slider = QSlider(Qt.Horizontal)
    window.kv_head_count_slider.setRange(1, 128)
    window.kv_head_count_slider.setValue(8)
    window.kv_head_count = window._spin(1, 128, 8)
    window.kv_head_count_slider.valueChanged.connect(window.kv_head_count.setValue)
    window.kv_head_count.valueChanged.connect(window.kv_head_count_slider.setValue)
    window._tip(window.kv_head_count, "Key/Value head count for GQA. e.g. 8 KV heads with 32 Query heads reduces KV cache memory 4x.")
    window._tip(window.kv_head_count_slider, "Slide to adjust KV head count.")
    form_layout.addWidget(_form_row("KV Heads (n_kv_head) ⓘ", _create_slider_spin_row(window.kv_head_count_slider, window.kv_head_count), label_width=155, window=window))

    # 10. Activation Fn & Normalization
    arch_widget = QWidget()
    arch_row = QHBoxLayout(arch_widget)
    arch_row.setContentsMargins(0, 0, 0, 0)
    arch_row.setSpacing(6)
    window.activation_fn = QComboBox()
    window.activation_fn.addItems(["SwiGLU", "GELU", "SiLU"])
    window._tip(window.activation_fn, "Non-linear feedforward activation function (SwiGLU used in Llama/Mistral, GELU in GPT-NeoX).")
    window.norm_type = QComboBox()
    window.norm_type.addItems(["RMSNorm", "LayerNorm"])
    window._tip(window.norm_type, "Normalization layer: RMSNorm (faster, modern standard) or LayerNorm (classic post/pre-norm).")
    arch_row.addWidget(window.activation_fn, 1)
    arch_row.addWidget(window.norm_type, 1)
    form_layout.addWidget(_form_row("Activation & Norm ⓘ", arch_widget, label_width=155, window=window))

    # 11. Sliding Window & Toggles
    toggles_widget = QWidget()
    toggles_row = QHBoxLayout(toggles_widget)
    toggles_row.setContentsMargins(0, 0, 0, 0)
    toggles_row.setSpacing(10)
    window.use_bias = QCheckBox("Use Bias")
    window.use_bias.setChecked(False)
    window._tip(window.use_bias, "Include learned bias parameters in linear layers. Disabled in modern LLMs for cleaner scaling.")
    window.tie_embeddings = QCheckBox("Tie Embeddings")
    window.tie_embeddings.setChecked(True)
    window._tip(window.tie_embeddings, "Share input embedding and output projection weight matrices to reduce parameter footprint.")
    window.attention_window = window._spin(0, 65536, 0)
    window.attention_window.setToolTip("Sliding window size (0 = Full Context attention)")
    window.attention_window.setFixedWidth(70)
    toggles_row.addWidget(window.use_bias)
    toggles_row.addWidget(window.tie_embeddings)
    toggles_row.addStretch(1)
    toggles_row.addWidget(QLabel("Window:"))
    toggles_row.addWidget(window.attention_window)
    form_layout.addWidget(_form_row("Features & Window ⓘ", toggles_widget, label_width=155, window=window))

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
    form_layout.addWidget(_form_row("Stride Window (stride) ⓘ", _create_slider_spin_row(window.sample_stride_slider, window.sample_stride), label_width=155, window=window))

    # 13. Dropout Regularization
    window.dropout_slider = QSlider(Qt.Horizontal)
    window.dropout_slider.setRange(0, 50)
    window.dropout_slider.setValue(10)
    window.dropout = window._double_spin(0.0, 0.9, 0.1, 0.01, 2)
    window.dropout_slider.valueChanged.connect(lambda v: window.dropout.setValue(v / 100.0))
    window.dropout.valueChanged.connect(lambda v: window.dropout_slider.setValue(int(v * 100)))
    window._tip(window.dropout, "Dropout regularization: Probability of dropping activations during training to prevent overfitting.")
    window._tip(window.dropout_slider, "Slide to adjust dropout probability (0.00 to 0.50).")
    form_layout.addWidget(_form_row("Dropout ⓘ", _create_slider_spin_row(window.dropout_slider, window.dropout), label_width=155, window=window))

    params_layout.addLayout(form_layout)

    # Donut Chart for Parameter Weight Breakdown
    donut_box = QFrame()
    donut_box.setObjectName("DonutBox")
    donut_box_layout = QVBoxLayout(donut_box)
    donut_box_layout.setContentsMargins(8, 6, 8, 6)
    donut_box_layout.setSpacing(4)

    donut_title = QLabel("Parameter Weight Breakdown")
    donut_title.setObjectName("SubTitle")
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
    # Card 3: LoRA & Training Optimization Engine & Hardware Runtime
    # -------------------------------------------------------------------------
    lora_card = QFrame()
    lora_card.setObjectName("Card")
    lora_card_outer = QVBoxLayout(lora_card)
    lora_card_outer.setContentsMargins(14, 12, 14, 12)
    lora_card_outer.setSpacing(8)

    card3_scroll = QScrollArea()
    card3_scroll.setWidgetResizable(True)
    card3_scroll.setFrameShape(QFrame.NoFrame)
    card3_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    card3_content = QWidget()
    lora_layout = QVBoxLayout(card3_content)
    lora_layout.setContentsMargins(0, 0, 4, 0)
    lora_layout.setSpacing(10)
    card3_scroll.setWidget(card3_content)
    lora_card_outer.addWidget(card3_scroll, 1)

    # -------------------------------------------------------------------------
    # LoRA Section (Visible in Fine-Tuning mode)
    # -------------------------------------------------------------------------
    window.lora_section_widget = QWidget()
    lora_sec_layout = QVBoxLayout(window.lora_section_widget)
    lora_sec_layout.setContentsMargins(0, 0, 0, 0)
    lora_sec_layout.setSpacing(8)

    lora_title = QLabel("LoRA Adapter Configuration")
    lora_title.setObjectName("SubTitle")
    lora_sec_layout.addWidget(lora_title)

    lora_form = QVBoxLayout()
    lora_form.setSpacing(6)

    # 1. Rank
    window.fine_tune_lora_rank_slider = QSlider(Qt.Horizontal)
    window.fine_tune_lora_rank_slider.setRange(1, 128)
    window.fine_tune_lora_rank_slider.setValue(8)
    window.fine_tune_lora_rank = window._spin(1, 128, 8)
    window.fine_tune_lora_rank_slider.valueChanged.connect(window.fine_tune_lora_rank.setValue)
    window.fine_tune_lora_rank.valueChanged.connect(window.fine_tune_lora_rank_slider.setValue)
    window._tip(window.fine_tune_lora_rank, "LoRA rank (r): Low-rank bottleneck dimension for adapter matrices (typically 8, 16, or 32).")
    window._tip(window.fine_tune_lora_rank_slider, "Slide to adjust LoRA rank dimension.")
    lora_form.addWidget(_form_row("Rank", _create_slider_spin_row(window.fine_tune_lora_rank_slider, window.fine_tune_lora_rank), label_width=110, window=window))

    # 2. Alpha
    window.fine_tune_lora_alpha_slider = QSlider(Qt.Horizontal)
    window.fine_tune_lora_alpha_slider.setRange(1, 256)
    window.fine_tune_lora_alpha_slider.setValue(16)
    window.fine_tune_lora_alpha = window._spin(1, 256, 16)
    window.fine_tune_lora_alpha_slider.valueChanged.connect(window.fine_tune_lora_alpha.setValue)
    window.fine_tune_lora_alpha.valueChanged.connect(window.fine_tune_lora_alpha_slider.setValue)
    window._tip(window.fine_tune_lora_alpha, "LoRA alpha: Scaling factor for LoRA updates (typically 2x of rank).")
    window._tip(window.fine_tune_lora_alpha_slider, "Slide to adjust LoRA alpha scaling factor.")
    lora_form.addWidget(_form_row("Alpha", _create_slider_spin_row(window.fine_tune_lora_alpha_slider, window.fine_tune_lora_alpha), label_width=110, window=window))

    # 3. Dropout
    window.fine_tune_lora_dropout_slider = QSlider(Qt.Horizontal)
    window.fine_tune_lora_dropout_slider.setRange(0, 50)
    window.fine_tune_lora_dropout_slider.setValue(5)
    window.fine_tune_lora_dropout = window._double_spin(0.0, 0.5, 0.05, 0.01, 2)
    window.fine_tune_lora_dropout_slider.valueChanged.connect(lambda v: window.fine_tune_lora_dropout.setValue(v / 100.0))
    window.fine_tune_lora_dropout.valueChanged.connect(lambda v: window.fine_tune_lora_dropout_slider.setValue(int(v * 100)))
    window._tip(window.fine_tune_lora_dropout, "LoRA dropout: Dropout probability applied to LoRA adapter layers during fine-tuning.")
    window._tip(window.fine_tune_lora_dropout_slider, "Slide to adjust LoRA dropout rate.")
    lora_form.addWidget(_form_row("Dropout ⓘ", _create_slider_spin_row(window.fine_tune_lora_dropout_slider, window.fine_tune_lora_dropout), label_width=110, window=window))

    window.lora_rank = window.fine_tune_lora_rank
    window.lora_alpha = window.fine_tune_lora_alpha
    window.lora_dropout = window.fine_tune_lora_dropout

    lora_sec_layout.addLayout(lora_form)

    # Adapter Target Selector
    target_sec_label = QLabel("Adapter Injection Target Selector")
    target_sec_label.setObjectName("SubTitle")
    lora_sec_layout.addWidget(target_sec_label)

    target_pills_widget = QWidget()
    target_pills_row = QHBoxLayout(target_pills_widget)
    target_pills_row.setContentsMargins(0, 0, 0, 0)
    target_pills_row.setSpacing(6)
    window.target_self_attn = QPushButton("● Self-Attn")
    window.target_self_attn.setObjectName("TargetPill")
    window.target_self_attn.setCheckable(True)
    window.target_self_attn.setChecked(True)
    window._tip(window.target_self_attn, "Inject LoRA adapters into Self-Attention query, key, value, and output projection matrices.")

    window.target_mlp = QPushButton("● MLP")
    window.target_mlp.setObjectName("TargetPill")
    window.target_mlp.setCheckable(True)
    window.target_mlp.setChecked(True)
    window._tip(window.target_mlp, "Inject LoRA adapters into Feed-Forward Network / MLP intermediate projections.")

    window.target_layernorm = QPushButton("○ LayerNorm")
    window.target_layernorm.setObjectName("TargetPill")
    window.target_layernorm.setCheckable(True)
    window.target_layernorm.setChecked(False)
    window._tip(window.target_layernorm, "Inject LoRA trainable scale/bias adapters into LayerNorm/RMSNorm blocks.")

    target_pills_row.addWidget(window.target_self_attn)
    target_pills_row.addWidget(window.target_mlp)
    target_pills_row.addWidget(window.target_layernorm)
    lora_sec_layout.addWidget(_form_row("Targets", target_pills_widget, label_width=110, window=window))

    def _sync_lora_targets(*_args: Any) -> None:
        self_attn = window.target_self_attn.isChecked()
        mlp = window.target_mlp.isChecked()
        window.target_self_attn.setText("● Self-Attn" if self_attn else "○ Self-Attn")
        window.target_mlp.setText("● MLP" if mlp else "○ MLP")
        if self_attn and mlp:
            target_str = "Attention + MLP"
        elif mlp:
            target_str = "MLP projections"
        else:
            target_str = "Attention projections"
        if hasattr(window, "lora_targets"):
            window._set_combo_text(window.lora_targets, target_str)

    window.target_self_attn.toggled.connect(_sync_lora_targets)
    window.target_mlp.toggled.connect(_sync_lora_targets)
    _sync_lora_targets()

    window.lora_section_widget.setVisible(False)  # Hidden in Base Model mode
    lora_layout.addWidget(window.lora_section_widget)

    # -------------------------------------------------------------------------
    # OPTIMIZATION ENGINE (Red Box 1 from Image 3)
    # -------------------------------------------------------------------------
    opt_header = QLabel("OPTIMIZATION ENGINE")
    opt_header.setObjectName("SectionLabel")
    lora_layout.addWidget(opt_header)

    opt_form = QVBoxLayout()
    opt_form.setSpacing(6)

    window.epochs = window._spin(1, 10000, 20)
    window._tip(window.epochs, "Number of complete dataset training epochs.")

    window.batch_size = window._spin(1, 1024, 8)
    window._tip(window.batch_size, "Micro-batch size per optimization step.")

    window.training_profile = QComboBox()
    window.training_profile.addItems(["Stable LLM", "Low-memory", "Code fine-tune", "Experimental Lion"])
    window._tip(window.training_profile, "Applies a practical optimizer, scheduler, precision, and regularization profile.")

    window.apply_training_profile_button = QPushButton("Apply Profile")
    window.apply_training_profile_button.setObjectName("SecondaryAction")
    window.apply_training_profile_button.setMaximumWidth(120)
    if hasattr(window, "apply_training_profile"):
        window.apply_training_profile_button.clicked.connect(window.apply_training_profile)
    window._tip(window.apply_training_profile_button, "Apply the selected training profile to the controls below.")

    window.learning_rate = window._double_spin(0.000001, 1.0, 0.0003, 0.0001, 6)
    window._tip(window.learning_rate, "Optimizer learning rate step size.")

    window.weight_decay = window._double_spin(0.0, 1.0, 0.1, 0.01, 4)
    window._tip(window.weight_decay, "L2 weight decay penalty coefficient to regularize weights.")

    window.optimizer_name = QComboBox()
    window.optimizer_name.addItems(["AdamW", "AdamW (8-bit)", "Adam", "Lion", "Adafactor"])
    window._tip(window.optimizer_name, "Optimizer algorithm. AdamW is standard; 8-bit AdamW saves GPU memory.")

    window.scheduler_name = QComboBox()
    window.scheduler_name.addItems(["Cosine decay", "Warmup linear", "Polynomial decay", "One-cycle", "Constant"])
    window._tip(window.scheduler_name, "Learning rate schedule. Cosine decay with warmup is recommended.")

    window.min_lr_ratio = window._double_spin(0.0, 1.0, 0.1, 0.01, 3)
    window._tip(window.min_lr_ratio, "Lowest learning-rate multiplier after decay (e.g. 0.100 = 10% of base LR).")

    window.polynomial_power = window._double_spin(0.1, 10.0, 1.0, 0.1, 2)
    window._tip(window.polynomial_power, "Shape of polynomial decay curve.")

    window.gradient_accumulation = window._spin(1, 256, 4)
    window._tip(window.gradient_accumulation, "Gradient accumulation steps: Simulates larger effective batch size.")

    window.max_grad_norm = window._double_spin(0.1, 100.0, 1.0, 0.1, 3)
    window._tip(window.max_grad_norm, "Maximum gradient norm clipping threshold to prevent exploding gradients.")

    window.warmup_steps = window._spin(0, 1_000_000, 500)
    window._tip(window.warmup_steps, "Steps used to ramp up learning rate from 0.")

    if not hasattr(window, "sample_stride"):
        window.sample_stride = window._spin(1, 4096, 256)
    window._tip(window.sample_stride, "Token stride samples between sliding windows.")

    window.eval_interval = window._spin(0, 1_000_000, 250)
    window._tip(window.eval_interval, "Training steps between validation checks.")

    window.max_eval_batches = window._spin(0, 1_000_000, 25)
    window._tip(window.max_eval_batches, "Maximum validation batches per validation check.")

    window.save_interval = window._spin(1, 1_000_000, 1000)
    window._tip(window.save_interval, "Training steps between checkpoint saves.")

    window.data_loader_workers = window._spin(0, 64, 4)
    window._tip(window.data_loader_workers, "CPU worker processes used to prepare training batches.")

    window.compile_model = QCheckBox("Torch compile")
    window.compile_model.setChecked(False)
    window._tip(window.compile_model, "Compile model using PyTorch Inductor for kernel fusion.")

    window.seed = window._spin(1, 2_147_483_647, 1337)
    window._tip(window.seed, "Random seed for reproducible initialization and sampling order.")

    opt_form.addWidget(_form_row("Epochs", _paired_row(window.epochs, "Batch", window.batch_size), label_width=110, window=window))
    opt_form.addWidget(_form_row("Profile", _paired_row(window.training_profile, "", window.apply_training_profile_button, stretch1=3, stretch2=2), label_width=110, window=window))
    opt_form.addWidget(_form_row("LR", _paired_row(window.learning_rate, "Decay", window.weight_decay), label_width=110, window=window))
    opt_form.addWidget(_form_row("Optimizer", _paired_row(window.optimizer_name, "Schedule", window.scheduler_name), label_width=110, window=window))
    opt_form.addWidget(_form_row("Min LR", _paired_row(window.min_lr_ratio, "Poly power", window.polynomial_power), label_width=110, window=window))
    opt_form.addWidget(_form_row("Grad accum", _paired_row(window.gradient_accumulation, "Max grad", window.max_grad_norm), label_width=110, window=window))
    opt_form.addWidget(_form_row("Stride samples", _paired_row(window.sample_stride, "Warmup", window.warmup_steps), label_width=110, window=window))
    opt_form.addWidget(_form_row("Eval every", _paired_row(window.eval_interval, "Eval batches", window.max_eval_batches), label_width=110, window=window))
    opt_form.addWidget(_form_row("Save every", _paired_row(window.save_interval, "CPU workers", window.data_loader_workers), label_width=110, window=window))
    opt_form.addWidget(_form_row("Kernel fusion", _paired_row(window.compile_model, "Seed", window.seed), label_width=110, window=window))

    lora_layout.addLayout(opt_form)

    # -------------------------------------------------------------------------
    # HARDWARE & RUNTIME CONTROL (Red Box 2 from Image 3)
    # -------------------------------------------------------------------------
    runtime_header = QLabel("HARDWARE & RUNTIME")
    runtime_header.setObjectName("SectionLabel")
    lora_layout.addWidget(runtime_header)

    runtime_form = QVBoxLayout()
    runtime_form.setSpacing(6)

    window.device = QComboBox()
    window._tip(window.device, "Hardware target device (CUDA GPU or CPU).")

    window.device_info = QLabel("Hardware: Detecting...")
    window.device_info.setObjectName("Metric")
    window.device_info.setWordWrap(True)
    window.device_info.setStyleSheet("color: #4ade80; font-weight: 700; font-size: 11px;")
    if hasattr(window, "_configure_device_options"):
        window._configure_device_options()

    window.precision = QComboBox()
    window.precision.addItems(["BF16", "FP16", "FP32"])
    window._tip(window.precision, "Numeric precision mode. BF16 is fast and numerically stable on modern GPUs.")

    window.use_amp = QCheckBox("Mixed precision")
    window.use_amp.setChecked(getattr(window, "use_amp_default", True))
    window._tip(window.use_amp, "Use automatic mixed precision (AMP) to save memory and accelerate GPU training.")

    window.activation_checkpointing = QCheckBox("Activation checkpointing")
    window.activation_checkpointing.setChecked(True)
    window._tip(window.activation_checkpointing, "Recompute activations during backward pass to drastically reduce VRAM usage.")

    window.resume_training = QCheckBox("Resume latest")
    window.resume_training.setChecked(True)
    window._tip(window.resume_training, "Continue from the latest checkpoint if training was interrupted.")

    window.resume_safety = QCheckBox("Safe resume")
    window.resume_safety.setChecked(True)
    window._tip(window.resume_safety, "Verify that dataset tokenizer and model architecture match before resuming.")

    window.early_stopping = QCheckBox("Early stopping")
    window.early_stopping.setChecked(True)
    window._tip(window.early_stopping, "Automatically stop training when validation loss stops improving.")

    window.early_stopping_patience = window._spin(1, 100, 5)
    window._tip(window.early_stopping_patience, "Consecutive validation checks without improvement before early stopping triggers.")
    window.early_stopping.toggled.connect(window.early_stopping_patience.setEnabled)

    window.resume_checkpoint = QLineEdit()
    window.resume_checkpoint.textChanged.connect(
        lambda t: window.fine_tune_checkpoint.setText(t.strip()) if hasattr(window, "fine_tune_checkpoint") else None
    )
    window._tip(window.resume_checkpoint, "Optional specific checkpoint file to resume from.")

    window.resume_check_button = QPushButton("Check Resume")
    window.resume_check_button.setObjectName("SecondaryAction")
    window.resume_check_button.setMaximumWidth(160)
    if hasattr(window, "preview_resume_compatibility"):
        window.resume_check_button.clicked.connect(window.preview_resume_compatibility)
    window._tip(window.resume_check_button, "Inspect checkpoint compatibility before starting training.")

    runtime_form.addWidget(_form_row("Hardware", window.device_info, label_width=110, window=window))
    runtime_form.addWidget(_form_row("Precision", _paired_row(window.use_amp, "Mode", window.precision), label_width=110, window=window))
    runtime_form.addWidget(_form_row("VRAM saver", _paired_row(window.activation_checkpointing, "", None), label_width=110, window=window))
    runtime_form.addWidget(_form_row("Resume", _paired_row(window.resume_training, "", window.resume_safety), label_width=110, window=window))
    runtime_form.addWidget(_form_row("Early stop", _paired_row(window.early_stopping, "Patience", window.early_stopping_patience), label_width=110, window=window))
    runtime_form.addWidget(_form_row("Checkpoint", window._path_row(window.resume_checkpoint, directory=False), label_width=110, window=window))
    runtime_form.addWidget(_form_row("", window.resume_check_button, label_width=110, window=window))

    lora_layout.addLayout(runtime_form)

    # -------------------------------------------------------------------------
    # Launch Target & Actions
    # -------------------------------------------------------------------------
    window.training_launch_target = QComboBox()
    window.training_launch_target.addItems(["Local machine", "Cluster (Local SGD)"])
    window._tip(
        window.training_launch_target,
        "Select training execution target: 'Local machine' trains on local GPU/CPU. 'Cluster (Local SGD)' dispatches a distributed job across the cluster fleet.",
    )
    launch_target_row = _form_row("Launch Target", window.training_launch_target, label_width=110, window=window)
    lora_layout.addWidget(launch_target_row)

    actions_row = QHBoxLayout()
    actions_row.setSpacing(8)
    window.train_button = QPushButton("Start Base Pre-Training")
    window.train_button.setStyleSheet(
        "QPushButton { background: #10b981; color: white; border: none; border-radius: 6px; padding: 8px 14px; font-size: 12px; font-weight: 800; }"
        "QPushButton:hover { background: #059669; }"
    )
    window._tip(window.train_button, "Start model training using the active mode (Base Pre-training or LoRA Fine-tuning) and launch target.")

    window.stop_training_button = QPushButton("Stop")
    window.stop_training_button.setEnabled(False)
    window.stop_training_button.setStyleSheet(
        "QPushButton { background: #dc2626; color: white; border: none; border-radius: 6px; padding: 8px 14px; font-size: 12px; font-weight: 800; }"
        "QPushButton:hover { background: #b91c1c; }"
        "QPushButton:disabled { background: #1e2230; color: #64748b; }"
    )
    if hasattr(window, "stop_training_process"):
        window.stop_training_button.clicked.connect(window.stop_training_process)
    window._tip(window.stop_training_button, "Request a graceful stop and save a resumable checkpoint.")

    window.dry_run_button = QPushButton("Dry Run")
    window.dry_run_button.setObjectName("SecondaryAction")
    window._tip(window.dry_run_button, "Execute a test forward/backward pass with dummy data to verify GPU memory, gradient flow, and execution speed.")
    actions_row.addWidget(window.train_button, 2)
    actions_row.addWidget(window.stop_training_button, 1)
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
            is_tied = window.tie_embeddings.isChecked() if hasattr(window, "tie_embeddings") else True
            embed_params = v * h if is_tied else (2.0 * v * h)

            # Attention params per layer:
            # Q projection: h * h
            # Output projection: h * h
            # K and V projections: 2 * (h * kv_dim) where kv_dim = h * (kv_heads / num_heads)
            num_heads = float(window.num_heads.value()) if hasattr(window, "num_heads") else 32.0
            attn_text = window.attention_type.currentText().lower() if hasattr(window, "attention_type") else "mha"
            if "grouped" in attn_text or "gqa" in attn_text:
                kv_heads = float(window.kv_head_count.value()) if hasattr(window, "kv_head_count") else max(1.0, num_heads / 4.0)
            elif "multi-query" in attn_text or "mqa" in attn_text:
                kv_heads = 1.0
            else:
                kv_heads = num_heads

            kv_ratio = min(1.0, max(0.01, kv_heads / max(1.0, num_heads)))
            attn_layer_params = (2.0 + 2.0 * kv_ratio) * h * h

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
    window.num_heads.valueChanged.connect(recalculate_parameters)
    if hasattr(window, "intermediate_size"):
        window.intermediate_size.valueChanged.connect(recalculate_parameters)
    if hasattr(window, "activation_fn"):
        window.activation_fn.currentTextChanged.connect(recalculate_parameters)
    if hasattr(window, "tie_embeddings"):
        window.tie_embeddings.toggled.connect(recalculate_parameters)
    if hasattr(window, "attention_type"):
        window.attention_type.currentTextChanged.connect(recalculate_parameters)
    if hasattr(window, "kv_head_count"):
        window.kv_head_count.valueChanged.connect(recalculate_parameters)
    recalculate_parameters()

    def _execute_architecture_dry_run() -> None:
        try:
            vocab = int(window.vocab_size.value()) if hasattr(window, "vocab_size") else 32000
            m_cfg = window._current_model_config(vocab_size=vocab)
            t_cfg = window._current_training_config(training_mode="pretrain")

            import time
            import torch
            from engine.model_gpt import MicroGPT

            device_str = t_cfg.device
            if not torch.cuda.is_available() and device_str.startswith("cuda"):
                device_str = "cpu"

            test_model = MicroGPT(m_cfg).to(device_str)
            test_model.train()

            b = min(2, t_cfg.batch_size)
            seq = min(64, m_cfg.context_length)
            dummy_x = torch.randint(0, m_cfg.vocab_size, (b, seq), device=device_str)

            t0 = time.perf_counter()
            logits = test_model(dummy_x)
            loss = logits.sum()
            loss.backward()
            dt = (time.perf_counter() - t0) * 1000.0

            vram_info = ""
            if device_str.startswith("cuda") and torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                vram_info = f" | VRAM: {alloc:.1f} MB"
                torch.cuda.empty_cache()

            del test_model
            msg = f"✓ Dry Run passed: Forward+Backward in {dt:.1f}ms (batch={b}, seq={seq}){vram_info}. Tensor dimensions and gradients nominal."
            if hasattr(window, "training_log"):
                window.training_log.append(f"<span style='color:#10b981; font-weight:bold;'>{msg}</span>")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(window, "Dry Run Verified", msg)
        except Exception as exc:
            err_msg = f"✗ Dry Run failed: {exc}"
            if hasattr(window, "training_log"):
                window.training_log.append(f"<span style='color:#ef4444; font-weight:bold;'>{err_msg}</span>")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(window, "Dry Run Failed", err_msg)

    if hasattr(window, "dry_run_button"):
        window.dry_run_button.clicked.connect(_execute_architecture_dry_run)

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

    if not hasattr(window, "benchmark_prompts"):
        window.benchmark_prompts = QTextEdit()
    window.benchmark_tokens = window._spin(1, 1000, 100)
    window.benchmark_temperature = window._double_spin(0.0, 2.0, 0.7, 0.05, 2)
    window.benchmark_kv_cache = QCheckBox()
    window.benchmark_kv_cache.setChecked(True)

    return page
