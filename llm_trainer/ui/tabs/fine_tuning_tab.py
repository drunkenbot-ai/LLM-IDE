from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def build_fine_tuning_tab(window) -> QWidget:
    """Build the dedicated fine-tuning page.

    Returns:
        Fine-tuning page widget.
    """

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
    layout = QVBoxLayout(content)
    layout.setContentsMargins(18, 18, 18, 10)
    layout.setSpacing(10)
    scroll.setWidget(content)
    outer.addWidget(scroll, 1)
    layout.addWidget(window._page_title("Fine-Tuning Lab"))

    body = QHBoxLayout()
    body.setSpacing(12)
    left_zone = QVBoxLayout()
    left_zone.setSpacing(10)
    right_zone = QVBoxLayout()
    right_zone.setSpacing(10)
    body.addLayout(left_zone, 1)
    body.addLayout(right_zone, 1)
    layout.addLayout(body, 1)

    mode_form = QFormLayout()
    window._configure_form(mode_form)
    window.training_mode = QComboBox()
    window.training_mode.addItems(["Instruction fine-tune", "Conversation fine-tune", "Code fine-tune", "Fine-tune checkpoint"])
    window.training_mode.setMaximumWidth(300)
    window.training_mode.currentTextChanged.connect(window._update_training_mode_controls)
    window.training_mode.currentTextChanged.connect(window.refresh_fine_tune_workflow)
    window.training_mode.currentTextChanged.connect(window._refresh_fine_tune_default_output)
    window._tip(
        window.training_mode,
        "Instruction tunes request-following, conversation tunes chat behavior, code tunes programming help, generic fine-tune adapts domain data.",
    )
    window.fine_tune_checkpoint = QLineEdit()
    window._tip(window.fine_tune_checkpoint, "Base MicroGPT checkpoint used for fine-tuning. Must match tokenizer and model architecture.")
    # Architecture fields (n_embd, n_layer, etc.) are shared with the AI tab
    # and are not automatically kept in sync with whichever checkpoint is
    # selected here. Without this, a mismatched leftover architecture (e.g.
    # a "Tiny" preset still selected from an earlier session) either fails
    # the resume-compatibility check outright, or -- worse -- if the
    # mismatch happens to pass, training silently starts from a randomly
    # initialized model that only coincidentally matches the same shapes,
    # rather than actually continuing from the selected base checkpoint.
    # Syncing on every change to this field (not just via the separate
    # "Apply Recommended" button) makes the common path safe by default.
    window.fine_tune_checkpoint.textChanged.connect(lambda _text: window._sync_architecture_from_fine_tune_base())
    window.fine_tune_output_dir = QLineEdit()
    window._tip(
        window.fine_tune_output_dir,
        "Separate folder where fine-tuned checkpoints, adapters, telemetry, and final tuned model are saved.",
    )
    window.peft_method = QComboBox()
    window.peft_method.addItems(["Full fine-tune", "LoRA adapters"])
    window.peft_method.setMaximumWidth(300)
    window.peft_method.currentTextChanged.connect(window._update_training_mode_controls)
    window._tip(window.peft_method, "Parameter-efficient fine-tuning method. LoRA trains small adapters while freezing the base model.")
    window.lora_rank = window._spin(1, 256, 8)
    window._tip(window.lora_rank, "LoRA rank. Higher values increase adapter capacity and adapter size.")
    window.lora_alpha = window._double_spin(1.0, 512.0, 16.0, 1.0, 1)
    window._tip(window.lora_alpha, "LoRA alpha scaling. Common default is 2x the rank.")
    window.lora_dropout = window._double_spin(0.0, 0.9, 0.05, 0.01, 3)
    window._tip(window.lora_dropout, "Dropout used only inside LoRA adapters.")
    window.lora_targets = QComboBox()
    window.lora_targets.addItems(["Attention projections", "MLP projections", "Attention + MLP"])
    window.lora_targets.setMaximumWidth(300)
    window._tip(window.lora_targets, "Modules where LoRA adapters are attached.")
    window.fine_tune_check_button = QPushButton("Check Fine-tune")
    window.fine_tune_check_button.setMaximumWidth(180)
    window.fine_tune_check_button.clicked.connect(window.preview_fine_tune_compatibility)
    window._tip(window.fine_tune_check_button, "Inspect whether the base checkpoint can be used for fine-tuning.")
    window.fine_tune_dataset_status = QLabel("Dataset: not checked")
    window.fine_tune_dataset_status.setObjectName("Metric")
    window._tip(
        window.fine_tune_dataset_status,
        "Shows whether the prepared dataset purpose matches this fine-tune workflow.",
    )
    window.fine_tune_refresh_button = QPushButton("Refresh Dataset Fit")
    window.fine_tune_refresh_button.setMaximumWidth(190)
    window.fine_tune_refresh_button.clicked.connect(window.refresh_fine_tune_workflow)
    window._tip(window.fine_tune_refresh_button, "Re-read dataset_summary.json and check whether this dataset fits the fine-tune type.")
    window.apply_lora_preset_button = QPushButton("Apply Recommended LoRA")
    window.apply_lora_preset_button.setMaximumWidth(220)
    window.apply_lora_preset_button.clicked.connect(window.apply_recommended_fine_tune_settings)
    window._tip(window.apply_lora_preset_button, "Apply conservative LoRA, learning-rate, and clipping defaults for the selected fine-tune type.")

    mode_form.addRow("Fine-tune type", window.training_mode)
    mode_form.addRow("Base model", window._path_row(window.fine_tune_checkpoint, directory=False))
    mode_form.addRow("Fine-tune output", window._path_row(window.fine_tune_output_dir, directory=True))
    mode_form.addRow("PEFT", window.peft_method)
    mode_form.addRow("LoRA rank", window.lora_rank)
    mode_form.addRow("LoRA alpha", window.lora_alpha)
    mode_form.addRow("LoRA dropout", window.lora_dropout)
    mode_form.addRow("LoRA target", window.lora_targets)
    mode_form.addRow("Dataset fit", window.fine_tune_dataset_status)
    mode_form.addRow("", window.fine_tune_refresh_button)
    mode_form.addRow("", window.apply_lora_preset_button)
    mode_form.addRow("", window.fine_tune_check_button)
    left_zone.addWidget(window._card("ADAPTATION CONTROL", mode_form), 0)

    runtime_form = QFormLayout()
    window._configure_form(runtime_form)
    window.fine_tune_launch_target = QComboBox()
    window.fine_tune_launch_target.addItems(["Local machine", "Remote workers", "RunPod cloud"])
    window.fine_tune_launch_target.setMaximumWidth(300)
    window._tip(
        window.fine_tune_launch_target,
        "Local runs fine-tuning on this computer. Remote queues it for workers. RunPod creates a cloud GPU worker automatically.",
    )
    window.fine_tune_runtime_hint = QLabel("Uses AI tab device, precision, resume, and checkpoint settings.")
    window.fine_tune_runtime_hint.setObjectName("Metric")
    window.fine_tune_runtime_hint.setWordWrap(True)
    runtime_form.addRow("Launch", window.fine_tune_launch_target)
    runtime_form.addRow("Settings", window.fine_tune_runtime_hint)
    left_zone.addWidget(window._card("FINE-TUNE RUNTIME", runtime_form), 0)

    metrics_grid = QGridLayout()
    metrics_grid.setHorizontalSpacing(8)
    metrics_grid.setVerticalSpacing(8)
    window.fine_tune_eta_metric = window._metric_chip("ETA: -", "Estimated time remaining for the fine-tune run.")
    window.fine_tune_epoch_metric = window._metric_chip("Epoch: -", "Current fine-tune epoch and total epochs.")
    window.fine_tune_step_metric = window._metric_chip("Step: -", "Current optimizer step and total planned steps.")
    window.fine_tune_loss_metric = window._metric_chip("Train loss: -", "Latest fine-tune training loss.")
    window.fine_tune_val_metric = window._metric_chip("Val loss: -", "Latest fine-tune validation loss.")
    window.fine_tune_lr_metric = window._metric_chip("LR: -", "Current fine-tune learning rate.")
    window.fine_tune_speed_metric = window._metric_chip("Speed: -", "Fine-tune token throughput.")
    window.fine_tune_grad_metric = window._metric_chip("Grad: -", "Current fine-tune gradient norm.")
    for index, metric in enumerate((
        window.fine_tune_eta_metric,
        window.fine_tune_epoch_metric,
        window.fine_tune_step_metric,
        window.fine_tune_loss_metric,
        window.fine_tune_val_metric,
        window.fine_tune_lr_metric,
        window.fine_tune_speed_metric,
        window.fine_tune_grad_metric,
    )):
        metrics_grid.addWidget(metric, index // 2, index % 2)
    metrics_grid.setColumnStretch(0, 1)
    metrics_grid.setColumnStretch(1, 1)
    left_zone.addWidget(window._card("FINE-TUNE METRICS", metrics_grid), 0)

    builder_form = QFormLayout()
    window._configure_form(builder_form)
    window.fine_tune_dataset_builder_stage = QComboBox()
    window.fine_tune_dataset_builder_stage.addItems(["Instruction fine-tune", "Conversation fine-tune", "Code fine-tune"])
    window.fine_tune_dataset_builder_stage.setMaximumWidth(300)
    window.fine_tune_dataset_builder_stage.currentTextChanged.connect(window._refresh_fine_tune_default_output)
    window._tip(
        window.fine_tune_dataset_builder_stage,
        "Choose the fine-tune dataset type to configure in the Ingest tab.",
    )
    window.configure_fine_tune_dataset_button = QPushButton("Configure Ingest")
    window.configure_fine_tune_dataset_button.setMaximumWidth(180)
    window.configure_fine_tune_dataset_button.clicked.connect(window.configure_fine_tune_dataset_builder)
    window._tip(
        window.configure_fine_tune_dataset_button,
        "Switch to Ingest, set the matching dataset purpose, enable online datasets, and preselect a small starter dataset.",
    )
    builder_form.addRow("Dataset type", window.fine_tune_dataset_builder_stage)
    builder_form.addRow("", window.configure_fine_tune_dataset_button)
    left_zone.addWidget(window._card("FINE-TUNE DATASET BUILDER", builder_form), 0)

    guidance = QTextEdit()
    guidance.setReadOnly(True)
    guidance.setMaximumHeight(130)
    guidance.setPlainText(
        "Fine-tuning starts from a compatible checkpoint and uses the prepared dataset selected in AI.\n"
        "- Prepare Instruction fine-tune data for request-following behavior.\n"
        "- Prepare Conversation fine-tune data for chat behavior.\n"
        "- Prepare Code fine-tune data for programming help.\n"
        "- Reuse or import the base tokenizer so token IDs stay compatible.\n"
        "- LoRA is recommended for most experiments.\n"
        "- Stop saves a resumable checkpoint; Resume latest can continue it."
    )
    left_zone.addWidget(window._card("WORKFLOW", _single_widget_layout(guidance)), 0)

    window.fine_tune_preview = QTextEdit()
    window.fine_tune_preview.setReadOnly(True)
    window.fine_tune_preview.setMinimumHeight(160)
    window.fine_tune_preview.setText("No compatibility check has been run.")
    window._tip(window.fine_tune_preview, "Compatibility report for the selected base checkpoint.")
    right_zone.addWidget(window._card("FINE-TUNE COMPATIBILITY", _single_widget_layout(window.fine_tune_preview)), 0)

    window.fine_tune_log = QTextEdit()
    window.fine_tune_log.setReadOnly(True)
    window.fine_tune_log.document().setMaximumBlockCount(1500)
    window.fine_tune_log.setMinimumHeight(320)
    window.fine_tune_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    right_zone.addWidget(window._card("FINE-TUNE TELEMETRY", _single_widget_layout(window.fine_tune_log)), 1)

    window.fine_tune_button = QPushButton("Start Fine-Tune")
    window.fine_tune_button.setMaximumWidth(220)
    window.fine_tune_button.clicked.connect(window.start_fine_tuning)
    window._tip(window.fine_tune_button, "Start fine-tuning from the selected compatible base checkpoint.")
    window.stop_fine_tune_button = QPushButton("Stop")
    window.stop_fine_tune_button.setEnabled(False)
    window.stop_fine_tune_button.setMaximumWidth(120)
    window.stop_fine_tune_button.clicked.connect(window.stop_active_task)
    action_row = QHBoxLayout()
    action_row.addWidget(window.fine_tune_button)
    action_row.addWidget(window.stop_fine_tune_button)
    action_row.addStretch(1)
    layout.addLayout(action_row)

    window.fine_tune_progress = window._thin_progress()
    outer.addWidget(window.fine_tune_progress)
    QTimer.singleShot(0, window._update_training_mode_controls)
    return page


def _single_widget_layout(widget: QWidget) -> QVBoxLayout:
    """Wrap one widget in a vertical layout.

    Args:
        widget: Widget to place in a layout.

    Returns:
        Layout containing the widget.
    """

    layout = QVBoxLayout()
    layout.addWidget(widget, 1)
    return layout