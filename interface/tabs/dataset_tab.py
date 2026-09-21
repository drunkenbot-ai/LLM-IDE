"""Data Ingestion Matrix & Tokenizer screen for DrunkenBot IDE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from engine.conversation_datasets import CONVERSATION_DATASET_PRESETS
from interface.charts import DatasetBarChartWidget

try:
    import psutil
except ImportError:
    psutil = None


def build_dataset_tab(window: Any) -> QWidget:
    """Build the dataset ingestion and streaming tokenizer page.

    Faithfully restores the complete two-column matrix layout, metric chips,
    Source Array, Tokenizer Core, and Dataset Statistics charts.

    Args:
        window: Main application window holding shared state.

    Returns:
        Dataset page widget.
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
    layout.setContentsMargins(18, 16, 18, 10)
    layout.setSpacing(10)
    scroll.setWidget(content)
    outer.addWidget(scroll, 1)

    # =========================================================================
    # Header: Title + Quality Metric Chips Bar
    # =========================================================================
    title_row = QHBoxLayout()
    title_row.setSpacing(8)
    title = window._page_title("Data Ingestion Matrix")
    title_row.addWidget(title, 0)

    window.dataset_quality_samples = window._metric_chip("Documents: -", "Prepared source documents before token sliding windows.")
    window.dataset_quality_tokens = window._metric_chip("Tokens: -", "Total encoded tokens available for training.")
    window.dataset_quality_windows = window._metric_chip("Windows: -", "Sliding context windows the trainer can sample.")
    window.dataset_quality_vocab = window._metric_chip("Vocab: -", "Tokenizer vocabulary size used by the dataset.")
    window.dataset_quality_rating = window._metric_chip("Rating: -", "Five-star dataset quality score based on tokens, windows, vocabulary, diversity, and extraction health.")
    window.dataset_quality_code = window._metric_chip("Code/prose: -", "Code and prose sample split.")
    window.dataset_quality_balance = window._metric_chip("Balance: -", "Code/prose balance detected during preview or preparation.")
    window.dataset_quality_readiness = window._metric_chip("Readiness: -", "Training readiness score based on size, duplicates, extraction quality, and dataset mix.")
    window.dataset_quality_cache = window._metric_chip("Cache: -", "Files reused from cache versus processed this run.")
    window.dataset_quality_duplicates = window._metric_chip("Duplicates: -", "Likely exact or extracted-text duplicate files.")
    window.dataset_quality_extraction = window._metric_chip("Extraction: -", "Files with suspicious text extraction quality.")
    window.dataset_quality_warning = window._metric_chip("Warnings: none", "Dataset quality warnings, if any.")

    header_quality_items = [
        window.dataset_quality_samples,
        window.dataset_quality_tokens,
        window.dataset_quality_windows,
        window.dataset_quality_vocab,
        window.dataset_quality_rating,
        window.dataset_quality_code,
        window.dataset_quality_readiness,
        window.dataset_quality_warning,
    ]
    for item in header_quality_items:
        item.setMaximumWidth(210)
        title_row.addWidget(item, 1)
    layout.addLayout(title_row)

    # Active Dataset Recipe Status Banner
    recipe_banner = QFrame()
    recipe_banner.setObjectName("RecipeBanner")
    recipe_banner.setStyleSheet(
        "#RecipeBanner {"
        "  background-color: #171724;"
        "  border: 1px solid #312e81;"
        "  border-left: 4px solid #6366f1;"
        "  border-radius: 6px;"
        "  margin-top: 2px;"
        "}"
    )
    banner_layout = QHBoxLayout(recipe_banner)
    banner_layout.setContentsMargins(12, 6, 12, 6)
    banner_layout.setSpacing(10)

    window.dataset_tab_recipe_icon = QLabel("🥣")
    window.dataset_tab_recipe_icon.setStyleSheet("font-size: 15px;")
    banner_layout.addWidget(window.dataset_tab_recipe_icon)

    window.dataset_tab_recipe_label = QLabel("ACTIVE RECIPE: Default 11-Pillar Frontier Base • 11 Categories • 250M Target Tokens")
    window.dataset_tab_recipe_label.setStyleSheet("font-weight: bold; color: #e0e7ff; font-size: 12px;")
    banner_layout.addWidget(window.dataset_tab_recipe_label, 1)

    window.dataset_tab_recipe_status = QLabel("Balanced (100.0%)")
    window.dataset_tab_recipe_status.setStyleSheet("background-color: #064e3b; color: #34d399; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
    banner_layout.addWidget(window.dataset_tab_recipe_status)

    recipe_config_btn = QPushButton("Configure Recipe Matrix ➔")
    recipe_config_btn.setStyleSheet(
        "QPushButton { background-color: #4f46e5; color: white; border: none; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 11px; }"
        "QPushButton:hover { background-color: #4338ca; }"
    )
    recipe_config_btn.setToolTip("Open the Dataset Recipe Matrix page to configure category percentages, ratios, and token allocations.")
    recipe_config_btn.clicked.connect(lambda: window._switch_page(1) if hasattr(window, "_switch_page") else None)
    banner_layout.addWidget(recipe_config_btn)
    layout.addWidget(recipe_banner)

    def update_recipe_banner() -> None:
        rec = getattr(window, "active_dataset_recipe", None)
        if rec:
            enabled_count = len([c for c in rec.categories if c.enabled])
            t_str = f"{rec.total_target_tokens / 1_000_000:.0f}M" if rec.total_target_tokens >= 1_000_000 else f"{rec.total_target_tokens:,}"
            window.dataset_tab_recipe_label.setText(
                f"ACTIVE RECIPE: {rec.name}  •  {enabled_count} Active Categories  •  {t_str} Target Token Budget"
            )
            if rec.is_balanced():
                window.dataset_tab_recipe_status.setText(f"Balanced ({rec.total_percentage():.1f}%)")
                window.dataset_tab_recipe_status.setStyleSheet("background-color: #064e3b; color: #34d399; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
            else:
                window.dataset_tab_recipe_status.setText(f"Unbalanced ({rec.total_percentage():.1f}%)")
                window.dataset_tab_recipe_status.setStyleSheet("background-color: #451a03; color: #fbbf24; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        else:
            window.dataset_tab_recipe_label.setText("ACTIVE RECIPE: Default 11-Pillar Frontier Base")
            window.dataset_tab_recipe_status.setText("Ready")

    window.update_dataset_tab_recipe_banner = update_recipe_banner
    update_recipe_banner()

    # =========================================================================
    # Two-Column Body Layout
    # =========================================================================
    ingestion_body = QHBoxLayout()
    ingestion_body.setSpacing(14)
    left_column = QVBoxLayout()
    left_column.setSpacing(10)
    right_column = QVBoxLayout()
    right_column.setSpacing(10)

    # -------------------------------------------------------------------------
    # LEFT COLUMN: SOURCE ARRAY & INGEST TELEMETRY
    # -------------------------------------------------------------------------
    source_form = QFormLayout()
    window._configure_form(source_form)

    window.input_dir = QLineEdit()
    window._tip(window.input_dir, "Source vault: Folder containing raw text, PDFs, Markdown, source code, or JSONL files.")
    window.dataset_dir = QLineEdit(str(Path.cwd() / "runs" / "dataset"))
    window._tip(window.dataset_dir, "Dataset core: Output destination folder where prepared shards, tokens, and metadata are saved.")

    window.max_workers = window._spin(1, 64, 4)
    window._tip(
        window.max_workers,
        "Parallel extraction lanes: Number of CPU worker processes extracting source files simultaneously.",
    )

    window.prepare_mode = QComboBox()
    window.prepare_mode.addItems(["Incremental update", "Full rebuild", "Force reprocess"])
    window.prepare_mode.setMaximumWidth(260)
    window._tip(
        window.prepare_mode,
        "Prepare mode: 'Full rebuild' regenerates tokenizer and shards from scratch. 'Incremental' updates only new files. 'Force reprocess' ignores cached extraction.",
    )

    available_gb = 2.0
    if psutil is not None:
        try:
            available_gb = max(0.5, round(psutil.virtual_memory().available * 0.8 / (1024**3), 1))
        except Exception:
            available_gb = 8.0
    window.tokenizer_training_max_gb = window._double_spin(0.0, 256.0, available_gb, 0.5, 1)
    window._tip(
        window.tokenizer_training_max_gb,
        "Tokenizer training cap (GiB): Maximum corpus sample size shown to the tokenizer trainer to build vocabulary without excessive RAM consumption.",
    )

    window.code_training_mode = QCheckBox("Code-aware processing")
    window.code_training_mode.setChecked(False)
    window.include_source_code = QCheckBox("Include source files")
    window.include_source_code.setChecked(True)

    source_pipeline_row = QWidget()
    source_pipeline_layout = QHBoxLayout(source_pipeline_row)
    source_pipeline_layout.setContentsMargins(0, 0, 0, 0)
    source_pipeline_layout.setSpacing(8)
    lanes_label = QLabel("Parallel lanes")
    lanes_label.setMinimumWidth(85)
    mode_label = QLabel("Prepare mode")
    mode_label.setMinimumWidth(85)
    source_pipeline_layout.addWidget(lanes_label)
    source_pipeline_layout.addWidget(window.max_workers, 1)
    source_pipeline_layout.addWidget(mode_label)
    source_pipeline_layout.addWidget(window.prepare_mode, 2)
    source_form.addRow("Pipeline", source_pipeline_row)
    source_form.addRow("Tokenizer training cap (GiB)", window.tokenizer_training_max_gb)

    source_card = window._card("SOURCE ARRAY", source_form)
    source_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    left_column.addWidget(source_card, 0)

    # Ingest Telemetry Monospace Terminal Box
    window.dataset_log = QTextEdit()
    window.dataset_log.setReadOnly(True)
    window.dataset_log.document().setMaximumBlockCount(1200)
    window.dataset_log.setMinimumHeight(320)
    window.dataset_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    window._tip(window.dataset_log, "Real-time streaming telemetry and audit logs during dataset preparation.")
    log_layout = QVBoxLayout()
    log_layout.addWidget(window.dataset_log, 1)
    left_column.addWidget(window._card("INGEST TELEMETRY", log_layout), 1)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: TOKENIZER CORE
    # -------------------------------------------------------------------------
    tokenizer_form = QFormLayout()
    window._configure_form(tokenizer_form)

    window.auto_vocab = QCheckBox("Choose automatically")
    window.auto_vocab.setChecked(True)
    window._tip(window.auto_vocab, "Auto vocabulary: Automatically derives the optimal vocabulary size based on corpus token density.")

    window.manual_vocab_size = window._spin(256, 100000, 8000)
    window.manual_vocab_size.setEnabled(False)
    window._tip(window.manual_vocab_size, "Manual vocabulary: Fixed target vocabulary size (e.g. 8,000, 32,000, or 50,257).")
    window.auto_vocab.toggled.connect(lambda checked: window.manual_vocab_size.setEnabled(not checked and not window._tokenizer_strategy_reuses()))

    window.auto_vocab_label = QLabel("32,000")
    window.auto_vocab_label.setObjectName("Metric")
    window.auto_vocab_label.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 12px;")
    window._tip(window.auto_vocab_label, "Selected vocab: Currently derived or configured tokenizer vocabulary size.")

    window.tokenizer_strategy = QComboBox()
    window.tokenizer_strategy.addItems(["Auto", "Train new tokenizer", "Reuse dataset tokenizer", "Import tokenizer.json"])
    window.tokenizer_strategy.setMaximumWidth(260)
    window._tip(
        window.tokenizer_strategy,
        "Tokenizer policy: Auto, train freshly from corpus, reuse existing tokenizer, or import an external tokenizer.json.",
    )

    window.tokenizer_path = QLineEdit()
    window.tokenizer_path.setEnabled(False)
    window._tip(window.tokenizer_path, "Import tokenizer: Filepath to an existing tokenizer.json file to reuse.")
    window.tokenizer_path_row = window._path_row(window.tokenizer_path, directory=False, file_filter="Tokenizer JSON (*.json);;All files (*)")
    window.tokenizer_path_row.setEnabled(False)
    window.tokenizer_strategy.currentTextChanged.connect(window._update_tokenizer_strategy_controls)

    window.min_frequency = window._spin(1, 1000, 2)
    window._tip(window.min_frequency, "Min frequency: Minimum occurrence threshold for subwords to be included in vocabulary.")

    context_max = 1000 if not bool(QApplication.instance().property("license_valid")) else 1_000_000
    window.context_length = window._spin(16, context_max, 4096)
    window._tip(window.context_length, "Context window: Token window length used for chunking sequences during ingestion.")

    window.validation_split = window._double_spin(0.0, 0.5, 0.1, 0.01, 3)
    window._tip(window.validation_split, "Validation split: Fraction of tokens held out into val_tokens.npy for validation loss checks.")

    # Background attributes maintained for compatibility with dataset config serializers
    window.include_prose = QCheckBox("Include explanations")
    window.include_prose.setChecked(True)
    window.extract_code_blocks = QCheckBox("Extract code blocks")
    window.extract_code_blocks.setChecked(False)
    window.preserve_indentation = QCheckBox("Preserve indentation")
    window.preserve_indentation.setChecked(True)
    window.instruction_samples = QCheckBox("Instruction style samples")
    window.instruction_samples.setChecked(False)
    window.reasoning_sample_mode = QComboBox()
    window.reasoning_sample_mode.addItems(["No reasoning wrapper", "Reasoning scaffold", "Detailed code reasoning"])

    tokenizer_form.addRow("Auto vocabulary", window.auto_vocab)
    tokenizer_form.addRow("Manual vocabulary", window.manual_vocab_size)
    tokenizer_form.addRow("Selected vocab", window.auto_vocab_label)
    tokenizer_form.addRow("Tokenizer policy", window.tokenizer_strategy)
    tokenizer_form.addRow("Import tokenizer", window.tokenizer_path_row)
    tokenizer_form.addRow("Min frequency", window.min_frequency)
    tokenizer_form.addRow("Context window", window.context_length)
    tokenizer_form.addRow("Validation split", window.validation_split)

    tokenizer_card = window._card("TOKENIZER CORE", tokenizer_form)
    tokenizer_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    right_column.addWidget(tokenizer_card, 0)

    # Detached chart widgets preserved for telemetry callbacks without UI rendering
    window.dataset_mix_chart = DatasetBarChartWidget("Dataset Composition", "Percent")
    window.dataset_sequence_chart = DatasetBarChartWidget("Token Distribution", "Tokens")

    # Dataset Advisor (preserved for DatasetQualityMixin cleanup suggestions)
    window.dataset_advisor = QTextEdit()
    window.dataset_advisor.setReadOnly(True)
    window.dataset_advisor.setMinimumHeight(100)
    window.dataset_advisor.setPlainText("Run Preview Dataset to get cleanup suggestions.")
    window._tip(window.dataset_advisor, "Actionable dataset cleanup advice from preview quality checks.")
    advisor_layout = QVBoxLayout()
    advisor_layout.addWidget(window.dataset_advisor)
    advisor_card = window._card("DATASET ADVISOR", advisor_layout)
    advisor_card.setVisible(False)  # Compact or toggled on demand
    right_column.addWidget(advisor_card)

    # Bottom Action Row
    action_row = QHBoxLayout()
    action_row.setSpacing(10)

    window.health_check_button = QPushButton("Check Health")
    window._tip(window.health_check_button, "Validate source, dataset, model, export, GGUF, and hardware readiness before long work.")
    window.health_check_button.clicked.connect(window.check_project_health)
    window.health_check_button.setMaximumWidth(140)

    window.preview_dataset_button = QPushButton("Preview Dataset")
    window._tip(window.preview_dataset_button, "Scan source files and show dataset quality plus sample text/code snippets without preparing tokens.")
    window.preview_dataset_button.clicked.connect(window.preview_dataset)
    window.preview_dataset_button.setMaximumWidth(160)

    window.prepare_button = QPushButton("Prepare Dataset")
    window.prepare_button.setStyleSheet(
        "QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f59e0b, stop:1 #d97706); color: #000; font-weight: bold; padding: 10px 20px; border-radius: 6px; }"
        "QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #fbbf24, stop:1 #f59e0b); }"
    )
    window._tip(window.prepare_button, "Read source files, clean text, train tokenizer, split tokens, and save the dataset project.")
    window.prepare_button.clicked.connect(window.prepare_dataset)
    window.prepare_button.setMaximumWidth(280)

    window.stop_dataset_button = QPushButton("Stop")
    window.stop_dataset_button.setEnabled(False)
    window.stop_dataset_button.setMaximumWidth(100)
    window.stop_dataset_button.clicked.connect(window.stop_active_task)
    window._tip(window.stop_dataset_button, "Request a graceful stop for dataset preparation.")

    action_row.addWidget(window.health_check_button)
    action_row.addWidget(window.preview_dataset_button)
    action_row.addWidget(window.prepare_button, 1)
    action_row.addWidget(window.stop_dataset_button)
    action_row.addStretch(1)
    right_column.addLayout(action_row)

    right_column.addStretch(1)
    ingestion_body.addLayout(left_column, 1)
    ingestion_body.addLayout(right_column, 1)
    layout.addLayout(ingestion_body, 1)

    window.dataset_progress = window._thin_progress()
    outer.addWidget(window.dataset_progress)

    if hasattr(window, "_update_online_dataset_stage_controls"):
        window._update_online_dataset_stage_controls()

    return page
