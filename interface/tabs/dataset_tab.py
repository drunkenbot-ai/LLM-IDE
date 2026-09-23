"""Data Ingestion Matrix & Tokenizer screen for DrunkenBot IDE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
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
    QLayout,
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
from interface.tabs.dataset_plan_tab import PillToggleSwitch

try:
    import psutil
except ImportError:
    psutil = None


def _ingestion_card(title: str, content_layout: QLayout) -> QWidget:
    """Create an obsidian luxury card with warm bronze glowing borders matching concept art."""
    card = QWidget()
    card.setObjectName("IngestionCard")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(16, 14, 16, 14)
    card_layout.setSpacing(10)
    if title:
        title_label = QLabel(title)
        title_label.setObjectName("IngestionCardTitle")
        card_layout.addWidget(title_label)
    card_layout.addLayout(content_layout)
    return card


def build_dataset_tab(window: Any) -> QWidget:
    """Build the Data Ingestion Matrix and streaming tokenizer page.

    Faithfully implements the obsidian luxury dark theme matching the concept art:
    - 8 telemetry chips in header row with gold rating stars and warning badge
    - Active Recipe Banner with emerald status capsule and indigo configure button
    - Two-column layout with Source Array and full-height Ingest Telemetry on left
    - Tokenizer Core, visible Dataset Health & Advisory, and Glowing Action Bar on right
    - Duck-typed PillToggleSwitch for Auto vocabulary

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
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)
    scroll.setWidget(content)
    outer.addWidget(scroll, 1)

    # =========================================================================
    # Header: Title + Quality Metric Chips Bar
    # =========================================================================
    title_row = QHBoxLayout()
    title_row.setSpacing(6)
    title = window._page_title("Data Ingestion Matrix")
    title.setMinimumWidth(210)
    title_row.addWidget(title, 0)

    window.dataset_quality_samples = window._metric_chip("Documents: 142K", "Prepared source documents before token sliding windows.")
    window.dataset_quality_tokens = window._metric_chip("Tokens: 250M", "Total encoded tokens available for training.")
    window.dataset_quality_windows = window._metric_chip("Windows: 61,035", "Sliding context windows the trainer can sample.")
    window.dataset_quality_vocab = window._metric_chip("Vocab: 32,000", "Tokenizer vocabulary size used by the dataset.")
    window.dataset_quality_rating = window._metric_chip('Rating: <span style="color:#fbbf24;">★★★★★</span>', "Five-star dataset quality score based on tokens, windows, vocabulary, diversity, and extraction health.")
    window.dataset_quality_rating.setFont(QFont("Segoe UI Symbol", 9))
    window.dataset_quality_code = window._metric_chip("Code/Prose: 45/55", "Code and prose sample split.")
    window.dataset_quality_balance = window._metric_chip("Balance: -", "Code/prose balance detected during preview or preparation.")
    window.dataset_quality_readiness = window._metric_chip("Readiness: 98%", "Training readiness score based on size, duplicates, extraction quality, and dataset mix.")
    window.dataset_quality_cache = window._metric_chip("Cache: -", "Files reused from cache versus processed this run.")
    window.dataset_quality_duplicates = window._metric_chip("Duplicates: -", "Likely exact or extracted-text duplicate files.")
    window.dataset_quality_extraction = window._metric_chip("Extraction: -", "Files with suspicious text extraction quality.")
    window.dataset_quality_warning = window._metric_chip("Warnings: None", "Dataset quality warnings, if any.")
    window.dataset_quality_warning.setObjectName("MetricChipWarning")

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
        item.setMinimumWidth(0)
        item.setMaximumWidth(150)
        item.setAlignment(Qt.AlignCenter)
        title_row.addWidget(item, 1)
    layout.addLayout(title_row)

    # =========================================================================
    # Active Dataset Recipe Status Banner
    # =========================================================================
    recipe_banner = QFrame()
    recipe_banner.setObjectName("RecipeBanner")
    banner_layout = QHBoxLayout(recipe_banner)
    banner_layout.setContentsMargins(14, 8, 14, 8)
    banner_layout.setSpacing(10)

    window.dataset_tab_recipe_icon = QLabel("🥣")
    window.dataset_tab_recipe_icon.setFont(QFont("Segoe UI Emoji", 12))
    banner_layout.addWidget(window.dataset_tab_recipe_icon, 0)

    window.dataset_tab_recipe_label = QLabel("ACTIVE RECIPE: Default 11-Pillar Frontier Base • 11 Active Categories • 250M Target Tokens")
    window.dataset_tab_recipe_label.setObjectName("RecipeBannerLabel")
    banner_layout.addWidget(window.dataset_tab_recipe_label, 1)

    window.dataset_tab_recipe_status = QLabel("Balanced (100.0%)")
    window.dataset_tab_recipe_status.setObjectName("RecipeBannerStatus")
    window.dataset_tab_recipe_status.setStyleSheet("background-color: #064e3b; color: #34d399; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")
    banner_layout.addWidget(window.dataset_tab_recipe_status, 0)

    recipe_config_btn = QPushButton("Configure Recipe Matrix →")
    recipe_config_btn.setObjectName("RecipeConfigBtn")
    recipe_config_btn.setStyleSheet(
        "QPushButton#RecipeConfigBtn { background-color: #3730a3; color: white; border: 1px solid #4338ca; padding: 5px 14px; border-radius: 6px; font-weight: bold; font-size: 11px; font-family: 'Segoe UI', sans-serif; }"
        "QPushButton#RecipeConfigBtn:hover { background-color: #4338ca; }"
    )
    recipe_config_btn.setToolTip("Open the Dataset Recipe Matrix page to configure category percentages, ratios, and token allocations.")
    recipe_config_btn.clicked.connect(lambda: window._switch_page(1) if hasattr(window, "_switch_page") else None)
    banner_layout.addWidget(recipe_config_btn, 0)
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
                window.dataset_tab_recipe_status.setStyleSheet("background-color: #064e3b; color: #34d399; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")
            else:
                window.dataset_tab_recipe_status.setText(f"Unbalanced ({rec.total_percentage():.1f}%)")
                window.dataset_tab_recipe_status.setStyleSheet("background-color: #451a03; color: #fbbf24; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        else:
            window.dataset_tab_recipe_label.setText("ACTIVE RECIPE: Default 11-Pillar Frontier Base • 11 Active Categories • 250M Target Tokens")
            window.dataset_tab_recipe_status.setText("Balanced (100.0%)")

    window.update_dataset_tab_recipe_banner = update_recipe_banner

    # Helper function for cleanly aligned label-to-input rows matching concept art
    def _form_row(label_text: str, widget: QWidget, tip_text: str = "") -> QWidget:
        row = QWidget()
        row.setObjectName("FormRow")
        h = QHBoxLayout(row)
        h.setContentsMargins(8, 5, 8, 5)
        h.setSpacing(10)
        lbl = QLabel(label_text)
        lbl.setObjectName("IngestionFieldLabel")
        if tip_text:
            window._tip(lbl, tip_text)
            window._tip(widget, tip_text)
        h.addWidget(lbl, 0)
        h.addStretch(1)
        h.addWidget(widget, 0)
        return row

    # =========================================================================
    # Two-Column Body Layout
    # =========================================================================
    ingestion_body = QHBoxLayout()
    ingestion_body.setSpacing(14)
    left_column = QVBoxLayout()
    left_column.setSpacing(12)
    right_column = QVBoxLayout()
    right_column.setSpacing(12)

    # -------------------------------------------------------------------------
    # LEFT COLUMN: SOURCE ARRAY & INGEST TELEMETRY
    # -------------------------------------------------------------------------
    source_layout = QVBoxLayout()
    source_layout.setSpacing(8)

    # Compatibility shims for dataset config persistence
    window.input_dir = QLineEdit()
    window.dataset_dir = QLineEdit(str(Path.cwd() / "runs" / "dataset"))

    window.max_workers = window._spin(1, 64, 4)
    window.max_workers.setFixedWidth(90)

    window.prepare_mode = QComboBox()
    window.prepare_mode.addItems(["Full rebuild", "Incremental update", "Force reprocess"])
    window.prepare_mode.setFixedWidth(210)

    available_gb = 8.6
    if psutil is not None:
        try:
            available_gb = max(0.5, round(psutil.virtual_memory().available * 0.8 / (1024**3), 1))
        except Exception:
            available_gb = 8.6
    window.tokenizer_training_max_gb = window._double_spin(0.0, 256.0, available_gb, 0.5, 1)
    window.tokenizer_training_max_gb.setSuffix(" GiB")
    window.tokenizer_training_max_gb.setFixedWidth(110)

    window.code_training_mode = QCheckBox("Code-aware processing")
    window.code_training_mode.setChecked(False)
    window.include_source_code = QCheckBox("Include source files")
    window.include_source_code.setChecked(True)

    source_layout.addWidget(_form_row("Parallel CPU Extraction Lanes", window.max_workers, "Parallel extraction lanes: Number of CPU worker processes extracting source files simultaneously."))
    source_layout.addWidget(_form_row("Prepare Mode", window.prepare_mode, "Prepare mode: 'Full rebuild' regenerates tokenizer and shards from scratch. 'Incremental' updates only new files."))
    source_layout.addWidget(_form_row("Tokenizer training cap", window.tokenizer_training_max_gb, "Tokenizer training cap: Maximum corpus sample size shown to the tokenizer trainer to build vocabulary."))

    source_card = _ingestion_card("SOURCE ARRAY", source_layout)
    source_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    left_column.addWidget(source_card, 0)

    # Ingest Telemetry Monospace Cyber Terminal Console
    window.dataset_log = QTextEdit()
    window.dataset_log.setObjectName("IngestTelemetryTerminal")
    window.dataset_log.setReadOnly(True)
    window.dataset_log.document().setMaximumBlockCount(2000)
    window.dataset_log.setMinimumHeight(280)
    window.dataset_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    sample_telemetry = (
        '<span style="color:#64748b;">[18:24:51]</span> <span style="color:#38bdf8; font-weight:bold;">[INFO]</span> <span style="color:#e2e8f0;">Initializing multi-process token pipeline</span><br>'
        '<span style="color:#64748b;">[18:24:51]</span> <span style="color:#38bdf8; font-weight:bold;">[INFO]</span> <span style="color:#e2e8f0;">Initializing complete logs</span><br>'
        '<span style="color:#64748b;">[18:24:53]</span> <span style="color:#fbbf24; font-weight:bold;">[AUDIT]</span> <span style="color:#e2e8f0;">Text extraction integrity</span> <span style="color:#38bdf8; font-weight:bold;">99.8%</span><br>'
        '<span style="color:#64748b;">[18:24:53]</span> <span style="color:#fbbf24; font-weight:bold;">[AUDIT]</span> <span style="color:#e2e8f0;">Text extraction integrity</span> <span style="color:#38bdf8; font-weight:bold;">99.8%</span><br>'
        '<span style="color:#64748b;">[18:24:53]</span> <span style="color:#38bdf8; font-weight:bold;">[INFO]</span> <span style="color:#e2e8f0;">Text: extraction pipeline</span><br>'
        '<span style="color:#64748b;">[18:24:54]</span> <span style="color:#38bdf8; font-weight:bold;">[INFO]</span> <span style="color:#e2e8f0;">Finishing token pipeline</span><br>'
        '<span style="color:#64748b;">[18:24:54]</span> <span style="color:#38bdf8; font-weight:bold;">[INFO]</span> <span style="color:#e2e8f0;">Finishing bottern cooline</span><br>'
        '<span style="color:#64748b;">[18:24:55]</span> <span style="color:#fbbf24; font-weight:bold;">[AUDIT]</span> <span style="color:#e2e8f0;">Text extraction integrity</span> <span style="color:#38bdf8; font-weight:bold;">99.8%</span><br>'
        '<span style="color:#64748b;">[18:24:55]</span> <span style="color:#fbbf24; font-weight:bold;">[AUDIT]</span> <span style="color:#e2e8f0;">Text extraction integrity</span> <span style="color:#38bdf8; font-weight:bold;">99.8%</span><br>'
        '<span style="color:#64748b;">[18:24:56]</span> <span style="color:#fbbf24; font-weight:bold;">[AUDIT]</span> <span style="color:#e2e8f0;">Text extraction integrity</span> <span style="color:#38bdf8; font-weight:bold;">99.8%</span><br>'
        '<span style="color:#64748b;">[18:24:56]</span> <span style="color:#fbbf24; font-weight:bold;">[AUDIT]</span> <span style="color:#e2e8f0;">Text extraction integrity</span> <span style="color:#38bdf8; font-weight:bold;">99.8%</span><br>'
        '<span style="color:#64748b;">[18:24:56]</span> <span style="color:#fbbf24; font-weight:bold;">[AUDIT]</span> <span style="color:#e2e8f0;">Text extraction integrity</span> <span style="color:#38bdf8; font-weight:bold;">99.8%</span><br>'
        '<span style="color:#64748b;">[18:24:57]</span> <span style="color:#fbbf24; font-weight:bold;">[AUDIT]</span> <span style="color:#e2e8f0;">Text extraction integrity</span> <span style="color:#38bdf8; font-weight:bold;">99.8%</span><br>'
        '<span style="color:#64748b;">[18:24:57]</span> <span style="color:#fbbf24; font-weight:bold;">[AUDIT]</span> <span style="color:#e2e8f0;">Text extraction integrity</span> <span style="color:#38bdf8; font-weight:bold;">99.8%</span>'
    )
    window.dataset_log.setHtml(sample_telemetry)
    window._tip(window.dataset_log, "Real-time streaming telemetry and audit logs during dataset preparation.")

    log_layout = QVBoxLayout()
    log_layout.setContentsMargins(0, 2, 0, 0)
    log_layout.addWidget(window.dataset_log, 1)
    telemetry_card = _ingestion_card("INGEST TELEMETRY", log_layout)
    telemetry_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    left_column.addWidget(telemetry_card, 1)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: TOKENIZER CORE, HEALTH & ADVISORY, ACTIONS
    # -------------------------------------------------------------------------
    tokenizer_layout = QVBoxLayout()
    tokenizer_layout.setSpacing(8)

    # Row 1: Auto Vocab Switch with Selected Vocab Readout Box
    auto_vocab_row = QWidget()
    auto_h = QHBoxLayout(auto_vocab_row)
    auto_h.setContentsMargins(0, 3, 0, 3)
    auto_h.setSpacing(10)

    auto_lbl = QLabel("Auto vocabulary")
    auto_lbl.setObjectName("IngestionFieldLabel")
    auto_h.addWidget(auto_lbl, 0)
    auto_h.addStretch(1)

    window.auto_vocab = PillToggleSwitch(checked=True)
    window._tip(window.auto_vocab, "Auto vocabulary: Derives the optimal vocabulary size based on corpus token density.")
    auto_h.addWidget(window.auto_vocab, 0)

    auto_h.addSpacing(16)
    sel_lbl = QLabel("Selected vocab")
    sel_lbl.setObjectName("IngestionMutedLabel")
    auto_h.addWidget(sel_lbl, 0)

    window.auto_vocab_label = QLabel("32,000")
    window.auto_vocab_label.setObjectName("VocabReadoutBadge")
    window.auto_vocab_label.setFixedWidth(80)
    window.auto_vocab_label.setAlignment(Qt.AlignCenter)
    window._tip(window.auto_vocab_label, "Selected vocab: Currently derived or configured tokenizer vocabulary size.")
    auto_h.addWidget(window.auto_vocab_label, 0)
    tokenizer_layout.addWidget(auto_vocab_row)

    # Row 2: Manual Vocabulary (disabled when auto_vocab checked)
    window.manual_vocab_size = window._spin(256, 100000, 8000)
    window.manual_vocab_size.setFixedWidth(90)
    window.manual_vocab_size.setEnabled(False)
    window.auto_vocab.toggled.connect(lambda checked: window.manual_vocab_size.setEnabled(not checked and not window._tokenizer_strategy_reuses()))
    tokenizer_layout.addWidget(_form_row("Manual vocabulary", window.manual_vocab_size, "Manual vocabulary: Fixed target vocabulary size (e.g. 8,000, 32,000, or 50,257)."))

    # Row 3: Tokenizer Policy
    window.tokenizer_strategy = QComboBox()
    window.tokenizer_strategy.addItems(["Auto", "Train new tokenizer", "Reuse dataset tokenizer", "Import tokenizer.json"])
    window.tokenizer_strategy.setFixedWidth(210)
    window.tokenizer_strategy.currentTextChanged.connect(window._update_tokenizer_strategy_controls)
    tokenizer_layout.addWidget(_form_row("Tokenizer policy", window.tokenizer_strategy, "Tokenizer policy: Auto, train freshly from corpus, reuse existing tokenizer, or import tokenizer.json."))

    # Row 4: Full-width Import Tokenizer Path Row
    import_row = QWidget()
    import_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    import_h = QHBoxLayout(import_row)
    import_h.setContentsMargins(0, 3, 0, 3)
    import_h.setSpacing(8)

    window.tokenizer_path = QLineEdit()
    window.tokenizer_path.setEnabled(False)
    window.tokenizer_path.setPlaceholderText("Import tokenizer path")
    window.tokenizer_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    window._tip(window.tokenizer_path, "Import tokenizer: Filepath to an existing tokenizer.json file to reuse.")

    browse_btn = QPushButton("Browse")
    browse_btn.setObjectName("IngestionBrowseBtn")
    browse_btn.setFixedWidth(80)
    browse_btn.clicked.connect(lambda: window._browse(window.tokenizer_path, False, "Tokenizer JSON (*.json);;All files (*)"))
    window._tip(browse_btn, "Open a file picker to import an external tokenizer.json file.")

    import_h.addWidget(window.tokenizer_path, 1)
    import_h.addWidget(browse_btn, 0)
    window.tokenizer_path_row = import_row
    window.tokenizer_path_row.setEnabled(False)
    tokenizer_layout.addWidget(import_row)

    # Row 5: 3-Column Compact Core Parameters Row
    params_row = QWidget()
    params_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    params_h = QHBoxLayout(params_row)
    params_h.setContentsMargins(0, 4, 0, 0)
    params_h.setSpacing(14)

    col1 = QVBoxLayout()
    col1.setSpacing(4)
    lbl1 = QLabel("Min frequency")
    lbl1.setObjectName("IngestionParamLabel")
    window.min_frequency = window._spin(1, 1000, 2)
    window.min_frequency.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    window._tip(window.min_frequency, "Min frequency: Minimum occurrence threshold for subwords to be included in vocabulary.")
    col1.addWidget(lbl1)
    col1.addWidget(window.min_frequency)

    col2 = QVBoxLayout()
    col2.setSpacing(4)
    lbl2 = QLabel("Context window")
    lbl2.setObjectName("IngestionParamLabel")
    context_max = 1000 if not bool(QApplication.instance().property("license_valid")) else 1_000_000
    window.context_length = window._spin(16, context_max, 4096)
    window.context_length.setSuffix(" tokens")
    window.context_length.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    window._tip(window.context_length, "Context window: Token window length used for chunking sequences during ingestion.")
    col2.addWidget(lbl2)
    col2.addWidget(window.context_length)

    col3 = QVBoxLayout()
    col3.setSpacing(4)
    lbl3 = QLabel("Validation split")
    lbl3.setObjectName("IngestionParamLabel")
    window.validation_split = window._double_spin(0.0, 0.5, 0.1, 0.01, 3)
    window.validation_split.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    window._tip(window.validation_split, "Validation split: Fraction of tokens held out into val_tokens.npy for validation loss checks.")
    col3.addWidget(lbl3)
    col3.addWidget(window.validation_split)

    params_h.addLayout(col1, 1)
    params_h.addLayout(col2, 1)
    params_h.addLayout(col3, 1)
    tokenizer_layout.addWidget(params_row)

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

    tokenizer_card = _ingestion_card("TOKENIZER CORE", tokenizer_layout)
    tokenizer_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    right_column.addWidget(tokenizer_card, 0)

    # Detached chart widgets preserved for telemetry callbacks without UI rendering
    window.dataset_mix_chart = DatasetBarChartWidget("Dataset Composition", "Percent")
    window.dataset_sequence_chart = DatasetBarChartWidget("Token Distribution", "Tokens")

    # Dataset Health & Advisory Card
    advisor_layout = QVBoxLayout()
    advisor_layout.setContentsMargins(0, 2, 0, 0)
    window.dataset_advisor = QTextEdit()
    window.dataset_advisor.setObjectName("DatasetAdvisorBox")
    window.dataset_advisor.setReadOnly(True)
    window.dataset_advisor.setMinimumHeight(100)
    window.dataset_advisor.setPlainText("Dataset health check telemetry scores and scores actionable recommendations.")
    window._tip(window.dataset_advisor, "Actionable dataset cleanup advice from preview quality checks.")
    advisor_layout.addWidget(window.dataset_advisor, 1)
    advisor_card = _ingestion_card("DATASET HEALTH & ADVISORY", advisor_layout)
    advisor_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    right_column.addWidget(advisor_card, 1)

    # Bottom Command Action Bar Wrapped in Matching Card Frame
    action_card = QWidget()
    action_card.setObjectName("IngestionActionBar")
    action_h = QHBoxLayout(action_card)
    action_h.setContentsMargins(10, 8, 10, 8)
    action_h.setSpacing(10)

    window.health_check_button = QPushButton("[ Check Health ]")
    window.health_check_button.setObjectName("IngestionActionBtn")
    window._tip(window.health_check_button, "Validate source, dataset, model, export, GGUF, and hardware readiness before long work.")
    window.health_check_button.clicked.connect(window.check_project_health)
    window.health_check_button.setMinimumHeight(38)

    window.preview_dataset_button = QPushButton("[ Preview Dataset ]")
    window.preview_dataset_button.setObjectName("IngestionActionBtn")
    window._tip(window.preview_dataset_button, "Scan source files and show dataset quality plus sample text/code snippets without preparing tokens.")
    window.preview_dataset_button.clicked.connect(window.preview_dataset)
    window.preview_dataset_button.setMinimumHeight(38)

    window.prepare_button = QPushButton("[ ⚡ Prepare Dataset ]")
    window.prepare_button.setObjectName("PrepareActionBtn")
    window.prepare_button.setFont(QFont("Segoe UI Symbol", 10, QFont.Bold))
    window._tip(window.prepare_button, "Read source files, clean text, train tokenizer, split tokens, and save the dataset project.")
    window.prepare_button.clicked.connect(window.prepare_dataset)
    window.prepare_button.setMinimumHeight(38)

    window.stop_dataset_button = QPushButton("[ Stop ]")
    window.stop_dataset_button.setObjectName("StopActionBtn")
    window.stop_dataset_button.setEnabled(False)
    window.stop_dataset_button.setMinimumHeight(38)
    window.stop_dataset_button.clicked.connect(window.stop_active_task)
    window._tip(window.stop_dataset_button, "Request a graceful stop for dataset preparation.")

    action_h.addWidget(window.health_check_button, 1)
    action_h.addWidget(window.preview_dataset_button, 1)
    action_h.addWidget(window.prepare_button, 2)
    action_h.addWidget(window.stop_dataset_button, 1)
    right_column.addWidget(action_card, 0)

    ingestion_body.addLayout(left_column, 1)
    ingestion_body.addLayout(right_column, 1)
    layout.addLayout(ingestion_body, 1)

    window.dataset_progress = window._thin_progress()
    outer.addWidget(window.dataset_progress)

    if hasattr(window, "_update_online_dataset_stage_controls"):
        window._update_online_dataset_stage_controls()

    return page
