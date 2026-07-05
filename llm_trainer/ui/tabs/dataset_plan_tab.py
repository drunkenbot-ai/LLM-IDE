from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from llm_trainer.conversation_datasets import CONVERSATION_DATASET_PRESETS


DATASET_DOMAIN_DEFAULTS: dict[str, float] = {
    "stories": 22.0,
    "reasoning": 18.0,
    "social_emotional": 12.0,
    "factual_knowledge": 13.0,
    "mathematics": 8.0,
    "code_technical": 10.0,
    "language_basics": 7.0,
    "structured_qa": 5.0,
    "safety_uncertainty": 3.0,
    "general_prose": 2.0,
}

DATASET_DOMAIN_PRESETS: dict[str, dict[str, float]] = {
    "Balanced Tiny LLM": DATASET_DOMAIN_DEFAULTS,
    "Code Assistant": {
        "stories": 8.0,
        "reasoning": 18.0,
        "social_emotional": 5.0,
        "factual_knowledge": 8.0,
        "mathematics": 12.0,
        "code_technical": 32.0,
        "language_basics": 4.0,
        "structured_qa": 8.0,
        "safety_uncertainty": 3.0,
        "general_prose": 2.0,
    },
    "Chat Assistant": {
        "stories": 15.0,
        "reasoning": 12.0,
        "social_emotional": 22.0,
        "factual_knowledge": 10.0,
        "mathematics": 5.0,
        "code_technical": 6.0,
        "language_basics": 8.0,
        "structured_qa": 8.0,
        "safety_uncertainty": 6.0,
        "general_prose": 8.0,
    },
    "Reasoning Tutor": {
        "stories": 10.0,
        "reasoning": 28.0,
        "social_emotional": 6.0,
        "factual_knowledge": 12.0,
        "mathematics": 18.0,
        "code_technical": 8.0,
        "language_basics": 4.0,
        "structured_qa": 8.0,
        "safety_uncertainty": 3.0,
        "general_prose": 3.0,
    },
    "Storyteller": {
        "stories": 40.0,
        "reasoning": 8.0,
        "social_emotional": 18.0,
        "factual_knowledge": 8.0,
        "mathematics": 3.0,
        "code_technical": 2.0,
        "language_basics": 8.0,
        "structured_qa": 3.0,
        "safety_uncertainty": 2.0,
        "general_prose": 8.0,
    },
}

DATASET_DOMAIN_LABELS: dict[str, str] = {
    "stories": "Stories",
    "reasoning": "Reasoning",
    "social_emotional": "Social / emotions",
    "factual_knowledge": "Facts / knowledge",
    "mathematics": "Mathematics",
    "code_technical": "Code / technical",
    "language_basics": "Language basics",
    "structured_qa": "Structured Q&A",
    "safety_uncertainty": "Safety / uncertainty",
    "general_prose": "General prose",
}


def default_data_root() -> Path:
    """Return the bundled default data folder.

    Returns:
        Absolute path to the packaged ``default_data`` folder.
    """

    return Path(__file__).resolve().parents[2] / "default_data"


def default_data_category(path: Path) -> str:
    """Infer the Dataset Blueprint category for a bundled file.

    Args:
        path: Bundled source file.

    Returns:
        Dataset category key used by the sampler.
    """

    name = path.stem.lower()
    if path.suffix.lower() in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".sh", ".ps1"}:
        return "code_technical"
    if "story" in name:
        return "stories"
    if "reasoning" in name or name.startswith("why"):
        return "reasoning"
    if "emotion" in name or "conversation" in name:
        return "social_emotional"
    if "geography" in name or "science" in name or "history" in name:
        return "factual_knowledge"
    if "math" in name:
        return "mathematics"
    return "general_prose"


def iter_default_data_files() -> list[tuple[Path, str]]:
    """List bundled default data files with categories.

    Returns:
        Pairs of file path and Dataset Blueprint category.
    """

    root = default_data_root()
    if not root.exists():
        return []
    supported = {".txt", ".md", ".text", ".jsonl", ".json", ".py", ".js", ".ts", ".sh", ".ps1"}
    return [
        (path, default_data_category(path))
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in supported and path.stat().st_size > 0
    ]


def build_dataset_plan_tab(window) -> QWidget:
    """Build the dataset blueprint page.

    Args:
        window: Main application window that owns shared helper methods.

    Returns:
        Dataset blueprint page widget.
    """

    page = QWidget()
    outer = QVBoxLayout(page)
    outer.setContentsMargins(18, 18, 18, 12)
    outer.setSpacing(12)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)

    title_row = QHBoxLayout()
    title = QLabel("Dataset Blueprint")
    title.setObjectName("PageTitle")
    window.dataset_plan_total_label = QLabel("Total: 100.0%")
    window.dataset_plan_total_label.setObjectName("Metric")
    title_row.addWidget(title)
    title_row.addSpacing(12)
    title_row.addWidget(window.dataset_plan_total_label)
    title_row.addStretch(1)
    layout.addLayout(title_row)

    preset_row = QHBoxLayout()
    preset_row.setSpacing(10)
    preset_label = QLabel("Recipe preset")
    window.dataset_plan_preset = QComboBox()
    window.dataset_plan_preset.addItems([*DATASET_DOMAIN_PRESETS.keys(), "Custom"])
    window.dataset_plan_preset.setMinimumWidth(260)
    window.dataset_plan_normalize_button = QPushButton("Normalize")
    window.dataset_plan_apply_button = QPushButton("Apply To Ingestion")
    preset_row.addWidget(preset_label)
    preset_row.addWidget(window.dataset_plan_preset)
    preset_row.addWidget(window.dataset_plan_normalize_button)
    preset_row.addWidget(window.dataset_plan_apply_button)
    preset_row.addStretch(1)
    layout.addLayout(preset_row)

    body_grid = QGridLayout()
    body_grid.setHorizontalSpacing(14)
    body_grid.setVerticalSpacing(12)

    domain_grid = QGridLayout()
    domain_grid.setHorizontalSpacing(14)
    domain_grid.setVerticalSpacing(6)
    window.dataset_plan_spins = {}
    for index, (key, label) in enumerate(DATASET_DOMAIN_LABELS.items()):
        spin = window._double_spin(0.0, 100.0, DATASET_DOMAIN_DEFAULTS[key], 1.0, 1)
        spin.setMinimumWidth(92)
        spin.setMaximumHeight(30)
        spin.valueChanged.connect(window._dataset_plan_mark_custom)
        spin.valueChanged.connect(window._update_dataset_plan_total)
        window.dataset_plan_spins[key] = spin

        cell = QWidget()
        cell_layout = QHBoxLayout(cell)
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(8)
        cell_label = QLabel(label)
        cell_label.setMinimumWidth(112)
        cell_layout.addWidget(cell_label)
        cell_layout.addWidget(spin, 1)
        row = index // 3
        column = index % 3
        domain_grid.addWidget(cell, row, column)

    domain_card = window._card("TARGET DATA RECIPE", domain_grid)
    domain_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    body_grid.addWidget(domain_card, 0, 0, 1, 2)

    conversation_form = QFormLayout()
    window._configure_form(conversation_form)
    window.dataset_stage = QComboBox()
    window.dataset_stage.addItems(["Base pretraining", "Instruction fine-tune", "Conversation fine-tune", "Code fine-tune"])
    window.dataset_stage.setMaximumWidth(240)
    window.include_conversation_datasets = QCheckBox("Online")
    window.include_conversation_datasets.setChecked(False)
    purpose_row = QWidget()
    purpose_layout = QHBoxLayout(purpose_row)
    purpose_layout.setContentsMargins(0, 0, 0, 0)
    purpose_layout.setSpacing(8)
    purpose_layout.addWidget(window.dataset_stage, 1)
    purpose_layout.addWidget(window.include_conversation_datasets)
    conversation_form.addRow("Purpose", purpose_row)
    window.conversation_datasets_status = QLabel("Base pretraining: choose optional online corpus datasets, or use local files only.")
    window.conversation_datasets_status.setObjectName("Muted")
    conversation_form.addRow("", window.conversation_datasets_status)
    window.conversation_dataset_button = QPushButton("Online datasets off")
    window.conversation_dataset_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    window.conversation_dataset_menu = QMenu(window.conversation_dataset_button)
    window.conversation_dataset_button.setMenu(window.conversation_dataset_menu)
    window.conversation_dataset_actions = {}
    window.conversation_dataset_widget_actions = {}
    for dataset_id, preset in CONVERSATION_DATASET_PRESETS.items():
        checkbox = QCheckBox(preset.label)
        checkbox.setEnabled(False)
        checkbox.setToolTip(preset.description)
        checkbox.toggled.connect(lambda _checked=False: window._update_conversation_dataset_button_text())
        widget_action = QWidgetAction(window.conversation_dataset_menu)
        widget_action.setDefaultWidget(checkbox)
        window.conversation_dataset_menu.addAction(widget_action)
        window.conversation_dataset_actions[dataset_id] = checkbox
        window.conversation_dataset_widget_actions[dataset_id] = widget_action
    conversation_form.addRow("Online sets", window.conversation_dataset_button)
    window.local_conversation_dataset = QLineEdit()
    window.local_instruction_dataset = QLineEdit()
    conversation_form.addRow(
        "Conversation JSON",
        window._multi_file_path_row(window.local_conversation_dataset, file_filter="JSON datasets (*.json *.jsonl);;All files (*)"),
    )
    conversation_form.addRow(
        "Instruction JSON",
        window._multi_file_path_row(window.local_instruction_dataset, file_filter="JSON datasets (*.json *.jsonl);;All files (*)"),
    )
    window.conversation_sample_limit = window._spin(0, 2_000_000, 20000)
    window.conversation_sample_limit.setMaximumHeight(30)
    window.conversation_sample_limit.setEnabled(False)
    window.include_conversation_datasets.toggled.connect(window.conversation_sample_limit.setEnabled)
    window.include_conversation_datasets.toggled.connect(window._update_online_dataset_stage_controls)
    window.dataset_stage.currentTextChanged.connect(window._update_online_dataset_stage_controls)
    conversation_form.addRow("Rows / set", window.conversation_sample_limit)
    conversation_card = window._card("ONLINE / STRUCTURED DATA", conversation_form)
    body_grid.addWidget(conversation_card, 1, 0)

    default_grid = QGridLayout()
    default_grid.setHorizontalSpacing(8)
    default_grid.setVerticalSpacing(4)
    window.default_data_actions = {}
    for index, (path, category) in enumerate(iter_default_data_files()):
        label = f"{path.name}  |  {DATASET_DOMAIN_LABELS.get(category, category)}"
        checkbox = QCheckBox(label)
        checkbox.setChecked(True)
        checkbox.setToolTip(str(path))
        checkbox.setMaximumHeight(24)
        window.default_data_actions[str(path)] = checkbox
        default_grid.addWidget(checkbox, index, 0)
    if not window.default_data_actions:
        default_grid.addWidget(QLabel("No bundled default data files were found."), 0, 0)
    default_card = window._card("BUNDLED DEFAULT DATA", default_grid)
    body_grid.addWidget(default_card, 1, 1)
    body_grid.setColumnStretch(0, 1)
    body_grid.setColumnStretch(1, 1)
    layout.addLayout(body_grid)

    guide = QTextEdit()
    guide.setReadOnly(True)
    guide.setMinimumHeight(230)
    guide.setPlainText(
        "Use this panel before ingestion to decide the desired training mix.\n\n"
        "Recommended starting blend for a small general model:\n"
        "- 20-25% stories for language flow and simple world modeling.\n"
        "- 15-20% reasoning for step-by-step tasks and explanations.\n"
        "- 10-15% social/emotional data for natural conversation tone.\n"
        "- 10-15% factual knowledge for geography, science, history, and general facts.\n"
        "- 5-10% mathematics for numeracy and symbolic patterns.\n"
        "- Add code/technical data when you want coding ability.\n\n"
        "Apply To Ingestion maps this richer recipe into the current ingestion families: "
        "local prose, online base, instruction, conversation, and source code. Exact "
        "domain enforcement will become stronger when files and online datasets are tagged "
        "by topic in the next dataset-manager pass."
    )
    window._tip(
        guide,
        "Explains how the high-level domain recipe is converted into ingestion mixture weights.",
    )
    guide_layout = QVBoxLayout()
    guide_layout.addWidget(guide)
    guide_card = window._card("BLUEPRINT NOTES", guide_layout)
    layout.addWidget(guide_card, 1)

    window.dataset_plan_preset.currentTextChanged.connect(window.apply_dataset_plan_preset)
    window.dataset_plan_normalize_button.clicked.connect(window.normalize_dataset_plan)
    window.dataset_plan_apply_button.clicked.connect(window.apply_dataset_plan_to_ingestion)
    window._tip(window.dataset_plan_preset, "Choose a starting dataset recipe for the model personality and target ability.")
    window._tip(window.dataset_plan_normalize_button, "Scale all blueprint percentages so the total is exactly 100 percent.")
    window._tip(window.dataset_plan_apply_button, "Copy this high-level recipe into the Ingestion tab mixture controls.")
    window._update_online_dataset_stage_controls()
    window._update_dataset_plan_total()

    scroll.setWidget(content)
    outer.addWidget(scroll, 1)
    return page
