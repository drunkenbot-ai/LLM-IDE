from __future__ import annotations

from pathlib import Path
import re
from typing import Any

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
    QTreeWidget,
    QTreeWidgetItem,
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

# Keep generated_curriculum as a legacy wrapper so older project-local copies
# made before the default_data flattening still classify correctly.
GENERIC_DEFAULT_DATA_FOLDERS = {"base_training", "code_training", "generated_curriculum"}

CATEGORY_ALIASES: dict[str, str] = {
    "story": "stories",
    "stories": "stories",
    "reason": "reasoning",
    "reasoning": "reasoning",
    "why": "reasoning",
    "emotion": "social_emotional",
    "emotions": "social_emotional",
    "social": "social_emotional",
    "conversation": "social_emotional",
    "dialog": "social_emotional",
    "dialogue": "social_emotional",
    "geography": "factual_knowledge",
    "science": "factual_knowledge",
    "biology": "factual_knowledge",
    "physics": "factual_knowledge",
    "chemistry": "factual_knowledge",
    "astronomy": "factual_knowledge",
    "weather": "factual_knowledge",
    "earth": "factual_knowledge",
    "history": "factual_knowledge",
    "facts": "factual_knowledge",
    "knowledge": "factual_knowledge",
    "math": "mathematics",
    "mathematics": "mathematics",
    "code": "code_technical",
    "coding": "code_technical",
    "computer": "code_technical",
    "computers": "code_technical",
    "cs": "code_technical",
    "programming": "code_technical",
    "technical": "code_technical",
    "language": "language_basics",
    "grammar": "language_basics",
    "qa": "structured_qa",
    "question": "structured_qa",
    "answers": "structured_qa",
    "instruction": "structured_qa",
    "instructions": "structured_qa",
    "fine": "structured_qa",
    "safety": "safety_uncertainty",
    "ethics": "safety_uncertainty",
    "honesty": "safety_uncertainty",
    "fairness": "safety_uncertainty",
    "uncertainty": "safety_uncertainty",
    "everyday": "general_prose",
    "health": "general_prose",
    "finance": "general_prose",
    "jobs": "general_prose",
    "prose": "general_prose",
}

DEFAULT_DATA_STAGE_FOLDERS: dict[str, str] = {
    "fine_tune_conversation": "conversation",
    "fine_tune_instruction": "instruction",
    "fine_tune_code": "code",
}

DEFAULT_DATA_STAGE_CATEGORIES: dict[str, str] = {
    "fine_tune_conversation": "conversation",
    "fine_tune_instruction": "instruction",
    "fine_tune_code": "code_technical",
}

CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".sh", ".ps1"}
SUPPORTED_DEFAULT_SUFFIXES = {".txt", ".md", ".text", ".jsonl", ".json", *CODE_SUFFIXES}


def default_data_root() -> Path:
    """Return the bundled default data folder.

    Returns:
        Absolute path to the packaged ``default_data`` folder.
    """

    return Path(__file__).resolve().parents[2] / "default_data"


def blueprint_data_root(window: Any | None = None) -> Path:
    """Return the active Dataset Blueprint data root.

    Args:
        window: Optional main window carrying a project-local data root.

    Returns:
        Project-local training data root when available, otherwise bundled data.
    """

    root = getattr(window, "blueprint_data_root", None)
    if root:
        path = Path(root)
        if path.exists():
            return path
    return default_data_root()


def _slugify_category(value: str) -> str:
    """Convert folder/file text into a stable category key.

    Args:
        value: Folder name, file stem, or user-facing text.

    Returns:
        Lowercase underscore category key.
    """

    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "general_prose"


def dataset_category_label(key: str) -> str:
    """Return a readable label for a category key.

    Args:
        key: Dataset category key.

    Returns:
        User-facing label.
    """

    return DATASET_DOMAIN_LABELS.get(key, key.replace("_", " ").title())


def _category_from_text(value: str) -> str | None:
    """Infer a known category from free text.

    Args:
        value: Folder name or file stem.

    Returns:
        Canonical category key when known.
    """

    tokens = [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]
    for token in tokens:
        if token in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[token]
    return None


def default_data_category(path: Path, root: Path | None = None) -> str:
    """Infer the Dataset Blueprint category for a bundled file.

    Args:
        path: Bundled source file.

    Returns:
        Dataset category key used by the sampler.
    """

    root = root or default_data_root()
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    relative_parts_lower = [part.lower() for part in relative.parts]
    for folder, category in DEFAULT_DATA_STAGE_CATEGORIES.items():
        if folder in relative_parts_lower[:-1]:
            return category
    if path.suffix.lower() in CODE_SUFFIXES:
        return "code_technical"
    for parent in reversed(relative.parts[:-1]):
        category = _category_from_text(parent)
        if category:
            return category
    stem_category = _category_from_text(path.stem)
    if stem_category:
        return stem_category
    for parent in reversed(relative.parts[:-1]):
        slug = _slugify_category(parent)
        if slug and slug not in GENERIC_DEFAULT_DATA_FOLDERS:
            return slug
    return "general_prose"


def default_data_stage(path: Path, root: Path | None = None) -> str:
    """Infer which training stage should use a bundled file.

    Args:
        path: Bundled source file.

    Returns:
        Stage key: base, instruction, conversation, or code.
    """

    root = root or default_data_root()
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parts = {part.lower() for part in relative.parts[:-1]}
    for folder, stage in DEFAULT_DATA_STAGE_FOLDERS.items():
        if folder in parts:
            return stage
    return "base"


def iter_default_data_files(root: Path | None = None) -> list[tuple[Path, str]]:
    """List default/project data files with categories.

    Args:
        root: Optional source root. Defaults to bundled default data.

    Returns:
        Pairs of file path and Dataset Blueprint category.
    """

    root = root or default_data_root()
    if not root.exists():
        return []
    return [
        (path, default_data_category(path, root))
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_DEFAULT_SUFFIXES and path.stat().st_size > 0
    ]


def file_token_vocab_stats(path: Path, sample_bytes: int = 256 * 1024) -> dict[str, int | bool]:
    """Estimate token and vocabulary counts for a data file.

    Args:
        path: Source file path.
        sample_bytes: Maximum bytes to read for a fast estimate.

    Returns:
        Dictionary containing size, estimated tokens, estimated vocab, and
        whether values were extrapolated from a sample.
    """

    size = path.stat().st_size
    if path.suffix.lower() in {".json", ".jsonl", ".txt", ".md", ".text", *CODE_SUFFIXES}:
        with path.open("rb") as handle:
            raw = handle.read(sample_bytes)
        text = raw.decode("utf-8", errors="ignore")
        pieces = re.findall(r"\w+|[^\w\s]", text)
        vocab = {piece.lower() for piece in pieces if piece.strip()}
        multiplier = size / max(len(raw), 1) if raw and size > len(raw) else 1.0
        return {
            "bytes": size,
            "tokens": int(round(len(pieces) * multiplier)),
            "vocab": int(round(len(vocab) * min(multiplier, 3.0))),
            "sampled": size > len(raw),
        }
    return {"bytes": size, "tokens": 0, "vocab": 0, "sampled": False}


def file_stats_text(path: Path) -> str:
    """Return compact size/token/vocab text for the tree viewer."""

    try:
        stats = file_token_vocab_stats(path)
    except OSError:
        return ""
    size = float(stats["bytes"])
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            size_text = f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
            break
        size /= 1024
    prefix = "~" if stats.get("sampled") else ""
    return f"{size_text} | {prefix}{int(stats['tokens']):,} tok | {prefix}{int(stats['vocab']):,} vocab"


def dataset_plan_defaults(default_files: list[tuple[Path, str]] | None = None) -> dict[str, float]:
    """Return default blueprint weights plus discovered default-data categories.

    Args:
        default_files: Optional pre-discovered bundled file/category pairs.

    Returns:
        Default category weight mapping.
    """

    if default_files is None:
        return dict(DATASET_DOMAIN_DEFAULTS)
    categories: list[str] = []
    seen: set[str] = set()
    for _path, category in default_files:
        if category not in seen:
            categories.append(category)
            seen.add(category)
    if not categories:
        return dict(DATASET_DOMAIN_DEFAULTS)
    return {category: DATASET_DOMAIN_DEFAULTS.get(category, 0.0) for category in categories}


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
    active_data_root = blueprint_data_root(window)
    default_files = iter_default_data_files(active_data_root)
    plan_defaults = dataset_plan_defaults(default_files)
    window.blueprint_data_root = active_data_root

    window.dataset_plan_total_label = QLabel("Total: 100.0%")
    window.dataset_plan_total_label.setObjectName("Metric")
    window.dataset_plan_source_label = QLabel(f"Source: {active_data_root}")
    window.dataset_plan_source_label.setObjectName("Muted")
    title_row.addWidget(title)
    title_row.addSpacing(12)
    title_row.addWidget(window.dataset_plan_total_label)
    title_row.addWidget(window.dataset_plan_source_label, 1)
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
    window.dataset_plan_domain_grid = domain_grid
    window.dataset_plan_spins = {}
    for index, (key, value) in enumerate(plan_defaults.items()):
        spin = window._double_spin(0.0, 100.0, value, 1.0, 1)
        spin.setMinimumWidth(92)
        spin.setMaximumHeight(30)
        spin.valueChanged.connect(window._dataset_plan_mark_custom)
        spin.valueChanged.connect(window._update_dataset_plan_total)
        window.dataset_plan_spins[key] = spin

        cell = QWidget()
        cell_layout = QHBoxLayout(cell)
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(8)
        cell_label = QLabel(dataset_category_label(key))
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

    window.default_data_tree_updating = False
    window.default_data_tree = QTreeWidget()
    window.default_data_tree.setHeaderLabels(["Category / file", "Stats"])
    window.default_data_tree.setRootIsDecorated(True)
    window.default_data_tree.setAlternatingRowColors(False)
    window.default_data_tree.setMinimumHeight(260)
    window.default_data_tree.setColumnWidth(0, 420)
    window.default_data_actions = {}
    window.default_data_category_items = {}
    grouped_files: dict[str, list[Path]] = {}
    for path, category in default_files:
        grouped_files.setdefault(category, []).append(path)
    for category in sorted(grouped_files, key=dataset_category_label):
        category_item = QTreeWidgetItem([dataset_category_label(category), f"{len(grouped_files[category])} files"])
        category_item.setData(0, Qt.UserRole, {"kind": "category", "category": category})
        category_item.setFlags(category_item.flags() | Qt.ItemIsUserCheckable)
        category_item.setCheckState(0, Qt.Checked)
        window.default_data_tree.addTopLevelItem(category_item)
        window.default_data_category_items[category] = category_item
        for path in sorted(grouped_files[category], key=lambda item: item.name.lower()):
            child = QTreeWidgetItem([path.name, file_stats_text(path)])
            child.setToolTip(0, str(path))
            child.setData(0, Qt.UserRole, {"kind": "file", "path": str(path), "category": category})
            child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
            child.setCheckState(0, Qt.Checked)
            category_item.addChild(child)
            window.default_data_actions[str(path)] = child
        category_item.setExpanded(True)
    if not window.default_data_actions:
        window.default_data_tree.addTopLevelItem(QTreeWidgetItem(["No project/default data files were found.", ""]))
    window.default_data_tree.itemChanged.connect(window._handle_default_data_tree_changed)
    default_layout = QVBoxLayout()
    default_layout.addWidget(window.default_data_tree)
    default_card = window._card("BUNDLED DEFAULT DATA", default_layout)
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
