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
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from llm_trainer.conversation_datasets import CONVERSATION_DATASET_PRESETS



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

    return key.replace("_", " ").title()


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
    # The first directory below the configured root is the category.  Do not
    # interpret names or extensions: adding a folder is the complete
    # configuration needed to add a new category.
    folders = relative.parts[:-1]
    if folders:
        return _slugify_category(folders[0])
    return _slugify_category(path.stem)


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
    folders = relative.parts[:-1]
    return _slugify_category(folders[0]) if folders else "base"


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
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_DEFAULT_SUFFIXES
            and path.stat().st_size > 0
        )
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
            "characters": int(round(len(text) * multiplier)),
            "tokens": int(round(len(pieces) * multiplier)),
            "vocab": int(round(len(vocab) * min(multiplier, 3.0))),
            "sampled": size > len(raw),
        }
    return {"bytes": size, "characters": 0, "tokens": 0, "vocab": 0, "sampled": False}


def format_estimate(value: int, sampled: bool) -> str:
    """Format a numeric estimate for the tree widget."""

    prefix = "~" if sampled else ""
    return f"{prefix}{value:,}"


def dataset_plan_defaults(default_files: list[tuple[Path, str]] | None = None) -> dict[str, float]:
    """Return default blueprint weights plus discovered default-data categories.

    Args:
        default_files: Optional pre-discovered bundled file/category pairs.

    Returns:
        Default category weight mapping.
    """

    if default_files is None:
        return {}
    categories: list[str] = []
    seen: set[str] = set()
    for _path, category in default_files:
        if category not in seen:
            categories.append(category)
            seen.add(category)
    if not categories:
        return {}
    return {category: 0.0 for category in categories}


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
    layout.setAlignment(Qt.AlignTop)

    title_row = QHBoxLayout()
    title = QLabel("Dataset Sources")
    title.setObjectName("PageTitle")
    active_data_root = blueprint_data_root(window)
    default_files = iter_default_data_files(active_data_root)
    window.blueprint_data_root = active_data_root

    window.dataset_plan_source_label = QLabel(f"Source: {active_data_root}")
    window.dataset_plan_source_label.setObjectName("Muted")
    window.dataset_plan_refresh_button = QPushButton("Refresh")
    window.dataset_plan_refresh_button.setMaximumWidth(110)
    title_row.addWidget(title)
    title_row.addSpacing(12)
    title_row.addWidget(window.dataset_plan_source_label, 1)
    title_row.addStretch(1)
    layout.addLayout(title_row)

    external_form = QFormLayout()
    window._configure_form(external_form)
    window.external_dataset_dir = QLineEdit(str(Path.home() / "drunkenbot_datasets" / "default"))
    window.external_dataset_version = QLabel("Installed version: not installed")
    window.external_dataset_version.setObjectName("Muted")
    window._tip(window.external_dataset_dir, "Folder where downloaded dataset categories are installed and used as the ingestion source.")
    window._tip(window.external_dataset_version, "Version recorded from the installed dataset manifest.")
    window.external_dataset_download_button = QPushButton("Download latest dataset")
    window._tip(window.external_dataset_download_button, "Download, verify, and extract the latest dataset release into the install folder.")
    window.external_dataset_download_button.clicked.connect(window.download_latest_external_dataset)
    external_form.addRow("Install folder", window._path_row(window.external_dataset_dir, directory=True))
    external_form.addRow("Status", window.external_dataset_version)
    external_form.addRow("", window.external_dataset_download_button)
    external_card = window._card("EXTERNAL DATASET", external_form)
    body_grid = QGridLayout()
    body_grid.setHorizontalSpacing(14)
    body_grid.setVerticalSpacing(12)

    conversation_form = QFormLayout()
    window._configure_form(conversation_form)
    window.dataset_stage = QComboBox()
    # Workflows are directories in the configured training-data root.  This
    # keeps the UI in sync with the corpus instead of requiring code changes
    # for every new training workflow.
    workflow_names = sorted(
        (path.name for path in active_data_root.iterdir() if path.is_dir()),
        key=str.casefold,
    ) if active_data_root.exists() else []
    window.dataset_stage.addItems(workflow_names or ["base"])
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
    window.custom_huggingface_dataset = QLineEdit()
    window.custom_huggingface_dataset.setPlaceholderText("owner/dataset or Hugging Face URL")
    window._tip(window.custom_huggingface_dataset, "Optional Hugging Face dataset repository to load in addition to the selected presets.")
    conversation_form.addRow("Custom HF dataset", window.custom_huggingface_dataset)
    window.custom_huggingface_download = QPushButton("Download custom dataset")
    window.custom_huggingface_download.clicked.connect(window._download_custom_huggingface_dataset)
    window._tip(window.custom_huggingface_download, "Enable the custom dataset and download it during the next dataset preparation run.")
    conversation_form.addRow("", window.custom_huggingface_download)
    window.conversation_sample_limit = window._spin(0, 2_000_000, 20000)
    window.conversation_sample_limit.setMaximumHeight(30)
    window.conversation_sample_limit.setEnabled(False)
    window.include_conversation_datasets.toggled.connect(window.conversation_sample_limit.setEnabled)
    window.include_conversation_datasets.toggled.connect(window._update_online_dataset_stage_controls)
    window.dataset_stage.currentTextChanged.connect(window._update_online_dataset_stage_controls)
    conversation_form.addRow("Rows / set", window.conversation_sample_limit)
    conversation_card = window._card("OPTIONAL EXTERNAL / STRUCTURED DATA", conversation_form)
    conversation_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    conversation_card.setMaximumHeight(250)

    window.default_data_tree_updating = False
    window.default_data_tree = QTreeWidget()
    window.default_data_tree.setHeaderLabels(["Category / file", "Characters", "Vocab"])
    window.default_data_tree.setRootIsDecorated(True)
    window.default_data_tree.setAlternatingRowColors(False)
    window.default_data_tree.setMinimumHeight(600)
    window.default_data_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    window.default_data_tree.setColumnWidth(0, 360)
    window.default_data_tree.setColumnWidth(1, 130)
    window.default_data_tree.setColumnWidth(2, 130)
    window.default_data_actions = {}
    window.default_data_category_items = {}
    window.default_data_tree.clear()
    grouped_files: dict[str, list[Path]] = {}
    for path, category in default_files:
        grouped_files.setdefault(category, []).append(path)
    for category in sorted(grouped_files, key=dataset_category_label):
        total_characters = 0
        total_vocab = 0
        category_sampled = False
        category_item = QTreeWidgetItem([dataset_category_label(category), "0", "0"])
        category_item.setData(0, Qt.UserRole, {"kind": "category", "category": category})
        category_item.setFlags(category_item.flags() | Qt.ItemIsUserCheckable)
        category_item.setCheckState(0, Qt.Checked)
        window.default_data_tree.addTopLevelItem(category_item)
        window.default_data_category_items[category] = category_item
        for path in sorted(grouped_files[category], key=lambda item: item.name.lower()):
            try:
                stats = file_token_vocab_stats(path)
            except OSError:
                stats = {"characters": 0, "vocab": 0, "sampled": False}
            sampled = bool(stats.get("sampled", False))
            characters = int(stats.get("characters", 0))
            vocab = int(stats.get("vocab", 0))
            child = QTreeWidgetItem([path.name, format_estimate(characters, sampled), format_estimate(vocab, sampled)])
            child.setToolTip(0, str(path))
            child.setData(0, Qt.UserRole, {"kind": "file", "path": str(path), "category": category})
            child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
            child.setCheckState(0, Qt.Checked)
            category_item.addChild(child)
            window.default_data_actions[str(path)] = child
            total_characters += characters
            total_vocab += vocab
            category_sampled = category_sampled or sampled
        category_item.setText(1, format_estimate(total_characters, category_sampled))
        category_item.setText(2, format_estimate(total_vocab, category_sampled))
        category_item.setExpanded(False)
    if not window.default_data_actions:
        window.default_data_tree.addTopLevelItem(QTreeWidgetItem(["No project/default data files were found.", "", ""]))
    window.default_data_tree.itemChanged.connect(window._handle_default_data_tree_changed)
    default_layout = QVBoxLayout()
    tree_title_row = QHBoxLayout()
    tree_title_row.addWidget(QLabel("Downloaded and local dataset files"))
    tree_title_row.addStretch(1)
    tree_title_row.addWidget(window.dataset_plan_refresh_button)
    default_layout.addLayout(tree_title_row)
    default_layout.addWidget(window.default_data_tree)
    default_card = window._card("BUNDLED DEFAULT DATA", default_layout)
    body_grid.addWidget(external_card, 0, 0)
    body_grid.addWidget(conversation_card, 1, 0)
    body_grid.addWidget(default_card, 0, 1, 2, 1)
    body_grid.setColumnStretch(0, 1)
    body_grid.setColumnStretch(1, 1)
    layout.addLayout(body_grid)
    window.dataset_plan_refresh_button.clicked.connect(window.refresh_dataset_blueprint_files)
    window._tip(window.dataset_plan_refresh_button, "Reload this tree to include newly copied files and folders.")
    window._update_online_dataset_stage_controls()

    scroll.setWidget(content)
    outer.addWidget(scroll, 1)
    window.dataset_plan_progress = window._thin_progress()
    window.dataset_plan_progress.setVisible(False)
    outer.addWidget(window.dataset_plan_progress)
    return page


def populate_default_data_tree(window: Any, root: Path) -> None:
    """Reload the existing dataset tree without rebuilding the containing tab."""
    tree = window.default_data_tree
    tree.blockSignals(True)
    try:
        tree.clear()
        window.default_data_actions = {}
        window.default_data_category_items = {}
        grouped_files: dict[str, list[Path]] = {}
        for path, category in iter_default_data_files(root):
            grouped_files.setdefault(category, []).append(path)
        for category in sorted(grouped_files, key=dataset_category_label):
            category_item = QTreeWidgetItem([dataset_category_label(category), "0", "0"])
            category_item.setData(0, Qt.UserRole, {"kind": "category", "category": category})
            category_item.setFlags(category_item.flags() | Qt.ItemIsUserCheckable)
            category_item.setCheckState(0, Qt.Checked)
            tree.addTopLevelItem(category_item)
            window.default_data_category_items[category] = category_item
            total_characters = total_vocab = 0
            sampled_category = False
            for path in sorted(grouped_files[category], key=lambda item: item.name.lower()):
                try:
                    stats = file_token_vocab_stats(path)
                except OSError:
                    stats = {"characters": 0, "vocab": 0, "sampled": False}
                sampled = bool(stats.get("sampled", False))
                characters = int(stats.get("characters", 0))
                vocab = int(stats.get("vocab", 0))
                child = QTreeWidgetItem(
                    [path.name, format_estimate(characters, sampled), format_estimate(vocab, sampled)]
                )
                child.setToolTip(0, str(path))
                child.setData(0, Qt.UserRole, {"kind": "file", "path": str(path), "category": category})
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(0, Qt.Checked)
                category_item.addChild(child)
                window.default_data_actions[str(path)] = child
                total_characters += characters
                total_vocab += vocab
                sampled_category = sampled_category or sampled
            category_item.setText(1, format_estimate(total_characters, sampled_category))
            category_item.setText(2, format_estimate(total_vocab, sampled_category))
    finally:
        tree.blockSignals(False)
    if not window.default_data_actions:
        tree.addTopLevelItem(QTreeWidgetItem(["No project/default data files were found.", "", ""]))
