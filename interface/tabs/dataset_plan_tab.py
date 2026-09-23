"""Dataset Sources and Blueprint management tab."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from engine.conversation_datasets import CONVERSATION_DATASET_PRESETS
from interface.theme import is_system_theme

CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".sh", ".ps1"}
SUPPORTED_DEFAULT_SUFFIXES = {".txt", ".md", ".text", ".jsonl", ".json", *CODE_SUFFIXES}

CATEGORY_PALETTE = [
    "#3b82f6",  # Systems Code (Vibrant Blue)
    "#10b981",  # STEM / Math (Emerald Green)
    "#f59e0b",  # Algorithms (Electric Amber)
    "#8b5cf6",  # Hardware RTL (Purple / Violet)
    "#ec4899",  # Cybersecurity (Hot Pink)
    "#06b6d4",  # Biomedicine (Cyan)
    "#0ea5e9",  # Physical Sciences (Sky Blue)
    "#14b8a6",  # Finance (Teal)
    "#f97316",  # Legal Reasoning (Orange)
    "#a855f7",  # Multilingual (Deep Violet)
    "#64748b",  # Encyclopedic (Slate Gray)
]


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


def infer_category_type(category_slug: str) -> tuple[str, str]:
    """Return (icon_symbol, type_label) for category slug matching concept art."""
    slug = category_slug.lower()
    if any(k in slug for k in ["base", "clean", "general", "academic"]):
        return ("✦", "Base Pretraining")
    if any(k in slug for k in ["prose", "wiki", "book", "web", "article"]):
        return ("✦", "Pre-training Base")
    if any(k in slug for k in ["reason", "math", "logic", "stem", "cot"]):
        return ("◆", "Formal Reasoning")
    if any(k in slug for k in ["instruct", "chat", "dialog", "eval", "sft"]):
        return ("◆", "Instruction")
    if any(k in slug for k in ["code", "system", "kernel", "algo", "python", "c_"]):
        return ("◈", "Systems Code")
    if any(k in slug for k in ["tool", "agent", "trajectory", "action"]):
        return ("▲", "Tool Calling")
    if any(k in slug for k in ["identity", "fact", "synthetic", "forge"]):
        return ("★", "Synthetic Identity")
    return ("●", "Discipline")


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


def format_estimate(value: int, sampled: bool = False) -> str:
    """Format a numeric estimate for the tree widget."""
    prefix = "~" if sampled else ""
    return f"{prefix}{value:,}"


def format_token_count(tokens: int) -> str:
    """Format token/character count into a human-readable suffix (e.g. 4.2B or 250M)."""
    if tokens >= 1_000_000_000:
        return f"{tokens / 1_000_000_000:.2f}B"
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.2f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def format_size(bytes_count: int) -> str:
    """Format bytes count into human-readable disk units (e.g. 8.4 GB)."""
    if bytes_count >= 1024 * 1024 * 1024:
        return f"{bytes_count / (1024 * 1024 * 1024):.1f} GB"
    if bytes_count >= 1024 * 1024:
        return f"{bytes_count / (1024 * 1024):.1f} MB"
    if bytes_count >= 1024:
        return f"{bytes_count / 1024:.0f} KB"
    return f"{bytes_count} B"


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


class PillToggleSwitch(QWidget):
    """Sleek iOS-style amber pill toggle switch matching concept art."""

    toggled = Signal(bool)

    def __init__(self, checked: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._checked = checked
        self.setFixedSize(36, 18)
        self.setCursor(Qt.PointingHandCursor)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool, block_signal: bool = False) -> None:
        if self._checked != checked:
            self._checked = checked
            self.update()
            if not block_signal:
                self.toggled.emit(self._checked)

    def text(self) -> str:
        return ""

    def setText(self, text: str) -> None:
        pass

    def checkState(self) -> Qt.CheckState:
        return Qt.Checked if self._checked else Qt.Unchecked

    def setCheckState(self, state: Any) -> None:
        self.setChecked(state == Qt.Checked)

    def mousePressEvent(self, event: Any) -> None:
        if not self.isEnabled():
            return
        if event.button() == Qt.LeftButton:
            self._checked = not self._checked
            self.update()
            self.toggled.emit(self._checked)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if not self.isEnabled():
            painter.setOpacity(0.4)
        w = float(self.width())
        h = float(self.height())
        r = h / 2.0

        is_sys = is_system_theme()
        if self._checked:
            bg_color = QColor("#f59e0b")
            thumb_color = QColor("#ffffff")
            thumb_x = w - h + 2.0
        else:
            bg_color = QColor("#cbd5e1" if is_sys else "#242938")
            thumb_color = QColor("#64748b" if is_sys else "#94a3b8")
            thumb_x = 2.0

        # Track
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg_color)
        painter.drawRoundedRect(QRectF(0, 0, w, h), r, r)

        # Thumb
        painter.setBrush(thumb_color)
        painter.drawEllipse(QRectF(thumb_x, 2.0, h - 4.0, h - 4.0))
        painter.end()


class MiniSignalBarWidget(QWidget):
    """5-bar ascending sparkline visualizer matching concept art."""

    def __init__(self, magnitude: int = 4, color_hex: str = "#f59e0b", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.magnitude = max(1, min(5, magnitude))
        self.color_hex = color_hex
        self.setFixedSize(32, 16)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bar_w = 3.5
        spacing = 2.5
        heights = [4.0, 7.0, 10.0, 13.0, 16.0]
        spectrum = ["#f59e0b", "#10b981", "#06b6d4", "#3b82f6", "#a855f7"]

        is_sys = is_system_theme()
        inactive_color = QColor("#e2e8f0" if is_sys else "#222634")

        x = 0.0
        for i in range(5):
            h = heights[i]
            y = 16.0 - h
            active = (i < self.magnitude)
            col = QColor(spectrum[i]) if active else inactive_color
            painter.setPen(Qt.NoPen)
            painter.setBrush(col)
            painter.drawRoundedRect(QRectF(x, y, bar_w, h), 1.0, 1.0)
            x += bar_w + spacing
        painter.end()


class DatasetPreviewDialog(QDialog):
    """Modern modal dialog showing sample records, statistics, and metadata."""

    def __init__(self, title_text: str, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Dataset Preview: {title_text}")
        self.resize(760, 500)
        is_sys = is_system_theme()
        bg = "#ffffff" if is_sys else "#11131c"
        fg = "#0f172a" if is_sys else "#f8fafc"
        border = "#cbd5e1" if is_sys else "#23283a"
        self.setStyleSheet(f"QDialog {{ background-color: {bg}; color: {fg}; }}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # Header
        top_row = QHBoxLayout()
        heading = QLabel(f"📂 {title_text}")
        heading.setStyleSheet("font-size: 16px; font-weight: 800; color: #f59e0b;")
        top_row.addWidget(heading)
        top_row.addStretch(1)

        path_lbl = QLabel(str(path))
        path_lbl.setObjectName("PathBadge")
        top_row.addWidget(path_lbl)
        layout.addLayout(top_row)

        # Stats row
        try:
            stats = file_token_vocab_stats(path) if path.is_file() else {"characters": 0, "tokens": 0, "vocab": 0}
            size_str = format_size(path.stat().st_size) if path.is_file() else "-"
        except Exception:
            stats = {"characters": 0, "tokens": 0, "vocab": 0}
            size_str = "-"

        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        stats_box = QLabel(
            f"Size: <b>{size_str}</b>  •  "
            f"Est. Tokens: <b>{format_estimate(int(stats.get('tokens', 0)), False)}</b>  •  "
            f"Est. Characters: <b>{format_estimate(int(stats.get('characters', 0)), False)}</b>  •  "
            f"Vocab: <b>{format_estimate(int(stats.get('vocab', 0)), False)}</b>"
        )
        stats_box.setStyleSheet("font-size: 12px; color: #94a3b8;")
        stats_row.addWidget(stats_box)
        stats_row.addStretch(1)
        layout.addLayout(stats_row)

        # Content viewer
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet(
            f"background-color: {'#f8fafc' if is_sys else '#0c0d12'}; "
            f"border: 1px solid {border}; border-radius: 8px; font-family: Consolas, monospace; font-size: 12px; padding: 10px;"
        )

        sample_lines: list[str] = []
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    for _ in range(100):
                        line = f.readline()
                        if not line:
                            break
                        sample_lines.append(line.rstrip())
            except Exception as e:
                sample_lines.append(f"Could not read file: {e}")
        elif path.is_dir():
            sample_lines.append(f"Directory: {path}")
            sample_lines.append("Files contained:")
            for p in sorted(path.rglob("*")):
                if p.is_file():
                    sample_lines.append(f"  • {p.name} ({format_size(p.stat().st_size)})")

        text_edit.setPlainText("\n".join(sample_lines) or "No sample content available.")
        layout.addWidget(text_edit, 1)

        # Close button
        btn_box = QHBoxLayout()
        btn_box.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.accept)
        btn_box.addWidget(close_btn)
        layout.addLayout(btn_box)


class CategoryStorageBarWidget(QWidget):
    """Segmented bar showing relative storage / token share across discovered categories."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(12)
        self.setMouseTracking(True)
        self._segments: list[tuple[QRectF, str, int, float, str]] = []
        self._category_data: list[dict[str, Any]] = []

    def set_data(self, category_data: list[dict[str, Any]]) -> None:
        """Update the category list and repaint the bar."""
        self._category_data = category_data
        self.update()

    def apply_theme(self, theme: str = "") -> None:
        """Repaint upon application theme change."""
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = float(self.width())
        height = float(self.height())
        radius = 5.0

        clip_path = QPainterPath()
        clip_path.addRoundedRect(QRectF(0, 0, width, height), radius, radius)
        painter.setClipPath(clip_path)

        total_chars = sum(c.get("characters", 0) for c in self._category_data)
        self._segments.clear()

        is_sys = is_system_theme()
        bg_color = QColor("#e2e8f0" if is_sys else "#161924")
        painter.fillRect(QRectF(0, 0, width, height), bg_color)

        if total_chars <= 0 or not self._category_data:
            painter.end()
            return

        x = 0.0
        for item in self._category_data:
            chars = item.get("characters", 0)
            if chars <= 0:
                continue
            pct = chars / total_chars
            seg_w = pct * width
            rect = QRectF(x, 0, seg_w, height)
            color = QColor(item.get("color", "#f59e0b"))
            painter.fillRect(rect, color)
            self._segments.append((rect, item["name"], chars, pct * 100.0, item.get("color", "#f59e0b")))
            x += seg_w

        # Subtle border outline
        painter.setClipping(False)
        border_color = QColor("#cbd5e1" if is_sys else "#282e42")
        painter.setPen(QPen(border_color, 1.0))
        painter.drawRoundedRect(QRectF(0.5, 0.5, width - 1.0, height - 1.0), radius, radius)
        painter.end()

    def mouseMoveEvent(self, event: Any) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        global_pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
        for rect, cat_name, chars, pct, _ in self._segments:
            if rect.contains(pos):
                tip = f"<b>{cat_name}</b>: {pct:.1f}% ({format_token_count(chars)} chars)"
                self.setToolTip(tip)
                QToolTip.showText(global_pos, tip, self)
                return
        self.setToolTip("")

    def leaveEvent(self, event: Any) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)


def build_dataset_plan_tab(window: Any) -> QWidget:
    """Build the dataset blueprint page matching Model Architecture Studio design.

    Args:
        window: Main application window that owns shared helper methods.

    Returns:
        Dataset blueprint page widget.
    """
    page = QWidget()
    page.setObjectName("Page")
    outer = QVBoxLayout(page)
    outer.setContentsMargins(18, 14, 18, 12)
    outer.setSpacing(10)

    scroll = QScrollArea()
    scroll.setObjectName("PageScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    content = QWidget()
    content.setObjectName("PageContent")
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)

    # =========================================================================
    # Header: Title + Path Badge + 4 Telemetry Metric Chips
    # =========================================================================
    header_row = QHBoxLayout()
    header_row.setContentsMargins(0, 0, 0, 0)
    header_row.setSpacing(10)

    title = QLabel("Dataset Sources")
    title.setObjectName("PageTitle")
    header_row.addWidget(title)

    active_data_root = blueprint_data_root(window)
    window.blueprint_data_root = active_data_root

    window.dataset_plan_source_label = QLabel(f"Source: {active_data_root}")
    window.dataset_plan_source_label.setObjectName("PathBadge")
    window.dataset_plan_source_label.setMaximumWidth(340)
    window._tip(window.dataset_plan_source_label, f"Active data root: {active_data_root}")
    header_row.addWidget(window.dataset_plan_source_label)

    header_row.addStretch(1)

    window.sources_total_corpora_chip = window._metric_chip("Total Corpora: 0", "Total detected corpus categories.")
    window.sources_total_corpora_chip.setObjectName("RecipeChip")
    window.sources_total_corpora_chip.setMinimumWidth(110)

    window.sources_disk_usage_chip = window._metric_chip("Disk Usage: 0 B", "Total size of local datasets on disk.")
    window.sources_disk_usage_chip.setObjectName("RecipeChipAmber")
    window.sources_disk_usage_chip.setMinimumWidth(110)

    window.sources_total_chars_chip = window._metric_chip("Total Chars: 0", "Estimated total characters across local datasets.")
    window.sources_total_chars_chip.setObjectName("RecipeChipAmber")
    window.sources_total_chars_chip.setMinimumWidth(110)

    window.sources_status_chip = window._metric_chip("Status: Ready", "Ingestion readiness status.")
    window.sources_status_chip.setObjectName("RecipeChipGreen")
    window.sources_status_chip.setMinimumWidth(95)
    window.sources_status_chip.setMaximumWidth(125)

    header_row.addWidget(window.sources_total_corpora_chip)
    header_row.addWidget(window.sources_disk_usage_chip)
    header_row.addWidget(window.sources_total_chars_chip)
    header_row.addWidget(window.sources_status_chip)
    layout.addLayout(header_row)

    # =========================================================================
    # Segmented Category Filter Bar
    # =========================================================================
    filter_bar = QHBoxLayout()
    filter_bar.setContentsMargins(0, 0, 0, 2)
    filter_bar.setSpacing(8)

    window.filter_btn_all = QPushButton("All Sources")
    window.filter_btn_all.setObjectName("SourceFilterBtn")
    window.filter_btn_all.setCheckable(True)
    window.filter_btn_all.setChecked(True)

    window.filter_btn_base = QPushButton("Pre-training Base")
    window.filter_btn_base.setObjectName("SourceFilterBtn")
    window.filter_btn_base.setCheckable(True)

    window.filter_btn_instruct = QPushButton("Instruction && Reasoning")
    window.filter_btn_instruct.setObjectName("SourceFilterBtn")
    window.filter_btn_instruct.setCheckable(True)

    window.filter_btn_forge = QPushButton("Synthetic Forge")
    window.filter_btn_forge.setObjectName("SourceFilterBtn")
    window.filter_btn_forge.setCheckable(True)

    filter_bar.addWidget(window.filter_btn_all)
    filter_bar.addWidget(window.filter_btn_base)
    filter_bar.addWidget(window.filter_btn_instruct)
    filter_bar.addWidget(window.filter_btn_forge)
    filter_bar.addStretch(1)
    layout.addLayout(filter_bar)

    # Active category filter state
    window._active_source_filter = "all"

    # =========================================================================
    # Two-Column Body: 3 Cards Left + Full-Height Inventory Right
    # =========================================================================
    body_grid = QGridLayout()
    body_grid.setHorizontalSpacing(14)
    body_grid.setVerticalSpacing(10)

    # -------------------------------------------------------------------------
    # Left Card 1: LOCAL REPOSITORY & STORAGE
    # -------------------------------------------------------------------------
    external_layout = QVBoxLayout()
    external_layout.setSpacing(8)
    external_layout.setContentsMargins(0, 0, 0, 0)

    path_box = QHBoxLayout()
    path_box.setSpacing(6)

    window.external_dataset_dir = QLineEdit(str(Path.home() / "drunkenbot_datasets" / "default"))
    window.external_dataset_dir.setPlaceholderText("Install folder path...")
    window._tip(window.external_dataset_dir, "Folder where downloaded dataset categories are installed and used as the ingestion source.")
    path_box.addWidget(window.external_dataset_dir, 1)

    window.external_dataset_version = QLabel("v17.0.0")
    window.external_dataset_version.setObjectName("VersionBadge")
    window._tip(window.external_dataset_version, "Version recorded from the installed dataset manifest.")
    path_box.addWidget(window.external_dataset_version)

    browse_btn = QPushButton("Browse")
    browse_btn.setFixedWidth(74)
    browse_btn.clicked.connect(lambda: window._browse(window.external_dataset_dir, True))
    window._tip(browse_btn, "Browse for dataset install folder.")
    path_box.addWidget(browse_btn)
    external_layout.addLayout(path_box)

    window.external_dataset_download_button = QPushButton("Sync Latest Bundled Data")
    window.external_dataset_download_button.setObjectName("AmberActionBtn")
    window.external_dataset_download_button.setFixedHeight(32)
    window._tip(window.external_dataset_download_button, "Download, verify, and extract the latest dataset release into the install folder.")
    window.external_dataset_download_button.clicked.connect(window.download_latest_external_dataset)
    trial_mode = not bool(QApplication.instance().property("license_valid"))
    window.external_dataset_download_button.setEnabled(not trial_mode)
    external_layout.addWidget(window.external_dataset_download_button)

    external_card = window._card("LOCAL REPOSITORY & STORAGE", external_layout)

    # -------------------------------------------------------------------------
    # Left Card 2: REMOTE HUB & HUGGING FACE
    # -------------------------------------------------------------------------
    remote_layout = QVBoxLayout()
    remote_layout.setSpacing(8)
    remote_layout.setContentsMargins(0, 0, 0, 0)

    purpose_row = QHBoxLayout()
    purpose_row.setSpacing(8)
    purpose_lbl = QLabel("Purpose:")
    purpose_lbl.setStyleSheet("font-size: 12px; font-weight: 600;")
    purpose_row.addWidget(purpose_lbl)

    window.dataset_stage = QComboBox()
    workflow_names = sorted(
        (path.name for path in active_data_root.iterdir() if path.is_dir()),
        key=str.casefold,
    ) if active_data_root.exists() else []
    workflow_names = list(dict.fromkeys([
        "base",
        "instruction",
        "conversation",
        "tool_call",
        "code",
        "thinking",
        *workflow_names,
    ]))
    window.dataset_stage.addItems(workflow_names)
    window.dataset_stage.setEnabled(True)
    purpose_row.addWidget(window.dataset_stage, 1)

    window.include_conversation_datasets = QCheckBox("Online")
    window.include_conversation_datasets.setChecked(False)
    window.include_conversation_datasets.setEnabled(not trial_mode)
    purpose_row.addWidget(window.include_conversation_datasets)
    remote_layout.addLayout(purpose_row)

    window.conversation_datasets_status = QLabel("Base pretraining: choose optional online corpus datasets, or use local folders only.")
    window.conversation_datasets_status.setObjectName("Muted")
    window.conversation_datasets_status.setWordWrap(True)
    remote_layout.addWidget(window.conversation_datasets_status)

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
    remote_layout.addWidget(window.conversation_dataset_button)

    window.custom_huggingface_dataset = QLineEdit()
    window.custom_huggingface_dataset.setPlaceholderText("Hugging Face repository search... e.g. owner/dataset")
    window._tip(window.custom_huggingface_dataset, "Optional Hugging Face dataset repository to load in addition to the selected presets.")
    remote_layout.addWidget(window.custom_huggingface_dataset)

    hub_bottom_row = QHBoxLayout()
    hub_bottom_row.setSpacing(8)
    row_lbl = QLabel("Row count:")
    row_lbl.setStyleSheet("font-size: 12px; font-weight: 600;")
    hub_bottom_row.addWidget(row_lbl)

    window.conversation_sample_limit = window._spin(0, 2_000_000, 20000)
    window.conversation_sample_limit.setMaximumHeight(30)
    window.conversation_sample_limit.setEnabled(False)
    window.include_conversation_datasets.toggled.connect(window.conversation_sample_limit.setEnabled)
    window.include_conversation_datasets.toggled.connect(window._update_online_dataset_stage_controls)
    window.dataset_stage.currentTextChanged.connect(window._update_online_dataset_stage_controls)
    hub_bottom_row.addWidget(window.conversation_sample_limit)

    window.custom_huggingface_download = QPushButton("Import from Hub")
    window.custom_huggingface_download.clicked.connect(window._download_custom_huggingface_dataset)
    window.custom_huggingface_download.setEnabled(not trial_mode)
    window._tip(window.custom_huggingface_download, "Enable the custom dataset and download it during the next dataset preparation run.")
    hub_bottom_row.addWidget(window.custom_huggingface_download, 1)
    remote_layout.addLayout(hub_bottom_row)

    conversation_card = window._card("REMOTE HUB & HUGGING FACE", remote_layout)

    # -------------------------------------------------------------------------
    # Left Card 3: NEURAL SYNTHETIC FORGE
    # -------------------------------------------------------------------------
    forge_layout = QVBoxLayout()
    forge_layout.setSpacing(8)
    forge_layout.setContentsMargins(0, 0, 0, 0)

    record_row = QHBoxLayout()
    record_row.setSpacing(8)
    record_lbl = QLabel("Record count:")
    record_lbl.setStyleSheet("font-size: 12px; font-weight: 600;")
    record_row.addWidget(record_lbl)

    window.forge_sample_count = window._spin(50, 10000, 300)
    window.forge_sample_count.setMaximumHeight(30)
    window._tip(window.forge_sample_count, "Number of synthetic agent or identity records to generate.")
    record_row.addWidget(window.forge_sample_count)
    record_row.addStretch(1)
    forge_layout.addLayout(record_row)

    forge_btns = QHBoxLayout()
    forge_btns.setSpacing(8)

    window.generate_agent_button = QPushButton("Forge Agent Tool Data")
    window.generate_agent_button.setObjectName("ForgeActionBtn")
    window.generate_agent_button.clicked.connect(window.generate_synthetic_agent_data)
    window._tip(
        window.generate_agent_button,
        "Generate multi-hop web search, Python calculation, and contrastive negative tool trajectories directly into training_data/tool_call/.",
    )
    forge_btns.addWidget(window.generate_agent_button)

    window.generate_identity_button = QPushButton("Forge Identity Facts")
    window.generate_identity_button.setObjectName("ForgeActionBtn")
    window.generate_identity_button.clicked.connect(window.generate_synthetic_identity_data)
    window._tip(
        window.generate_identity_button,
        "Generate combinatorial model identity and self-awareness sentences directly into training_data/identity/.",
    )
    forge_btns.addWidget(window.generate_identity_button)
    forge_layout.addLayout(forge_btns)

    forge_card = window._card("NEURAL SYNTHETIC FORGE", forge_layout)

    # -------------------------------------------------------------------------
    # Right Column: DATASET INVENTORY & INSPECTOR (Concept Art Design)
    # -------------------------------------------------------------------------
    inventory_layout = QVBoxLayout()
    inventory_layout.setSpacing(8)
    inventory_layout.setContentsMargins(0, 0, 0, 0)

    toolbar_row = QHBoxLayout()
    toolbar_row.setSpacing(8)

    window.dataset_search_input = QLineEdit()
    window.dataset_search_input.setObjectName("TreeFilterInput")
    window.dataset_search_input.setPlaceholderText("Filter datasets...")
    toolbar_row.addWidget(window.dataset_search_input, 1)

    window.dataset_plan_refresh_button = QPushButton("Refresh Disk")
    window.dataset_plan_refresh_button.setFixedWidth(110)
    window.dataset_plan_refresh_button.clicked.connect(window.refresh_dataset_blueprint_files)
    window._tip(window.dataset_plan_refresh_button, "Reload this tree to include newly copied files and folders.")
    toolbar_row.addWidget(window.dataset_plan_refresh_button)
    inventory_layout.addLayout(toolbar_row)

    window.default_data_tree_updating = False
    window.default_data_tree = QTreeWidget()
    window.default_data_tree.setObjectName("DatasetInventoryTree")
    window.default_data_tree.setHeaderLabels(["Dataset Category", "Type", "Tokens / Chars", "Vocab Size", "Action"])
    window.default_data_tree.setRootIsDecorated(True)
    window.default_data_tree.setIndentation(18)
    window.default_data_tree.setAlternatingRowColors(False)
    window.default_data_tree.setMinimumHeight(440)
    window.default_data_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    tree_header = window.default_data_tree.header()
    tree_header.setStretchLastSection(False)
    tree_header.setSectionResizeMode(0, QHeaderView.Stretch)
    tree_header.setSectionResizeMode(1, QHeaderView.Fixed)
    tree_header.setSectionResizeMode(2, QHeaderView.Fixed)
    tree_header.setSectionResizeMode(3, QHeaderView.Fixed)
    tree_header.setSectionResizeMode(4, QHeaderView.Fixed)
    window.default_data_tree.setColumnWidth(1, 134)
    window.default_data_tree.setColumnWidth(2, 122)
    window.default_data_tree.setColumnWidth(3, 84)
    window.default_data_tree.setColumnWidth(4, 74)

    window.default_data_actions = {}
    window.default_data_category_items = {}
    window.default_data_tree.itemChanged.connect(window._handle_default_data_tree_changed)
    inventory_layout.addWidget(window.default_data_tree, 1)

    # Category Proportion Bar
    window.category_storage_bar = CategoryStorageBarWidget()
    inventory_layout.addWidget(window.category_storage_bar)

    # Legend container beneath bar
    window.category_legend_widget = QWidget()
    window.category_legend_layout = QHBoxLayout(window.category_legend_widget)
    window.category_legend_layout.setContentsMargins(2, 2, 2, 2)
    window.category_legend_layout.setSpacing(12)
    inventory_layout.addWidget(window.category_legend_widget)

    inventory_card = window._card("DATASET INVENTORY & INSPECTOR", inventory_layout)
    if inventory_card.layout() is not None:
        inventory_card.layout().setStretch(1, 1)
    inventory_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # Add cards to body_grid
    body_grid.addWidget(external_card, 0, 0)
    body_grid.addWidget(conversation_card, 1, 0)
    body_grid.addWidget(forge_card, 2, 0)
    body_grid.addWidget(inventory_card, 0, 1, 4, 1)
    body_grid.setRowStretch(0, 0)
    body_grid.setRowStretch(1, 0)
    body_grid.setRowStretch(2, 0)
    body_grid.setRowStretch(3, 1)
    body_grid.setColumnStretch(0, 4)
    body_grid.setColumnStretch(1, 6)
    layout.addLayout(body_grid, 1)

    # =========================================================================
    # Filter & Search Handlers
    # =========================================================================
    def _apply_filters() -> None:
        q = window.dataset_search_input.text().strip().lower()
        active_filter = getattr(window, "_active_source_filter", "all")
        tree = window.default_data_tree

        count = tree.topLevelItemCount()
        for i in range(count):
            item = tree.topLevelItem(i)
            cat_data = item.data(0, Qt.UserRole)
            if not isinstance(cat_data, dict):
                continue
            cat = cat_data.get("category", "").lower()
            cat_name = cat_data.get("name", "") or dataset_category_label(cat)

            # Category filter match
            if active_filter == "all":
                stage_match = True
            elif active_filter == "base":
                stage_match = any(k in cat for k in ["base", "prose", "wiki", "book", "web", "general", "clean", "academic"])
            elif active_filter == "instruct":
                stage_match = any(k in cat for k in ["instruct", "reason", "math", "code", "chat", "eval", "dialog"])
            elif active_filter == "forge":
                stage_match = any(k in cat for k in ["tool", "agent", "identity", "synthetic", "forge"])
            else:
                stage_match = True

            # Search text match
            cat_text = cat_name.lower()
            child_count = item.childCount()
            any_child_match = False
            for c in range(child_count):
                child = item.child(c)
                c_data = child.data(0, Qt.UserRole) or {}
                c_name = c_data.get("name", "") or Path(c_data.get("path", "")).name
                child_text = c_name.lower()
                child_match = (not q) or (q in child_text) or (q in cat_text)
                child.setHidden(not (stage_match and child_match))
                if child_match:
                    any_child_match = True

            item_match = stage_match and ((not q) or (q in cat_text) or any_child_match)
            item.setHidden(not item_match)
            if q and any_child_match and stage_match:
                item.setExpanded(True)
            elif not q and active_filter == "all":
                item.setExpanded(False)

    def _set_active_filter(active_key: str) -> None:
        window._active_source_filter = active_key
        window.filter_btn_all.setChecked(active_key == "all")
        window.filter_btn_base.setChecked(active_key == "base")
        window.filter_btn_instruct.setChecked(active_key == "instruct")
        window.filter_btn_forge.setChecked(active_key == "forge")
        _apply_filters()

    window.filter_btn_all.clicked.connect(lambda: _set_active_filter("all"))
    window.filter_btn_base.clicked.connect(lambda: _set_active_filter("base"))
    window.filter_btn_instruct.clicked.connect(lambda: _set_active_filter("instruct"))
    window.filter_btn_forge.clicked.connect(lambda: _set_active_filter("forge"))
    window.dataset_search_input.textChanged.connect(_apply_filters)

    # Initial data load
    populate_default_data_tree(window, active_data_root)

    window._update_online_dataset_stage_controls()
    window._apply_dataset_license_gating()

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

        total_global_characters = 0
        total_global_bytes = 0
        category_bar_data: list[dict[str, Any]] = []

        sorted_categories = sorted(grouped_files, key=dataset_category_label)
        palette = CATEGORY_PALETTE

        for idx, category in enumerate(sorted_categories):
            color = palette[idx % len(palette)]
            type_symbol, type_label = infer_category_type(category)

            category_item = QTreeWidgetItem()
            category_item.setData(0, Qt.UserRole, {"kind": "category", "category": category, "name": dataset_category_label(category)})
            category_item.setCheckState(0, Qt.Checked)
            tree.addTopLevelItem(category_item)
            window.default_data_category_items[category] = category_item

            total_characters = 0
            total_vocab = 0
            total_bytes = 0
            sampled_category = False
            first_path: Path | None = None

            for path in sorted(grouped_files[category], key=lambda item: item.name.lower()):
                if first_path is None:
                    first_path = path
                try:
                    stats = file_token_vocab_stats(path)
                except OSError:
                    stats = {"characters": 0, "vocab": 0, "sampled": False, "bytes": 0}
                sampled = bool(stats.get("sampled", False))
                characters = int(stats.get("characters", 0))
                vocab = int(stats.get("vocab", 0))
                bytes_cnt = int(stats.get("bytes", 0))

                child = QTreeWidgetItem()
                child.setToolTip(0, str(path))
                child.setData(0, Qt.UserRole, {"kind": "file", "path": str(path), "category": category, "name": path.name})
                child.setCheckState(0, Qt.Checked)
                category_item.addChild(child)
                window.default_data_actions[str(path)] = child

                # Child Column 0: Pill Toggle + File Icon + File Name
                child_col0 = QWidget()
                c0_l = QHBoxLayout(child_col0)
                c0_l.setContentsMargins(20, 0, 8, 0)
                c0_l.setSpacing(10)
                c0_l.setAlignment(Qt.AlignVCenter)

                ch_toggle = PillToggleSwitch(checked=True)
                child._toggle = ch_toggle

                def _make_child_toggle_handler(ch=child):
                    def _handler(checked: bool) -> None:
                        ch.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                        if hasattr(window, "_handle_default_data_tree_changed"):
                            window._handle_default_data_tree_changed(ch, 0)
                    return _handler

                ch_toggle.toggled.connect(_make_child_toggle_handler(child))
                c0_l.addWidget(ch_toggle)

                f_name_lbl = QLabel(path.name)
                f_name_lbl.setObjectName("FileNameLabel")
                c0_l.addWidget(f_name_lbl, 1)
                tree.setItemWidget(child, 0, child_col0)

                # Child Column 1: Suffix Badge
                child_col1 = QWidget()
                c1_l = QHBoxLayout(child_col1)
                c1_l.setContentsMargins(4, 0, 6, 0)
                c1_l.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                s_lbl = QLabel(path.suffix.lower())
                s_lbl.setObjectName("FileSuffixBadge")
                c1_l.addWidget(s_lbl)
                c1_l.addStretch(1)
                tree.setItemWidget(child, 1, child_col1)

                # Child Column 2: Tokens / Chars
                child_col2 = QWidget()
                c2_l = QHBoxLayout(child_col2)
                c2_l.setContentsMargins(4, 0, 6, 0)
                c2_l.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                f_dual = f"{format_estimate(int(characters * 0.28), sampled)} / {format_token_count(characters)}"
                f_metric_lbl = QLabel(f_dual)
                f_metric_lbl.setObjectName("MetricMonoLabel")
                c2_l.addWidget(f_metric_lbl)
                c2_l.addStretch(1)
                tree.setItemWidget(child, 2, child_col2)

                # Child Column 3: Vocab + Mini Meter
                child_col3 = QWidget()
                c3_l = QHBoxLayout(child_col3)
                c3_l.setContentsMargins(4, 0, 6, 0)
                c3_l.setSpacing(8)
                c3_l.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                f_vocab_lbl = QLabel(format_token_count(vocab))
                f_vocab_lbl.setObjectName("MetricMonoLabel")
                c3_l.addWidget(f_vocab_lbl)
                c_mag = 5 if vocab >= 500_000 else (4 if vocab >= 50_000 else (3 if vocab >= 5_000 else (2 if vocab >= 500 else 1)))
                c_meter = MiniSignalBarWidget(magnitude=c_mag, color_hex=color)
                c3_l.addWidget(c_meter)
                c3_l.addStretch(1)
                tree.setItemWidget(child, 3, child_col3)

                # Child Column 4: Preview Button
                child_col4 = QWidget()
                c4_l = QHBoxLayout(child_col4)
                c4_l.setContentsMargins(2, 0, 6, 0)
                c4_l.setAlignment(Qt.AlignCenter)
                f_prev_btn = QPushButton("Preview")
                f_prev_btn.setObjectName("PreviewBtn")
                f_prev_btn.setFixedSize(68, 24)

                def _make_file_preview(fn=path.name, fp=path):
                    def _preview() -> None:
                        dlg = DatasetPreviewDialog(fn, fp, window)
                        dlg.exec()
                    return _preview

                f_prev_btn.clicked.connect(_make_file_preview(path.name, path))
                c4_l.addWidget(f_prev_btn)
                tree.setItemWidget(child, 4, child_col4)

                total_characters += characters
                total_vocab += vocab
                total_bytes += bytes_cnt
                sampled_category = sampled_category or sampled

            category_item.setExpanded(False)

            # Category Column 0: Pill Toggle Switch + Category Name
            col0_w = QWidget()
            col0_l = QHBoxLayout(col0_w)
            col0_l.setContentsMargins(4, 0, 8, 0)
            col0_l.setSpacing(10)
            col0_l.setAlignment(Qt.AlignVCenter)

            cat_toggle = PillToggleSwitch(checked=True)
            category_item._toggle = cat_toggle

            def _make_cat_toggle_handler(cat_it=category_item):
                def _handler(checked: bool) -> None:
                    cat_it.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                    for c_idx in range(cat_it.childCount()):
                        ch_node = cat_it.child(c_idx)
                        if hasattr(ch_node, "_toggle") and ch_node._toggle is not None:
                            ch_node._toggle.setChecked(checked, block_signal=True)
                        ch_node.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                    if hasattr(window, "_handle_default_data_tree_changed"):
                        window._handle_default_data_tree_changed(cat_it, 0)
                return _handler

            cat_toggle.toggled.connect(_make_cat_toggle_handler(category_item))
            col0_l.addWidget(cat_toggle)

            cat_name_lbl = QLabel(dataset_category_label(category))
            cat_name_lbl.setObjectName("CategoryNameLabel")
            col0_l.addWidget(cat_name_lbl, 1)
            tree.setItemWidget(category_item, 0, col0_w)

            # Category Column 1: Glowing Symbol + Type Label
            col1_w = QWidget()
            col1_l = QHBoxLayout(col1_w)
            col1_l.setContentsMargins(4, 0, 6, 0)
            col1_l.setSpacing(8)
            col1_l.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            type_sym_lbl = QLabel(type_symbol)
            type_sym_lbl.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: bold;")
            col1_l.addWidget(type_sym_lbl)

            type_text_lbl = QLabel(type_label)
            type_text_lbl.setObjectName("CategoryTypeLabel")
            col1_l.addWidget(type_text_lbl)
            col1_l.addStretch(1)
            tree.setItemWidget(category_item, 1, col1_w)

            # Category Column 2: Tokens / Chars
            col2_w = QWidget()
            col2_l = QHBoxLayout(col2_w)
            col2_l.setContentsMargins(4, 0, 6, 0)
            col2_l.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            cat_dual_str = f"{format_estimate(int(total_characters * 0.28), sampled_category)} / {format_token_count(total_characters)}"
            cat_metric_lbl = QLabel(cat_dual_str)
            cat_metric_lbl.setObjectName("MetricMonoLabel")
            col2_l.addWidget(cat_metric_lbl)
            col2_l.addStretch(1)
            tree.setItemWidget(category_item, 2, col2_w)

            # Category Column 3: Vocab Size + Mini Signal Bar
            col3_w = QWidget()
            col3_l = QHBoxLayout(col3_w)
            col3_l.setContentsMargins(4, 0, 6, 0)
            col3_l.setSpacing(8)
            col3_l.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            cat_vocab_lbl = QLabel(format_token_count(total_vocab))
            cat_vocab_lbl.setObjectName("MetricMonoLabel")
            col3_l.addWidget(cat_vocab_lbl)
            cat_mag = 5 if total_vocab >= 500_000 else (4 if total_vocab >= 50_000 else (3 if total_vocab >= 5_000 else (2 if total_vocab >= 500 else 1)))
            cat_meter = MiniSignalBarWidget(magnitude=cat_mag, color_hex=color)
            col3_l.addWidget(cat_meter)
            col3_l.addStretch(1)
            tree.setItemWidget(category_item, 3, col3_w)

            # Category Column 4: Preview Action Button
            col4_w = QWidget()
            col4_l = QHBoxLayout(col4_w)
            col4_l.setContentsMargins(2, 0, 6, 0)
            col4_l.setAlignment(Qt.AlignCenter)
            cat_prev_btn = QPushButton("Preview")
            cat_prev_btn.setObjectName("PreviewBtn")
            cat_prev_btn.setFixedSize(68, 24)

            preview_target = first_path or (root / category)

            def _make_cat_preview(cn=dataset_category_label(category), cp=preview_target):
                def _preview() -> None:
                    dlg = DatasetPreviewDialog(cn, cp, window)
                    dlg.exec()
                return _preview

            cat_prev_btn.clicked.connect(_make_cat_preview(dataset_category_label(category), preview_target))
            col4_l.addWidget(cat_prev_btn)
            tree.setItemWidget(category_item, 4, col4_w)

            total_global_characters += total_characters
            total_global_bytes += total_bytes
            category_bar_data.append({
                "name": dataset_category_label(category),
                "slug": category,
                "characters": total_characters,
                "bytes": total_bytes,
                "color": color,
            })
    finally:
        tree.blockSignals(False)

    if not window.default_data_actions:
        window.default_data_tree.addTopLevelItem(QTreeWidgetItem(["No project/default data files were found.", "", "", "", ""]))

    # Update live telemetry chips
    total_corpora = len(window.default_data_category_items)
    if hasattr(window, "sources_total_corpora_chip"):
        window.sources_total_corpora_chip.setText(f"Total Corpora: {total_corpora}")
    if hasattr(window, "sources_disk_usage_chip"):
        window.sources_disk_usage_chip.setText(f"Disk Usage: {format_size(total_global_bytes)}")
    if hasattr(window, "sources_total_chars_chip"):
        window.sources_total_chars_chip.setText(f"Total Chars: {format_token_count(total_global_characters)}")
    if hasattr(window, "sources_status_chip"):
        window.sources_status_chip.setText("Status: Ready" if total_corpora > 0 else "Status: Empty")

    # Update category storage proportion bar
    if hasattr(window, "category_storage_bar") and window.category_storage_bar is not None:
        window.category_storage_bar.set_data(category_bar_data)

    # Update legend
    if hasattr(window, "category_legend_layout") and window.category_legend_layout is not None:
        while window.category_legend_layout.count():
            child_item = window.category_legend_layout.takeAt(0)
            if child_item.widget():
                child_item.widget().deleteLater()

        sorted_cats = sorted(category_bar_data, key=lambda c: c["characters"], reverse=True)
        top_cats = sorted_cats[:5]
        other_chars = sum(c["characters"] for c in sorted_cats[5:])

        for cat in top_cats:
            item_w = QWidget()
            h = QHBoxLayout(item_w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {cat['color']}; font-size: 11px;")
            name_lbl = QLabel(cat["name"])
            name_lbl.setObjectName("LegendLabel")
            h.addWidget(dot)
            h.addWidget(name_lbl)
            window.category_legend_layout.addWidget(item_w)

        if other_chars > 0:
            item_w = QWidget()
            h = QHBoxLayout(item_w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            dot = QLabel("●")
            dot.setStyleSheet("color: #64748b; font-size: 11px;")
            name_lbl = QLabel("Others")
            name_lbl.setObjectName("LegendLabel")
            h.addWidget(dot)
            h.addWidget(name_lbl)
            window.category_legend_layout.addWidget(item_w)

        window.category_legend_layout.addStretch(1)

    # Update filter button counts
    base_count = sum(1 for c in grouped_files if any(k in c.lower() for k in ["base", "prose", "wiki", "book", "web", "general", "clean", "academic"]))
    instruct_count = sum(1 for c in grouped_files if any(k in c.lower() for k in ["instruct", "reason", "math", "code", "chat", "eval", "dialog"]))
    forge_count = sum(1 for c in grouped_files if any(k in c.lower() for k in ["tool", "agent", "identity", "synthetic", "forge"]))

    if hasattr(window, "filter_btn_all"):
        window.filter_btn_all.setText(f"All Sources ({total_corpora})")
    if hasattr(window, "filter_btn_base"):
        window.filter_btn_base.setText(f"Pre-training Base ({base_count})")
    if hasattr(window, "filter_btn_instruct"):
        window.filter_btn_instruct.setText(f"Instruction && Reasoning ({instruct_count})")
    if hasattr(window, "filter_btn_forge"):
        window.filter_btn_forge.setText(f"Synthetic Forge ({forge_count})")
