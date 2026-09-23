"""Dataset Recipe Matrix tab for configuring multi-discipline category mixture."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QPoint, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from engine.dataset_recipe import (
    DEFAULT_RECIPE_PRESETS,
    DatasetRecipe,
    RecipeCategory,
    create_frontier_11_pillar_recipe,
    get_default_recipe,
    resolve_category_directories,
    scan_category_disk_stats,
)
from interface.theme import is_system_theme

LOGGER = logging.getLogger(__name__)

# Curated palette of modern, vibrant colors for category visualization
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
    "#e11d48",  # Rose
    "#84cc16",  # Lime Green
    "#d97706",  # Ochre
    "#4f46e5",  # Indigo
]


def format_token_count(tokens: int) -> str:
    """Format token count into a readable string (e.g. 62.5M or 500K)."""
    if tokens >= 1_000_000_000:
        return f"{tokens / 1_000_000_000:.2f}B"
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.0f}K"
    return str(tokens)


def get_candidate_roots(window: Any) -> list[Path]:
    """Return all plausible directory roots where category datasets may reside."""
    roots: list[Path] = []

    # 1. Project-bound directories
    if hasattr(window, "current_project_file") and window.current_project_file:
        p_dir = window.current_project_file.parent
        roots.extend([p_dir / "training_data", p_dir / "datasets", p_dir])
    elif hasattr(window, "current_project_dir") and window.current_project_dir:
        p_dir = Path(window.current_project_dir)
        roots.extend([p_dir / "training_data", p_dir / "datasets", p_dir])

    # 2. Input directory from ingestion tab
    if hasattr(window, "input_dir") and window.input_dir.text().strip():
        in_p = Path(window.input_dir.text().strip())
        if in_p.is_dir():
            roots.append(in_p)

    # 3. Known dataset repo
    dataset_repo = Path(r"E:\AI_Projects\dataset")
    if dataset_repo.is_dir():
        roots.append(dataset_repo)

    # 4. Standard runtime and workspace directories
    roots.extend([
        Path.cwd() / "training_data",
        Path.cwd() / "default_data",
        Path.cwd(),
    ])

    # Deduplicate existing directories
    seen = set()
    valid_roots = []
    for r in roots:
        if r.exists():
            resolved = r.resolve()
            if resolved not in seen:
                seen.add(resolved)
                valid_roots.append(r)
    return valid_roots


class CategoryFilesDialog(QDialog):
    """Dialog displaying all source files, sizes, and token counts for a category."""

    def __init__(
        self,
        category: RecipeCategory,
        candidate_roots: list[Path],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.category = category
        self.candidate_roots = candidate_roots
        self.setWindowTitle(f"Category Files: {category.name} ({category.slug})")
        self.resize(760, 500)
        self.setStyleSheet(
            "QDialog { background-color: #181820; color: #ececf1; }"
            "QLabel { color: #d4d4d8; font-size: 12px; }"
            "QTreeWidget { background-color: #1a1a24; border: 1px solid #333342; border-radius: 6px; color: #f3f4f6; }"
            "QHeaderView::section { background-color: #272736; color: #9ca3af; padding: 6px; font-weight: bold; border: 1px solid #333342; font-size: 11px; }"
            "QPushButton { background-color: #3b3b4a; color: white; border: 1px solid #4f4f62; padding: 6px 14px; border-radius: 4px; font-weight: 600; }"
            "QPushButton:hover { background-color: #4f4f62; }"
        )
        self._setup_ui()
        self._load_files()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        top_box = QHBoxLayout()
        info_box = QVBoxLayout()
        info_box.setSpacing(3)

        title = QLabel(f"📂 {self.category.name}")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f59e0b;")
        slug_lbl = QLabel(
            f"Category Slug: <b>{self.category.slug}</b>  •  "
            f"Target Quota: <b>{self.category.target_percentage:.1f}%</b> ({format_token_count(self.category.tokens_estimated)} tokens)"
        )
        slug_lbl.setStyleSheet("font-size: 11px; color: #9ca3af;")
        info_box.addWidget(title)
        info_box.addWidget(slug_lbl)
        top_box.addLayout(info_box)
        top_box.addStretch(1)

        # Rescan button
        rescan_btn = QPushButton("↻ Refresh Files")
        rescan_btn.setToolTip("Re-scan category folder to detect newly added or updated files")
        rescan_btn.clicked.connect(self._load_files)
        top_box.addWidget(rescan_btn)

        # Open in Explorer button
        explorer_btn = QPushButton("Open in File Explorer")
        explorer_btn.setToolTip("Open the category directory in Windows File Explorer")
        explorer_btn.clicked.connect(self._open_in_explorer)
        top_box.addWidget(explorer_btn)

        layout.addLayout(top_box)

        # Stats chips row
        self.stats_row = QHBoxLayout()
        self.files_chip = QLabel("Files: -")
        self.files_chip.setStyleSheet("background-color: #262633; color: #a5b4fc; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        self.files_chip.setToolTip("Total number of data files found in category folders")

        self.size_chip = QLabel("Total Size: -")
        self.size_chip.setStyleSheet("background-color: #262633; color: #34d399; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        self.size_chip.setToolTip("Total uncompressed byte size of files on disk")

        self.tokens_chip = QLabel("Est. Disk Tokens: -")
        self.tokens_chip.setStyleSheet("background-color: #262633; color: #fbbf24; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        self.tokens_chip.setToolTip("Estimated token count available on disk (~3.85 bytes/token)")

        self.paths_chip = QLabel("Folders: -")
        self.paths_chip.setStyleSheet("background-color: #262633; color: #9ca3af; padding: 4px 10px; border-radius: 4px;")
        self.paths_chip.setToolTip("Resolved directory locations on disk")

        self.stats_row.addWidget(self.files_chip)
        self.stats_row.addWidget(self.size_chip)
        self.stats_row.addWidget(self.tokens_chip)
        self.stats_row.addWidget(self.paths_chip, 1)
        layout.addLayout(self.stats_row)

        # Files Tree Widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File Name", "Size", "Estimated Tokens", "Folder Location"])
        self.tree.setColumnWidth(0, 260)
        self.tree.setColumnWidth(1, 100)
        self.tree.setColumnWidth(2, 130)
        self.tree.setAlternatingRowColors(True)
        self.tree.setStyleSheet("alternate-background-color: #1e1e2b; background-color: #161622;")
        layout.addWidget(self.tree, 1)

        # Bottom Close Button
        btn_box = QHBoxLayout()
        btn_box.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_box.addWidget(close_btn)
        layout.addLayout(btn_box)

    def _load_files(self) -> None:
        self.tree.clear()
        stats = scan_category_disk_stats(self.category, self.candidate_roots)
        files: list[Path] = stats.get("files", [])
        dirs: list[Path] = stats.get("directories", [])

        total_bytes = stats.get("disk_bytes", 0)
        total_tokens = stats.get("disk_tokens", 0)

        self.files_chip.setText(f"Files: {len(files)}")
        size_str = f"{total_bytes / (1024*1024):.2f} MB" if total_bytes >= 1024*1024 else f"{total_bytes / 1024:.1f} KB"
        self.size_chip.setText(f"Total Size: {size_str}")
        self.tokens_chip.setText(f"Est. Disk Tokens: ~{format_token_count(total_tokens)}")
        dir_names = ", ".join(d.name for d in dirs) if dirs else "Not found on disk"
        self.paths_chip.setText(f"Folders: {dir_names}")

        if not files:
            item = QTreeWidgetItem(["No files found in category source folders.", "", "", ""])
            self.tree.addTopLevelItem(item)
            return

        for p in sorted(files, key=lambda f: f.name.lower()):
            size = p.stat().st_size
            f_size_str = f"{size / (1024*1024):.2f} MB" if size >= 1024*1024 else f"{size / 1024:.1f} KB"
            f_tokens = int(round(size / 3.85))
            parent_folder = p.parent.name
            item = QTreeWidgetItem([p.name, f_size_str, f"~{format_token_count(f_tokens)}", parent_folder])
            item.setToolTip(0, str(p))
            item.setToolTip(3, str(p.parent))
            self.tree.addTopLevelItem(item)

    def _open_in_explorer(self) -> None:
        dirs = resolve_category_directories(self.category, self.candidate_roots)
        if dirs:
            try:
                os.startfile(str(dirs[0]))
            except Exception as exc:
                QMessageBox.warning(self, "Open Folder Error", f"Could not open folder: {exc}")
        else:
            QMessageBox.information(self, "Folder Not Found", "No folder found on disk for this category yet.")


class RecipeDistributionBar(QWidget):
    """Custom stacked multi-color horizontal bar visualizing category proportions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.recipe: DatasetRecipe | None = None
        self.setFixedHeight(22)
        self.setMinimumWidth(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self._segments: list[tuple[QRectF, RecipeCategory, str]] = []

    def set_recipe(self, recipe: DatasetRecipe) -> None:
        self.recipe = recipe
        self.update()

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        width = float(rect.width())
        height = float(rect.height())

        # Clip to rounded container
        clip_path = QPainterPath()
        clip_path.addRoundedRect(QRectF(0, 0, width, height), 6.0, 6.0)
        painter.setClipPath(clip_path)

        # Draw dark background
        painter.fillRect(rect, QColor("#1e1e24"))

        self._segments.clear()
        if not self.recipe:
            painter.end()
            return

        enabled_cats = [c for c in self.recipe.categories if c.enabled and c.target_percentage > 0]
        total_pct = sum(c.target_percentage for c in enabled_cats)

        if total_pct <= 0:
            painter.setPen(QColor("#777777"))
            painter.drawText(rect, Qt.AlignCenter, "No Active Categories (0.0%)")
            painter.end()
            return

        current_x = 0.0
        scale_denom = max(100.0, total_pct)

        for idx, cat in enumerate(enabled_cats):
            color_hex = CATEGORY_PALETTE[idx % len(CATEGORY_PALETTE)]
            seg_width = (cat.target_percentage / scale_denom) * width
            seg_rect = QRectF(current_x, 0, seg_width, height)
            self._segments.append((seg_rect, cat, color_hex))

            painter.fillRect(seg_rect, QColor(color_hex))

            # Segment border / divider
            if seg_width > 3:
                painter.setPen(QPen(QColor(0, 0, 0, 90), 1.0))
                painter.drawLine(int(current_x + seg_width), 0, int(current_x + seg_width), int(height))

            # Label inside segment if wide enough
            if seg_width > 55:
                painter.setPen(QColor("#ffffff"))
                font = painter.font()
                font.setPixelSize(11)
                font.setBold(True)
                painter.setFont(font)
                pct_text = f"{cat.target_percentage:.1f}%"
                painter.drawText(seg_rect, Qt.AlignCenter, pct_text)

            current_x += seg_width

        # If total < 100, draw unallocated remaining space
        if total_pct < 100.0:
            unalloc_rect = QRectF(current_x, 0, width - current_x, height)
            painter.fillRect(unalloc_rect, QColor("#2a2a32"))
            if (width - current_x) > 80:
                painter.setPen(QColor("#888899"))
                font = painter.font()
                font.setPixelSize(10)
                painter.setFont(font)
                painter.drawText(unalloc_rect, Qt.AlignCenter, f"Unallocated ({100.0 - total_pct:.1f}%)")

        # Outline
        painter.setClipping(False)
        painter.setPen(QPen(QColor("#3f3f4e"), 1.0))
        painter.drawRoundedRect(QRectF(0.5, 0.5, width - 1.0, height - 1.0), 6.0, 6.0)
        painter.end()

    def mouseMoveEvent(self, event: Any) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        global_pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()

        for rect, cat, _ in self._segments:
            if rect.contains(pos):
                disk_info = (
                    f"<br/>Disk: <b>{format_token_count(cat.disk_tokens)} tokens</b> ({cat.files_count} files)"
                    if cat.files_count > 0 else "<br/>Disk: <i>Not scanned yet (click ↻ Refresh)</i>"
                )
                lock_info = "<br/>Status: 🔒 <b>Locked</b>" if cat.locked else ""
                tip = (
                    f"<b>{cat.name}</b> ({cat.slug})<br/>"
                    f"Target Allocation: <b>{cat.target_percentage:.1f}%</b><br/>"
                    f"Projected Quota: <b>{cat.tokens_estimated:,} tokens</b> ({format_token_count(cat.tokens_estimated)})"
                    f"{disk_info}"
                    f"{lock_info}"
                )
                self.setToolTip(tip)
                QToolTip.showText(global_pos, tip, self)
                return

        if self.recipe:
            tot = self.recipe.total_percentage()
            tip = f"<b>Total Mixture</b>: {tot:.1f}% / 100.0% ({len(self.recipe.categories)} categories)"
            self.setToolTip(tip)
            QToolTip.showText(global_pos, tip, self)


class CategoryRowWidget(QFrame):
    """Dynamic row panel representing one dataset category."""

    def __init__(
        self,
        category: RecipeCategory,
        color_hex: str,
        candidate_roots: list[Path],
        on_change_callback: Any,
        on_slider_moved_callback: Any,
        on_delete_callback: Any,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.category = category
        self.color_hex = color_hex
        self.candidate_roots = candidate_roots
        self.on_change = on_change_callback
        self.on_slider_moved = on_slider_moved_callback
        self.on_delete = on_delete_callback
        self._updating = False

        self.setObjectName("CategoryRow")
        self.setFixedHeight(34)
        self.setStyleSheet(
            "#CategoryRow {"
            "  background-color: transparent;"
            "  border: none;"
            "  border-bottom: 1px solid rgba(128, 128, 128, 0.15);"
            "  border-radius: 6px;"
            "  margin-bottom: 1px;"
            "}"
            "#CategoryRow:hover {"
            "  background-color: rgba(128, 128, 128, 0.08);"
            "}"
        )
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(10)

        # 1. Enable Toggle Switch
        self.enabled_check = QCheckBox()
        self.enabled_check.setObjectName("CategoryToggle")
        self.enabled_check.setChecked(self.category.enabled)
        self.enabled_check.setToolTip("Toggle category ON or OFF in the active mixture")
        self.enabled_check.toggled.connect(self._handle_enabled_toggled)
        layout.addWidget(self.enabled_check)

        # 2. Color Swatch
        self.swatch = QFrame()
        self.swatch.setFixedSize(10, 10)
        self.swatch.setStyleSheet(f"background-color: {self.color_hex}; border-radius: 3px;")
        self.swatch.setToolTip(f"Mixture color indicator for {self.category.name}")
        layout.addWidget(self.swatch)

        # 3. Category Name
        self.name_label = QLabel(self.category.name)
        self.name_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        self.name_label.setToolTip(f"Category: {self.category.name} (slug: {self.category.slug})")
        self.name_label.setMinimumWidth(180)
        layout.addWidget(self.name_label)

        # 4. Folder / Source paths badge (Double-click opens file listing window)
        paths_str = ", ".join(self.category.source_paths) if self.category.source_paths else self.category.slug
        self.paths_badge = QLabel(f"📁 {paths_str}")
        self.paths_badge.setObjectName("FolderBadge")
        self.paths_badge.setCursor(Qt.PointingHandCursor)
        self.paths_badge.setFixedHeight(22)
        self.paths_badge.setToolTip("Double-click to inspect all files, byte sizes, and token counts in this category folder.")
        self.paths_badge.setMaximumWidth(150)
        self.paths_badge.mouseDoubleClickEvent = lambda _event: self._open_files_dialog()
        layout.addWidget(self.paths_badge)

        # 5. Continuous Slider (0 - 1000 = 0.0% to 100.0%)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setObjectName("CategorySlider")
        self.slider.setRange(0, 1000)
        self.slider.setValue(int(round(self.category.target_percentage * 10)))
        self.slider.setMinimumWidth(130)
        self.slider.setFixedHeight(18)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.slider.setToolTip(f"Drag to adjust {self.category.name} percentage (0.0% - 100.0%)")
        self.slider.valueChanged.connect(self._handle_slider_changed)
        layout.addWidget(self.slider, 1)

        # 6. Percentage Double Spinbox (0.00% to 100.00%)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 100.0)
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.5)
        self.spin.setSuffix(" %")
        self.spin.setValue(self.category.target_percentage)
        self.spin.setFixedWidth(104)
        self.spin.setFixedHeight(24)
        self.spin.setToolTip("Type or click to set exact percentage")
        self.spin.valueChanged.connect(self._handle_spin_changed)
        layout.addWidget(self.spin)

        # 7. Target Tokens Badge
        self.token_badge = QLabel(f"Quota: {format_token_count(self.category.tokens_estimated)}")
        self.token_badge.setObjectName("QuotaBadge")
        self.token_badge.setAlignment(Qt.AlignCenter)
        self.token_badge.setFixedWidth(88)
        self.token_badge.setFixedHeight(24)
        self.token_badge.setToolTip(f"Projected Target Quota: {self.category.tokens_estimated:,} tokens")
        layout.addWidget(self.token_badge)

        # 8. Disk Tokens Badge
        self.disk_badge = QLabel("Disk: -")
        self.disk_badge.setObjectName("DiskBadge")
        self.disk_badge.setAlignment(Qt.AlignCenter)
        self.disk_badge.setFixedWidth(92)
        self.disk_badge.setFixedHeight(24)
        self.disk_badge.setToolTip("Tokens available on disk across category files")
        layout.addWidget(self.disk_badge)

        # 9. Ratio Lock Toggle Button
        self.lock_btn = QPushButton("🔒 Locked" if self.category.locked else "🔓 Unlocked")
        self.lock_btn.setObjectName("CategoryLockBtn")
        self.lock_btn.setCheckable(True)
        self.lock_btn.setChecked(self.category.locked)
        self.lock_btn.setFixedWidth(78)
        self.lock_btn.setFixedHeight(24)
        self._update_lock_btn_style()
        self.lock_btn.toggled.connect(self._handle_lock_toggled)
        layout.addWidget(self.lock_btn)

        # 10. Delete Button
        self.delete_btn = QPushButton("×")
        self.delete_btn.setObjectName("CategoryDeleteBtn")
        self.delete_btn.setFixedSize(22, 22)
        self.delete_btn.setToolTip("Remove category from recipe")
        self.delete_btn.clicked.connect(self._handle_delete_clicked)
        layout.addWidget(self.delete_btn)

        self._apply_enabled_dimming()
        self.update_disk_stats_badge()

    def _update_lock_btn_style(self) -> None:
        if self.category.locked:
            self.lock_btn.setText("🔒 Locked")
            self.lock_btn.setToolTip("Ratio is LOCKED. Auto-normalize and live sliding will NOT alter this percentage.")
        else:
            self.lock_btn.setText("🔓 Unlocked")
            self.lock_btn.setToolTip("Ratio is UNLOCKED. Can be rebalanced to maintain 100%.")

    def _apply_enabled_dimming(self) -> None:
        is_on = self.category.enabled
        self.slider.setEnabled(is_on)
        self.spin.setEnabled(is_on)
        self.lock_btn.setEnabled(is_on)
        self.name_label.setEnabled(is_on)
        self.token_badge.setEnabled(is_on)
        self.disk_badge.setEnabled(is_on)

    def _handle_enabled_toggled(self, checked: bool) -> None:
        self.category.enabled = checked
        self._apply_enabled_dimming()
        self.on_change()

    def _handle_lock_toggled(self, checked: bool) -> None:
        self.category.locked = checked
        self._update_lock_btn_style()
        self.on_change()

    def _handle_slider_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        val = value / 10.0
        self.category.target_percentage = val
        self.spin.setValue(val)
        self._updating = False
        self.on_slider_moved(self.category, val)

    def _handle_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        self.category.target_percentage = value
        self.slider.setValue(int(round(value * 10)))
        self._updating = False
        self.on_slider_moved(self.category, value)

    def _handle_delete_clicked(self) -> None:
        self.on_delete(self.category)

    def _open_files_dialog(self) -> None:
        dlg = CategoryFilesDialog(self.category, self.candidate_roots, self)
        dlg.exec()
        self.update_disk_stats_badge()

    def update_disk_stats_badge(self) -> None:
        if self.category.files_count > 0:
            t_str = format_token_count(self.category.disk_tokens)
            self.disk_badge.setText(f"Disk: {t_str}")
            if self.category.disk_tokens >= self.category.tokens_estimated:
                self.disk_badge.setStyleSheet(
                    "background-color: #064e3b; color: #34d399; padding: 4px 6px; border-radius: 4px; font-size: 11px; border: 1px solid #059669; font-weight: bold;"
                )
                self.disk_badge.setToolTip(f"Ready: {self.category.disk_tokens:,} tokens on disk across {self.category.files_count} file(s) (>= target quota)")
            else:
                self.disk_badge.setStyleSheet(
                    "background-color: #451a03; color: #fbbf24; padding: 4px 6px; border-radius: 4px; font-size: 11px; border: 1px solid #b45309;"
                )
                self.disk_badge.setToolTip(f"Partial: {self.category.disk_tokens:,} tokens on disk ({self.category.files_count} files). Need more data for full quota.")
        else:
            self.disk_badge.setText("Disk: -")
            self.disk_badge.setStyleSheet(
                "background-color: #202028; color: #9ca3af; padding: 4px 6px; border-radius: 4px; font-size: 11px; border: 1px solid #333342;"
            )
            self.disk_badge.setToolTip("No files scanned on disk yet. Click '↻ Refresh Disk Data' or double-click folder badge.")

    def refresh_from_category(self) -> None:
        """Update controls when category percentage or tokens change externally."""
        self._updating = True
        self.enabled_check.setChecked(self.category.enabled)
        self.lock_btn.setChecked(self.category.locked)
        self._update_lock_btn_style()
        self.spin.setValue(self.category.target_percentage)
        self.slider.setValue(int(round(self.category.target_percentage * 10)))
        self.token_badge.setText(f"Quota: {format_token_count(self.category.tokens_estimated)}")
        self.token_badge.setToolTip(f"Projected Target Quota: {self.category.tokens_estimated:,} tokens ({self.category.target_percentage:.1f}% of budget)")
        self.update_disk_stats_badge()
        self._apply_enabled_dimming()
        self._updating = False


class AddCategoryDialog(QDialog):
    """Clean modal dialog to add a new category to the recipe."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Recipe Category")
        self.setFixedWidth(460)
        self.setStyleSheet(
            "QDialog { background-color: #18181b; color: #ececf1; }"
            "QLabel { color: #d4d4d8; font-size: 12px; }"
            "QLineEdit, QDoubleSpinBox { background-color: #27272a; border: 1px solid #3f3f46; color: white; padding: 6px; border-radius: 4px; }"
            "QPushButton { background-color: #3f3f46; color: white; border: none; padding: 6px 14px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #52525b; }"
        )
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Add New Category to Recipe")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #f59e0b;")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(10)

        # Name
        grid.addWidget(QLabel("Category Name:"), 0, 0)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Autonomous Driving & Robotics")
        self.name_edit.textChanged.connect(self._auto_slug)
        grid.addWidget(self.name_edit, 0, 1)

        # Slug
        grid.addWidget(QLabel("Category Slug:"), 1, 0)
        self.slug_edit = QLineEdit()
        self.slug_edit.setPlaceholderText("e.g. autonomous_driving")
        grid.addWidget(self.slug_edit, 1, 1)

        # Target Percentage
        grid.addWidget(QLabel("Target Percentage:"), 2, 0)
        self.pct_spin = QDoubleSpinBox()
        self.pct_spin.setRange(0.1, 100.0)
        self.pct_spin.setValue(5.0)
        self.pct_spin.setSuffix(" %")
        grid.addWidget(self.pct_spin, 2, 1)

        # Source Path
        grid.addWidget(QLabel("Source Folder:"), 3, 0)
        path_box = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Optional folder path or subfolder name")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_folder)
        path_box.addWidget(self.path_edit)
        path_box.addWidget(browse_btn)
        grid.addLayout(path_box, 3, 1)

        layout.addLayout(grid)

        # Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        add_btn = QPushButton("Add Category")
        add_btn.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold;")
        add_btn.clicked.connect(self._validate_and_accept)
        btn_box.addWidget(cancel_btn)
        btn_box.addWidget(add_btn)
        layout.addLayout(btn_box)

    def _auto_slug(self, text: str) -> None:
        if not self.slug_edit.isModified():
            import re
            slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
            self.slug_edit.setText(slug)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Category Source Folder")
        if folder:
            self.path_edit.setText(folder)

    def _validate_and_accept(self) -> None:
        name = self.name_edit.text().strip()
        slug = self.slug_edit.text().strip()
        if not name or not slug:
            QMessageBox.warning(self, "Missing Information", "Please provide a category name and slug.")
            return
        self.accept()

    def get_category(self) -> RecipeCategory:
        path = self.path_edit.text().strip()
        source_paths = [path] if path else [self.slug_edit.text().strip()]
        return RecipeCategory(
            name=self.name_edit.text().strip(),
            slug=self.slug_edit.text().strip(),
            target_percentage=self.pct_spin.value(),
            enabled=True,
            locked=False,
            source_paths=source_paths,
            is_builtin=False,
        )


def build_dataset_recipe_tab(window: Any) -> QWidget:
    """Build the dynamic Dataset Recipe Matrix screen.

    Allows fine-grained, dynamic control over dataset category proportions,
    adding custom categories, locking ratios, auto-normalizing, inspecting disk files,
    and saving recipes per project.
    """
    page = QWidget()
    page.setObjectName("DatasetRecipePage")
    root_layout = QVBoxLayout(page)
    root_layout.setContentsMargins(14, 8, 14, 8)
    root_layout.setSpacing(6)

    # Initialize current recipe on window if not present
    if not hasattr(window, "active_dataset_recipe") or window.active_dataset_recipe is None:
        window.active_dataset_recipe = get_default_recipe()

    recipe: DatasetRecipe = window.active_dataset_recipe
    recipe.recalculate_token_projections()

    # =========================================================================
    # Header & Metric Chips
    # =========================================================================
    header_box = QHBoxLayout()
    header_box.setContentsMargins(0, 0, 0, 4)
    title_box = QVBoxLayout()
    title_box.setSpacing(2)

    title_label = QLabel("DATASET RECIPE MATRIX")
    title_label.setObjectName("PageTitle")
    title_label.setStyleSheet("font-size: 20px; font-weight: 900; letter-spacing: 0.5px;")
    subtitle_label = QLabel("Dynamic Multi-Discipline Category Proportions, File Inspection & Token Allocation")
    subtitle_label.setObjectName("SubTitle")
    title_box.addWidget(title_label)
    title_box.addWidget(subtitle_label)
    header_box.addLayout(title_box)
    header_box.addStretch(1)

    window.recipe_total_categories_chip = window._metric_chip("Total Categories: 11", "Total active categories in the recipe.")
    window.recipe_total_categories_chip.setObjectName("RecipeChip")
    window.recipe_total_percentage_chip = window._metric_chip("Target Mixture: 100%", "Sum of active category percentages.")
    window.recipe_total_percentage_chip.setObjectName("RecipeChipAmber")
    window.recipe_projected_tokens_chip = window._metric_chip("Tokens Projected: 250M", "Estimated total tokens to be sampled.")
    window.recipe_projected_tokens_chip.setObjectName("RecipeChipAmber")
    window.recipe_balance_chip = window._metric_chip("Status: Balanced", "Validation status of active recipe mixture.")
    window.recipe_balance_chip.setObjectName("RecipeChipGreen")

    header_box.addWidget(window.recipe_total_categories_chip)
    header_box.addWidget(window.recipe_total_percentage_chip)
    header_box.addWidget(window.recipe_projected_tokens_chip)
    header_box.addWidget(window.recipe_balance_chip)
    root_layout.addLayout(header_box)

    # =========================================================================
    # Two-Column Body: Presets Left Panel + Matrix Distribution Right Panel
    # =========================================================================
    body_layout = QHBoxLayout()
    body_layout.setSpacing(14)
    body_layout.setContentsMargins(0, 0, 0, 0)

    # 1. Left Card: RECIPE PRESETS
    left_card = QFrame()
    left_card.setObjectName("Card")
    left_card.setFixedWidth(250)
    left_layout = QVBoxLayout(left_card)
    left_layout.setContentsMargins(14, 14, 14, 14)
    left_layout.setSpacing(10)

    presets_heading = QLabel("RECIPE PRESETS")
    presets_heading.setObjectName("SectionLabel")
    presets_heading.setStyleSheet("font-size: 11px; font-weight: 800; letter-spacing: 0.8px;")
    left_layout.addWidget(presets_heading)

    # Internal combo maintained for signal routing and project state tracking (not shown in layout)
    window.recipe_preset_combo = QComboBox()
    window.recipe_preset_combo.addItems([
        "Default 11-Pillar Frontier Base",
        "Code & Systems Heavy",
        "STEM & Formal Reasoning",
        "Balanced Tiny LLM",
        "Custom Mixture",
    ])

    # Preset quick list widget (acts as direct recipe selector buttons)
    preset_list = QTreeWidget()
    preset_list.setObjectName("PresetTree")
    preset_list.setHeaderHidden(True)
    preset_list.setRootIsDecorated(False)
    preset_list.setMinimumHeight(150)
    preset_names = [
        "● Default 11-Pillar Frontier Base",
        "● Code Heavy",
        "● Math & Reasoning",
        "● Balanced Tiny LLM",
        "● Custom Mixture",
    ]
    for p_name in preset_names:
        preset_list.addTopLevelItem(QTreeWidgetItem([p_name]))
    if preset_list.topLevelItemCount() > 0:
        preset_list.setCurrentItem(preset_list.topLevelItem(0))

    def on_tree_preset_clicked(item: QTreeWidgetItem, _col: int) -> None:
        idx = preset_list.indexOfTopLevelItem(item)
        if 0 <= idx < window.recipe_preset_combo.count():
            window.recipe_preset_combo.setCurrentIndex(idx)

    preset_list.itemClicked.connect(on_tree_preset_clicked)
    left_layout.addWidget(preset_list)

    tb_label = QLabel("Target Token Budget:")
    tb_label.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
    left_layout.addWidget(tb_label)

    window.recipe_target_tokens_spin = QSpinBox()
    window.recipe_target_tokens_spin.setRange(1, 100_000)
    window.recipe_target_tokens_spin.setSingleStep(10)
    window.recipe_target_tokens_spin.setSuffix(" M")
    window.recipe_target_tokens_spin.setValue(int(recipe.total_target_tokens // 1_000_000))
    left_layout.addWidget(window.recipe_target_tokens_spin)

    window.recipe_live_autobalance_check = QCheckBox("Live Auto-Balance (100%)")
    window.recipe_live_autobalance_check.setChecked(True)
    left_layout.addWidget(window.recipe_live_autobalance_check)

    # Action Buttons inside left panel
    window.recipe_save_btn = QPushButton("Save Recipe to Project")
    window.recipe_save_btn.setStyleSheet(
        "QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f59e0b, stop:1 #d97706); color: #000000; font-weight: 800; border-radius: 8px; height: 38px; border: none; font-size: 12px; }"
        "QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #fbbf24, stop:1 #f59e0b); }"
    )
    left_layout.addWidget(window.recipe_save_btn)

    window.recipe_export_btn = QPushButton("Export Recipe")
    window.recipe_export_btn.setObjectName("SecondaryAction")
    left_layout.addWidget(window.recipe_export_btn)

    window.recipe_import_btn = QPushButton("Import Recipe...")
    window.recipe_import_btn.setObjectName("SecondaryAction")
    left_layout.addWidget(window.recipe_import_btn)

    window.recipe_scan_disk_btn = QPushButton("↻ Refresh Disk Data")
    window.recipe_scan_disk_btn.setStyleSheet(
        "QPushButton { background: #14233a; color: #38bdf8; border: 1px solid #0284c7; border-radius: 8px; height: 34px; font-size: 12px; font-weight: 600; }"
        "QPushButton:hover { background: #0369a1; color: #ffffff; }"
    )
    left_layout.addWidget(window.recipe_scan_disk_btn)

    left_layout.addStretch(1)

    window.recipe_add_cat_btn = QPushButton("+ Add Custom Category")
    window.recipe_add_cat_btn.setObjectName("SecondaryAction")
    left_layout.addWidget(window.recipe_add_cat_btn)

    body_layout.addWidget(left_card)

    # 2. Right Card: DISTRIBUTION & CATEGORY SLIDERS
    right_card = QFrame()
    right_card.setObjectName("Card")
    right_layout = QVBoxLayout(right_card)
    right_layout.setContentsMargins(16, 14, 16, 14)
    right_layout.setSpacing(10)

    # Continuous Distribution Bar row
    dist_row = QHBoxLayout()
    dist_row.setSpacing(12)
    window.recipe_dist_bar = RecipeDistributionBar()
    window.recipe_dist_bar.set_recipe(recipe)
    dist_row.addWidget(window.recipe_dist_bar, 1)

    dist_pct_label = QLabel("100%")
    dist_pct_label.setStyleSheet("color: #cbd5e1; font-weight: 700; font-size: 12px;")
    dist_row.addWidget(dist_pct_label)
    right_layout.addLayout(dist_row)

    # Scrollable rows list
    list_scroll = QScrollArea()
    list_scroll.setWidgetResizable(True)
    list_scroll.setStyleSheet(
        "QScrollArea, QScrollArea::viewport, QWidget { background: transparent; border: none; }"
    )
    list_scroll.viewport().setStyleSheet("background: transparent; border: none;")
    list_container = QWidget()
    list_container.setStyleSheet("background: transparent; border: none;")
    list_layout = QVBoxLayout(list_container)
    list_layout.setContentsMargins(0, 4, 0, 4)
    list_layout.setSpacing(2)
    list_scroll.setWidget(list_container)
    right_layout.addWidget(list_scroll, 1)

    # Bottom Actions inside right card
    card_bottom_row = QHBoxLayout()
    card_bottom_row.setContentsMargins(0, 6, 0, 0)
    card_bottom_row.setSpacing(12)
    card_bottom_row.addStretch(1)

    window.recipe_normalize_btn = QPushButton("Auto-Normalize to 100%")
    window.recipe_normalize_btn.setObjectName("SecondaryAction")
    card_bottom_row.addWidget(window.recipe_normalize_btn)

    apply_ingestion_btn = QPushButton("Apply Recipe to Ingestion Matrix ➔")
    apply_ingestion_btn.setStyleSheet(
        "QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f59e0b, stop:1 #d97706); color: #000000; font-weight: 800; border-radius: 8px; padding: 8px 24px; font-size: 13px; border: none; }"
        "QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #fbbf24, stop:1 #f59e0b); }"
    )
    card_bottom_row.addWidget(apply_ingestion_btn)
    right_layout.addLayout(card_bottom_row)

    body_layout.addWidget(right_card, 1)
    root_layout.addLayout(body_layout, 1)

    window._recipe_row_widgets = []

    # =========================================================================
    # Logic & Event Handlers
    # =========================================================================
    def update_metrics_and_bar() -> None:
        rec: DatasetRecipe = window.active_dataset_recipe
        rec.recalculate_token_projections(window.recipe_target_tokens_spin.value() * 1_000_000)

        enabled_cats = [c for c in rec.categories if c.enabled]
        tot_pct = rec.total_percentage()

        window.recipe_total_categories_chip.setText(f"Categories: {len(enabled_cats)} Active")
        window.recipe_total_percentage_chip.setText(f"Total: {tot_pct:.1f}%")

        is_light = getattr(window, "theme_name", "") == "system" or is_system_theme()
        if rec.is_balanced():
            if is_light:
                window.recipe_total_percentage_chip.setStyleSheet("background-color: #ecfdf5; color: #047857; font-weight: bold; padding: 4px 8px; border-radius: 4px; border: 1px solid #10b981;")
                window.recipe_balance_chip.setText("Status: Balanced (100.0%)")
                window.recipe_balance_chip.setStyleSheet("background-color: #ecfdf5; color: #047857; font-weight: bold; padding: 4px 8px; border-radius: 4px; border: 1px solid #10b981;")
            else:
                window.recipe_total_percentage_chip.setStyleSheet("background-color: #064e3b; color: #34d399; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
                window.recipe_balance_chip.setText("Status: Balanced (100.0%)")
                window.recipe_balance_chip.setStyleSheet("background-color: #064e3b; color: #34d399; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
        else:
            if is_light:
                window.recipe_total_percentage_chip.setStyleSheet("background-color: #fef3c7; color: #b45309; font-weight: bold; padding: 4px 8px; border-radius: 4px; border: 1px solid #f59e0b;")
                window.recipe_balance_chip.setText(f"Status: Unbalanced ({tot_pct:.1f}%)")
                window.recipe_balance_chip.setStyleSheet("background-color: #fef3c7; color: #b45309; font-weight: bold; padding: 4px 8px; border-radius: 4px; border: 1px solid #f59e0b;")
            else:
                window.recipe_total_percentage_chip.setStyleSheet("background-color: #451a03; color: #fbbf24; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
                window.recipe_balance_chip.setText(f"Status: Unbalanced ({tot_pct:.1f}%)")
                window.recipe_balance_chip.setStyleSheet("background-color: #451a03; color: #fbbf24; font-weight: bold; padding: 4px 8px; border-radius: 4px;")

        tot_tok = sum(c.tokens_estimated for c in enabled_cats)
        window.recipe_projected_tokens_chip.setText(f"Projected: {format_token_count(tot_tok)} Tokens")

        window.recipe_dist_bar.set_recipe(rec)

        # Update dataset preparation tab banner if present
        if hasattr(window, "update_dataset_tab_recipe_banner"):
            window.update_dataset_tab_recipe_banner()

    def on_row_slider_moved(cat: RecipeCategory, new_val: float) -> None:
        rec: DatasetRecipe = window.active_dataset_recipe
        if window.recipe_live_autobalance_check.isChecked() and not cat.locked:
            rec.rebalance_on_category_change(cat.slug, new_val)
            for rw in window._recipe_row_widgets:
                rw.refresh_from_category()
        else:
            rec.recalculate_token_projections()
            for rw in window._recipe_row_widgets:
                if rw.category == cat:
                    rw.token_badge.setText(f"Quota: {format_token_count(cat.tokens_estimated)}")

        update_metrics_and_bar()

        # Switch preset selection to Custom if user adjusted percentages
        if preset_list.topLevelItemCount() > 4:
            preset_list.setCurrentItem(preset_list.topLevelItem(4))
        if window.recipe_preset_combo.currentIndex() != 4:
            window.recipe_preset_combo.blockSignals(True)
            window.recipe_preset_combo.setCurrentIndex(4)  # Custom Mixture
            window.recipe_preset_combo.blockSignals(False)

    def on_row_changed() -> None:
        update_metrics_and_bar()
        for rw in window._recipe_row_widgets:
            rw.refresh_from_category()
        if preset_list.topLevelItemCount() > 4:
            preset_list.setCurrentItem(preset_list.topLevelItem(4))
        if window.recipe_preset_combo.currentIndex() != 4:
            window.recipe_preset_combo.blockSignals(True)
            window.recipe_preset_combo.setCurrentIndex(4)  # Custom Mixture
            window.recipe_preset_combo.blockSignals(False)

    def on_row_delete(cat: RecipeCategory) -> None:
        rec: DatasetRecipe = window.active_dataset_recipe
        if len(rec.categories) <= 1:
            QMessageBox.warning(page, "Cannot Delete", "A recipe must have at least one category.")
            return
        rec.categories = [c for c in rec.categories if c != cat]
        rebuild_category_rows()
        on_normalize_clicked()

    def scan_all_disk_stats() -> None:
        rec: DatasetRecipe = window.active_dataset_recipe
        candidate_roots = get_candidate_roots(window)
        for cat in rec.categories:
            scan_category_disk_stats(cat, candidate_roots)
        for rw in window._recipe_row_widgets:
            rw.update_disk_stats_badge()

    def rebuild_category_rows() -> None:
        while list_layout.count():
            item = list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        window._recipe_row_widgets.clear()

        rec: DatasetRecipe = window.active_dataset_recipe
        candidate_roots = get_candidate_roots(window)

        for idx, cat in enumerate(rec.categories):
            color = CATEGORY_PALETTE[idx % len(CATEGORY_PALETTE)]
            # Quick scan of disk stats if not populated
            if cat.files_count == 0:
                scan_category_disk_stats(cat, candidate_roots)
            row_w = CategoryRowWidget(
                cat,
                color,
                candidate_roots,
                on_row_changed,
                on_row_slider_moved,
                on_row_delete,
            )
            list_layout.addWidget(row_w)
            window._recipe_row_widgets.append(row_w)

        list_layout.addStretch(1)
        update_metrics_and_bar()

    def on_preset_selected(idx: int) -> None:
        preset_keys = ["frontier_11_pillar", "code_heavy", "stem_reasoning", "balanced_tiny"]
        if idx < len(preset_keys):
            key = preset_keys[idx]
            preset = DEFAULT_RECIPE_PRESETS.get(key)
            if preset:
                window.active_dataset_recipe = DatasetRecipe.from_dict(preset.to_dict())
                window.recipe_target_tokens_spin.setValue(int(window.active_dataset_recipe.total_target_tokens // 1_000_000))
                rebuild_category_rows()

    def on_normalize_clicked() -> None:
        rec: DatasetRecipe = window.active_dataset_recipe
        rec.normalize_to_100()
        update_metrics_and_bar()
        for rw in window._recipe_row_widgets:
            rw.refresh_from_category()

    def on_add_category_clicked() -> None:
        dlg = AddCategoryDialog(page)
        if dlg.exec():
            new_cat = dlg.get_category()
            rec: DatasetRecipe = window.active_dataset_recipe
            rec.categories.append(new_cat)
            rebuild_category_rows()
            on_normalize_clicked()

    def on_save_to_project_clicked() -> None:
        project_dir = getattr(window, "current_project_dir", None)
        if not project_dir:
            if hasattr(window, "project_dir") and window.project_dir:
                project_dir = Path(window.project_dir)
            elif hasattr(window, "current_project_file") and window.current_project_file:
                project_dir = window.current_project_file.parent

        if not project_dir or not Path(project_dir).is_dir():
            folder = QFileDialog.getExistingDirectory(page, "Select Project Folder to Save Recipe")
            if not folder:
                return
            project_dir = Path(folder)

        recipe_file = Path(project_dir) / "recipe.json"
        rec: DatasetRecipe = window.active_dataset_recipe
        rec.save_to_file(recipe_file)
        QMessageBox.information(
            page,
            "Recipe Saved",
            f"Successfully saved recipe '{rec.name}' to:\n{recipe_file}",
        )
        if hasattr(window, "save_project"):
            window.save_project()

    def on_export_clicked() -> None:
        file_path, _ = QFileDialog.getSaveFileName(page, "Export Recipe to File", "recipe.json", "JSON Files (*.json)")
        if file_path:
            rec: DatasetRecipe = window.active_dataset_recipe
            rec.save_to_file(Path(file_path))
            QMessageBox.information(page, "Recipe Exported", f"Saved recipe to {file_path}")

    def on_import_clicked() -> None:
        file_path, _ = QFileDialog.getOpenFileName(page, "Import Recipe File", "", "JSON Files (*.json)")
        if file_path:
            loaded = DatasetRecipe.load_from_file(Path(file_path))
            if loaded:
                window.active_dataset_recipe = loaded
                window.recipe_target_tokens_spin.setValue(int(loaded.total_target_tokens // 1_000_000))
                rebuild_category_rows()
                QMessageBox.information(page, "Recipe Loaded", f"Loaded recipe '{loaded.name}' with {len(loaded.categories)} categories.")
            else:
                QMessageBox.critical(page, "Load Failed", f"Could not load valid recipe from {file_path}")

    def on_target_tokens_changed(value: int) -> None:
        window.active_dataset_recipe.total_target_tokens = value * 1_000_000
        update_metrics_and_bar()
        for rw in window._recipe_row_widgets:
            rw.refresh_from_category()

    # Connect signals
    window.recipe_preset_combo.currentIndexChanged.connect(on_preset_selected)
    window.recipe_normalize_btn.clicked.connect(on_normalize_clicked)
    window.recipe_scan_disk_btn.clicked.connect(scan_all_disk_stats)
    window.recipe_add_cat_btn.clicked.connect(on_add_category_clicked)
    window.recipe_save_btn.clicked.connect(on_save_to_project_clicked)
    window.recipe_export_btn.clicked.connect(on_export_clicked)
    window.recipe_import_btn.clicked.connect(on_import_clicked)
    window.recipe_target_tokens_spin.valueChanged.connect(on_target_tokens_changed)

    # Populate rows
    rebuild_category_rows()

    # Expose helper methods on window
    window.rebuild_recipe_tab = rebuild_category_rows
    window.get_active_recipe = lambda: window.active_dataset_recipe

    def on_apply_to_ingestion() -> None:
        rec: DatasetRecipe = window.active_dataset_recipe
        if not rec.is_balanced(1.0):
            res = QMessageBox.question(
                page,
                "Recipe Unbalanced",
                f"Active recipe total mixture is {rec.total_percentage():.1f}%, not 100.0%.\nWould you like to Auto-Normalize before applying?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if res == QMessageBox.Yes:
                on_normalize_clicked()
            elif res == QMessageBox.Cancel:
                return

        weights = {c.slug: c.target_percentage for c in rec.categories if c.enabled}
        if hasattr(window, "_set_mixture_weights"):
            window._set_mixture_weights(weights)

        # Update dataset banner
        if hasattr(window, "update_dataset_tab_recipe_banner"):
            window.update_dataset_tab_recipe_banner()

        # Switch to Ingestion Matrix tab (page index 2)
        if hasattr(window, "_switch_page"):
            window._switch_page(2)

    apply_ingestion_btn.clicked.connect(on_apply_to_ingestion)

    return page
