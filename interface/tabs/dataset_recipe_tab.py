from __future__ import annotations

import json
import logging
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
    QVBoxLayout,
    QWidget,
)

from engine.dataset_recipe import (
    DEFAULT_RECIPE_PRESETS,
    DatasetRecipe,
    RecipeCategory,
    create_frontier_11_pillar_recipe,
    get_default_recipe,
)

LOGGER = logging.getLogger(__name__)

# Curated palette of modern, vibrant colors for category visualization
CATEGORY_PALETTE = [
    "#3b82f6",  # Systems Code (Vibrant Blue)
    "#10b981",  # STEM / Math (Emerald Green)
    "#f59e0b",  # Algorithms (Electric Amber)
    "#8b5cf6",  # Hardware RTL (Purple / Violet)
    "#ec4899",  # Cybersecurity (Hot Pink)
    "#06b6d4",  # Biomedicine (Cyan)
    "#14b8a6",  # Finance (Teal)
    "#f97316",  # Legal Reasoning (Orange)
    "#a855f7",  # Multilingual (Deep Violet)
    "#64748b",  # Encyclopedic (Slate Gray)
    "#e11d48",  # Rose
    "#84cc16",  # Lime Green
    "#0284c7",  # Sky Blue
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


class RecipeDistributionBar(QWidget):
    """Custom stacked multi-color horizontal bar visualizing category proportions."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.recipe: Optional[DatasetRecipe] = None
        self.setFixedHeight(30)
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

        # Render each segment
        current_x = 0.0
        # If total is 100%, each percentage point is width / 100.
        # If total > 100 or < 100, we represent out of 100.0 so gaps or overflows are visually clear
        scale_denom = max(100.0, total_pct)

        for idx, cat in enumerate(enabled_cats):
            color_hex = CATEGORY_PALETTE[idx % len(CATEGORY_PALETTE)]
            seg_width = (cat.target_percentage / scale_denom) * width
            seg_rect = QRectF(current_x, 0, seg_width, height)
            self._segments.append((seg_rect, cat, color_hex))

            painter.fillRect(seg_rect, QColor(color_hex))

            # Segment border / divider
            if seg_width > 3:
                painter.setPen(QPen(QColor(0, 0, 0, 80), 1.0))
                painter.drawLine(int(current_x + seg_width), 0, int(current_x + seg_width), int(height))

            # Optional label inside segment if wide enough
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
        for rect, cat, _ in self._segments:
            if rect.contains(pos):
                self.setToolTip(
                    f"<b>{cat.name}</b> ({cat.slug})<br/>"
                    f"Target: <b>{cat.target_percentage:.1f}%</b><br/>"
                    f"Tokens: <b>{cat.tokens_estimated:,}</b> ({format_token_count(cat.tokens_estimated)})"
                )
                return
        if self.recipe:
            tot = self.recipe.total_percentage()
            self.setToolTip(f"Total Mixture: {tot:.1f}% / 100.0%")
        else:
            self.setToolTip("")


class CategoryRowWidget(QFrame):
    """Dynamic row panel representing one dataset category."""

    def __init__(
        self,
        category: RecipeCategory,
        color_hex: str,
        on_change_callback: Any,
        on_delete_callback: Any,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.category = category
        self.color_hex = color_hex
        self.on_change = on_change_callback
        self.on_delete = on_delete_callback
        self._updating = False

        self.setObjectName("CategoryRow")
        self.setStyleSheet(
            "#CategoryRow {"
            "  background-color: #1a1a20;"
            "  border: 1px solid #2b2b36;"
            "  border-radius: 6px;"
            "  margin-bottom: 4px;"
            "}"
            "#CategoryRow:hover {"
            "  border: 1px solid #3d3d4e;"
            "}"
        )
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # 1. Enable Checkbox
        self.enabled_check = QCheckBox()
        self.enabled_check.setChecked(self.category.enabled)
        self.enabled_check.setToolTip("Enable or disable this category from the mixture")
        self.enabled_check.toggled.connect(self._handle_enabled_toggled)
        layout.addWidget(self.enabled_check)

        # 2. Color Swatch
        self.swatch = QFrame()
        self.swatch.setFixedSize(14, 14)
        self.swatch.setStyleSheet(f"background-color: {self.color_hex}; border-radius: 3px;")
        layout.addWidget(self.swatch)

        # 3. Category Name & Slug
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        self.name_label = QLabel(self.category.name)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #ececf1;")
        self.slug_label = QLabel(f"slug: {self.category.slug}")
        self.slug_label.setStyleSheet("font-size: 11px; color: #8e8ea0;")
        info_layout.addWidget(self.name_label)
        info_layout.addWidget(self.slug_label)

        info_container = QWidget()
        info_container.setLayout(info_layout)
        info_container.setMinimumWidth(210)
        layout.addWidget(info_container)

        # 4. Folder / Source paths badge
        paths_str = ", ".join(self.category.source_paths) if self.category.source_paths else self.category.slug
        self.paths_badge = QLabel(f"📁 {paths_str}")
        self.paths_badge.setStyleSheet(
            "background-color: #262630; color: #9a9ab0; padding: 4px 8px; border-radius: 4px; font-size: 11px;"
        )
        self.paths_badge.setToolTip(f"Linked source folders: {paths_str}")
        self.paths_badge.setMaximumWidth(200)
        layout.addWidget(self.paths_badge)

        # 5. Continuous Slider (0 - 1000 = 0.0% to 100.0%)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(int(round(self.category.target_percentage * 10)))
        self.slider.setMinimumWidth(160)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.slider.valueChanged.connect(self._handle_slider_changed)
        layout.addWidget(self.slider, 1)

        # 6. Percentage Double Spinbox (0.00% to 100.00%)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 100.0)
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.5)
        self.spin.setSuffix(" %")
        self.spin.setValue(self.category.target_percentage)
        self.spin.setFixedWidth(90)
        self.spin.valueChanged.connect(self._handle_spin_changed)
        layout.addWidget(self.spin)

        # 7. Projected Tokens Chip
        self.token_badge = QLabel(f"{format_token_count(self.category.tokens_estimated)} tokens")
        self.token_badge.setAlignment(Qt.AlignCenter)
        self.token_badge.setFixedWidth(100)
        self.token_badge.setStyleSheet(
            "background-color: #23232c; color: #e5a93c; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 11px;"
        )
        self.token_badge.setToolTip(f"{self.category.tokens_estimated:,} projected tokens")
        layout.addWidget(self.token_badge)

        # 8. Ratio Lock Toggle Button
        self.lock_btn = QPushButton("🔒 Locked" if self.category.locked else "🔓 Unlocked")
        self.lock_btn.setCheckable(True)
        self.lock_btn.setChecked(self.category.locked)
        self.lock_btn.setFixedWidth(95)
        self._update_lock_btn_style()
        self.lock_btn.toggled.connect(self._handle_lock_toggled)
        layout.addWidget(self.lock_btn)

        # 9. Delete Button
        self.delete_btn = QPushButton("✕")
        self.delete_btn.setFixedSize(28, 28)
        self.delete_btn.setStyleSheet(
            "QPushButton { background-color: transparent; color: #71717a; border: 1px solid #3f3f46; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #dc2626; color: white; border: 1px solid #dc2626; }"
        )
        self.delete_btn.setToolTip("Remove category from recipe")
        self.delete_btn.clicked.connect(self._handle_delete_clicked)
        layout.addWidget(self.delete_btn)

        self._apply_enabled_dimming()

    def _update_lock_btn_style(self) -> None:
        if self.category.locked:
            self.lock_btn.setText("🔒 Locked")
            self.lock_btn.setStyleSheet(
                "background-color: #312e81; color: #a5b4fc; border: 1px solid #4338ca; border-radius: 4px; font-size: 11px; padding: 4px 6px;"
            )
            self.lock_btn.setToolTip("Ratio is LOCKED. Auto-normalize will not alter this category's percentage.")
        else:
            self.lock_btn.setText("🔓 Unlocked")
            self.lock_btn.setStyleSheet(
                "background-color: #202028; color: #9ca3af; border: 1px solid #374151; border-radius: 4px; font-size: 11px; padding: 4px 6px;"
            )
            self.lock_btn.setToolTip("Ratio is UNLOCKED. Auto-normalize will adjust this category to hit 100%.")

    def _apply_enabled_dimming(self) -> None:
        is_on = self.category.enabled
        self.slider.setEnabled(is_on)
        self.spin.setEnabled(is_on)
        self.lock_btn.setEnabled(is_on)
        opacity = "1.0" if is_on else "0.4"
        self.name_label.setStyleSheet(f"font-weight: bold; font-size: 13px; color: #ececf1; opacity: {opacity};")
        self.token_badge.setEnabled(is_on)

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
        self.on_change()

    def _handle_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        self.category.target_percentage = value
        self.slider.setValue(int(round(value * 10)))
        self._updating = False
        self.on_change()

    def _handle_delete_clicked(self) -> None:
        self.on_delete(self.category)

    def refresh_from_category(self) -> None:
        """Update controls when category percentage or tokens change externally (e.g. normalization)."""
        self._updating = True
        self.enabled_check.setChecked(self.category.enabled)
        self.lock_btn.setChecked(self.category.locked)
        self._update_lock_btn_style()
        self.spin.setValue(self.category.target_percentage)
        self.slider.setValue(int(round(self.category.target_percentage * 10)))
        self.token_badge.setText(f"{format_token_count(self.category.tokens_estimated)} tokens")
        self.token_badge.setToolTip(f"{self.category.tokens_estimated:,} projected tokens")
        self._apply_enabled_dimming()
        self._updating = False


class AddCategoryDialog(QDialog):
    """Clean modal dialog to add a new category to the recipe."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
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
        self.name_edit.setPlaceholderText("e.g. Autonomous Driving & Vision")
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
        self.path_edit.setPlaceholderText("Optional path or subfolder")
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
    adding custom categories, locking ratios, auto-normalizing, and saving recipes per project.
    """
    page = QWidget()
    page.setObjectName("DatasetRecipePage")
    root_layout = QVBoxLayout(page)
    root_layout.setContentsMargins(18, 14, 18, 14)
    root_layout.setSpacing(12)

    # Initialize current recipe on window if not present
    if not hasattr(window, "active_dataset_recipe") or window.active_dataset_recipe is None:
        window.active_dataset_recipe = get_default_recipe()

    recipe: DatasetRecipe = window.active_dataset_recipe
    recipe.recalculate_token_projections()

    # =========================================================================
    # Header & Metric Chips
    # =========================================================================
    header_box = QHBoxLayout()
    title_box = QVBoxLayout()
    title_box.setSpacing(2)

    title_label = QLabel("DATASET RECIPE MATRIX")
    title_label.setStyleSheet("font-size: 17px; font-weight: 800; letter-spacing: 0.5px; color: #f3f4f6;")
    subtitle_label = QLabel("Dynamic Multi-Discipline Category Proportions & Token Allocation")
    subtitle_label.setStyleSheet("font-size: 11px; color: #9ca3af;")
    title_box.addWidget(title_label)
    title_box.addWidget(subtitle_label)
    header_box.addLayout(title_box)
    header_box.addStretch(1)

    # Stat metric chips
    window.recipe_total_categories_chip = window._metric_chip("Categories: 10 Active", "Total active categories in the recipe.")
    window.recipe_total_percentage_chip = window._metric_chip("Total: 100.0%", "Sum of active category percentages.")
    window.recipe_projected_tokens_chip = window._metric_chip("Projected: 250.0M Tokens", "Estimated total tokens to be sampled.")
    window.recipe_balance_chip = window._metric_chip("Status: Balanced", "Validation status of active recipe mixture.")

    header_box.addWidget(window.recipe_total_categories_chip)
    header_box.addWidget(window.recipe_total_percentage_chip)
    header_box.addWidget(window.recipe_projected_tokens_chip)
    header_box.addWidget(window.recipe_balance_chip)
    root_layout.addLayout(header_box)

    # =========================================================================
    # Stacked Distribution Visualizer Bar
    # =========================================================================
    dist_container = QWidget()
    dist_layout = QVBoxLayout(dist_container)
    dist_layout.setContentsMargins(0, 0, 0, 0)
    dist_layout.setSpacing(4)

    bar_label_box = QHBoxLayout()
    bar_heading = QLabel("MIXTURE DISTRIBUTION")
    bar_heading.setStyleSheet("font-size: 10px; font-weight: bold; color: #71717a; letter-spacing: 0.5px;")
    bar_label_box.addWidget(bar_heading)
    bar_label_box.addStretch(1)
    dist_layout.addLayout(bar_label_box)

    window.recipe_dist_bar = RecipeDistributionBar()
    window.recipe_dist_bar.set_recipe(recipe)
    dist_layout.addWidget(window.recipe_dist_bar)
    root_layout.addWidget(dist_container)

    # =========================================================================
    # Action & Preset Toolbar
    # =========================================================================
    toolbar_card = QFrame()
    toolbar_card.setStyleSheet("background-color: #18181f; border: 1px solid #2a2a36; border-radius: 6px;")
    tb_layout = QHBoxLayout(toolbar_card)
    tb_layout.setContentsMargins(12, 10, 12, 10)
    tb_layout.setSpacing(10)

    # Preset ComboBox
    tb_layout.addWidget(QLabel("Recipe Preset:"))
    window.recipe_preset_combo = QComboBox()
    window.recipe_preset_combo.addItems([
        "Default 11-Pillar Frontier Base",
        "Code & Systems Heavy",
        "STEM & Formal Reasoning",
        "Balanced Tiny LLM",
        "Custom Mixture",
    ])
    window.recipe_preset_combo.setFixedWidth(230)
    tb_layout.addWidget(window.recipe_preset_combo)

    # Target Token Budget Spinbox (in Millions)
    tb_layout.addWidget(QLabel("Target Tokens:"))
    window.recipe_target_tokens_spin = QSpinBox()
    window.recipe_target_tokens_spin.setRange(1, 100_000)
    window.recipe_target_tokens_spin.setSingleStep(10)
    window.recipe_target_tokens_spin.setSuffix(" M")
    window.recipe_target_tokens_spin.setValue(int(recipe.total_target_tokens // 1_000_000))
    window.recipe_target_tokens_spin.setFixedWidth(110)
    window.recipe_target_tokens_spin.setToolTip("Total token budget in millions (e.g. 250 M = 250,000,000 tokens)")
    tb_layout.addWidget(window.recipe_target_tokens_spin)

    tb_layout.addStretch(1)

    # Normalize to 100% Button
    window.recipe_normalize_btn = QPushButton("Auto-Normalize (100%)")
    window.recipe_normalize_btn.setStyleSheet(
        "QPushButton { background-color: #2563eb; color: white; font-weight: bold; border-radius: 4px; padding: 6px 12px; }"
        "QPushButton:hover { background-color: #1d4ed8; }"
    )
    window.recipe_normalize_btn.setToolTip("Proportionally rebalance unlocked categories so the total sum is exactly 100.0%")
    tb_layout.addWidget(window.recipe_normalize_btn)

    # Add Category Button
    window.recipe_add_cat_btn = QPushButton("+ Add Category")
    window.recipe_add_cat_btn.setStyleSheet(
        "QPushButton { background-color: #10b981; color: white; font-weight: bold; border-radius: 4px; padding: 6px 12px; }"
        "QPushButton:hover { background-color: #059669; }"
    )
    window.recipe_add_cat_btn.setToolTip("Add a custom dataset category with folder path")
    tb_layout.addWidget(window.recipe_add_cat_btn)

    # Save Recipe to Project Button
    window.recipe_save_btn = QPushButton("Save to Project")
    window.recipe_save_btn.setStyleSheet(
        "QPushButton { background-color: #3b3b48; color: #f3f4f6; border: 1px solid #4b4b5c; border-radius: 4px; padding: 6px 12px; font-weight: 600; }"
        "QPushButton:hover { background-color: #4b4b5c; }"
    )
    window.recipe_save_btn.setToolTip("Save this recipe to project directory (recipe.json)")
    tb_layout.addWidget(window.recipe_save_btn)

    # Export / Import Buttons
    window.recipe_export_btn = QPushButton("Export...")
    window.recipe_export_btn.setStyleSheet(
        "QPushButton { background-color: transparent; color: #9ca3af; border: 1px solid #3b3b48; border-radius: 4px; padding: 6px 10px; }"
        "QPushButton:hover { color: white; border: 1px solid #6b7280; }"
    )
    window.recipe_import_btn = QPushButton("Import...")
    window.recipe_import_btn.setStyleSheet(
        "QPushButton { background-color: transparent; color: #9ca3af; border: 1px solid #3b3b48; border-radius: 4px; padding: 6px 10px; }"
        "QPushButton:hover { color: white; border: 1px solid #6b7280; }"
    )
    tb_layout.addWidget(window.recipe_export_btn)
    tb_layout.addWidget(window.recipe_import_btn)

    root_layout.addWidget(toolbar_card)

    # =========================================================================
    # Scrollable Category Row List
    # =========================================================================
    list_scroll = QScrollArea()
    list_scroll.setWidgetResizable(True)
    list_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

    list_container = QWidget()
    list_layout = QVBoxLayout(list_container)
    list_layout.setContentsMargins(0, 0, 0, 0)
    list_layout.setSpacing(6)
    list_scroll.setWidget(list_container)
    root_layout.addWidget(list_scroll, 1)

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

        if rec.is_balanced():
            window.recipe_total_percentage_chip.setStyleSheet("background-color: #064e3b; color: #34d399; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
            window.recipe_balance_chip.setText("Status: Balanced (100.0%)")
            window.recipe_balance_chip.setStyleSheet("background-color: #064e3b; color: #34d399; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
        else:
            window.recipe_total_percentage_chip.setStyleSheet("background-color: #451a03; color: #fbbf24; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
            window.recipe_balance_chip.setText(f"Status: Unbalanced ({tot_pct:.1f}%)")
            window.recipe_balance_chip.setStyleSheet("background-color: #451a03; color: #fbbf24; font-weight: bold; padding: 4px 8px; border-radius: 4px;")

        tot_tok = sum(c.tokens_estimated for c in enabled_cats)
        window.recipe_projected_tokens_chip.setText(f"Projected: {format_token_count(tot_tok)} Tokens")

        window.recipe_dist_bar.set_recipe(rec)

    def on_row_changed() -> None:
        update_metrics_and_bar()
        # Refresh token badges on all row widgets
        for rw in window._recipe_row_widgets:
            rw.refresh_from_category()
        # If user changed percentages manually, switch preset combo to Custom
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
        update_metrics_and_bar()

    def rebuild_category_rows() -> None:
        # Clear existing rows
        while list_layout.count():
            item = list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        window._recipe_row_widgets.clear()

        rec: DatasetRecipe = window.active_dataset_recipe
        for idx, cat in enumerate(rec.categories):
            color = CATEGORY_PALETTE[idx % len(CATEGORY_PALETTE)]
            row_w = CategoryRowWidget(cat, color, on_row_changed, on_row_delete)
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
                # Deep copy preset
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
            # Try from search_box or project paths
            if hasattr(window, "project_dir") and window.project_dir:
                project_dir = Path(window.project_dir)
            elif hasattr(window, "current_project_file") and window.current_project_file:
                project_dir = window.current_project_file.parent

        if not project_dir or not Path(project_dir).is_dir():
            # Prompt user to choose project folder or export
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

    # Bottom Actions Bar
    bottom_bar = QHBoxLayout()
    bottom_bar.setContentsMargins(4, 8, 4, 4)

    apply_ingestion_btn = QPushButton("Apply Recipe to Ingestion Matrix ➔")
    apply_ingestion_btn.setStyleSheet(
        "QPushButton { background-color: #e5a93c; color: #141414; font-weight: 800; border-radius: 6px; padding: 10px 20px; font-size: 13px; }"
        "QPushButton:hover { background-color: #f59e0b; }"
    )
    window._tip(apply_ingestion_btn, "Sync this recipe's categories and sampling percentages with the Data Ingestion Matrix.")

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

        # Propagate mixture weights to window if applicable
        weights = {c.slug: c.target_percentage for c in rec.categories if c.enabled}
        if hasattr(window, "_set_mixture_weights"):
            window._set_mixture_weights(weights)

        # Switch to Ingestion Matrix tab (page index 2)
        if hasattr(window, "_switch_page"):
            window._switch_page(2)
            if hasattr(window, "dataset_status"):
                window.dataset_status.setText(f"Recipe active: {rec.name}")

    apply_ingestion_btn.clicked.connect(on_apply_to_ingestion)
    bottom_bar.addStretch(1)
    bottom_bar.addWidget(apply_ingestion_btn)
    root_layout.addLayout(bottom_bar)

    return page
