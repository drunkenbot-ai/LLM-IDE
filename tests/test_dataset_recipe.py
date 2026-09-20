"""Unit and integration tests for dataset recipe engine and UI."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from engine.dataset_recipe import (
    DEFAULT_RECIPE_PRESETS,
    DatasetRecipe,
    RecipeCategory,
    create_balanced_tiny_recipe,
    create_code_heavy_recipe,
    create_frontier_11_pillar_recipe,
    create_stem_reasoning_recipe,
    get_default_recipe,
)


def test_recipe_category_initialization() -> None:
    cat = RecipeCategory(
        name="Systems Code",
        slug="code_pretraining",
        target_percentage=25.0,
        enabled=True,
        locked=False,
        source_paths=["code_pretraining"],
    )
    assert cat.name == "Systems Code"
    assert cat.slug == "code_pretraining"
    assert cat.target_percentage == 25.0
    assert cat.enabled is True
    assert cat.locked is False
    assert cat.source_paths == ["code_pretraining"]

    d = cat.to_dict()
    restored = RecipeCategory.from_dict(d)
    assert restored == cat


def test_presets_validity_and_balance() -> None:
    for preset_id, recipe in DEFAULT_RECIPE_PRESETS.items():
        assert recipe.recipe_id == preset_id
        assert len(recipe.categories) > 0
        total = recipe.total_percentage()
        assert abs(total - 100.0) < 0.1, f"Preset {preset_id} total is {total}%, expected ~100%"
        assert recipe.is_balanced()


def test_normalize_to_100_unlocked() -> None:
    cats = [
        RecipeCategory("Cat A", "a", 10.0),
        RecipeCategory("Cat B", "b", 10.0),
    ]
    recipe = DatasetRecipe("test", "Test Recipe", categories=cats)
    assert recipe.total_percentage() == 20.0
    assert not recipe.is_balanced()

    recipe.normalize_to_100()
    assert recipe.is_balanced()
    assert abs(cats[0].target_percentage - 50.0) < 0.1
    assert abs(cats[1].target_percentage - 50.0) < 0.1


def test_normalize_to_100_with_locked_categories() -> None:
    cats = [
        RecipeCategory("Locked A", "a", 30.0, locked=True),
        RecipeCategory("Unlocked B", "b", 10.0, locked=False),
        RecipeCategory("Unlocked C", "c", 10.0, locked=False),
    ]
    recipe = DatasetRecipe("test", "Test Recipe", categories=cats)
    recipe.normalize_to_100()

    assert cats[0].target_percentage == 30.0  # Locked retains exact percentage
    assert recipe.is_balanced()
    # Remaining 70% split equally between B and C (35% each)
    assert abs(cats[1].target_percentage - 35.0) < 0.1
    assert abs(cats[2].target_percentage - 35.0) < 0.1


def test_normalize_with_disabled_categories() -> None:
    cats = [
        RecipeCategory("Disabled A", "a", 50.0, enabled=False),
        RecipeCategory("Active B", "b", 20.0, enabled=True),
        RecipeCategory("Active C", "c", 20.0, enabled=True),
    ]
    recipe = DatasetRecipe("test", "Test Recipe", categories=cats)
    recipe.normalize_to_100()

    assert recipe.is_balanced()
    assert cats[0].target_percentage == 50.0  # untouched because disabled
    assert abs(cats[1].target_percentage - 50.0) < 0.1
    assert abs(cats[2].target_percentage - 50.0) < 0.1


def test_token_projections() -> None:
    cats = [
        RecipeCategory("A", "a", 50.0),
        RecipeCategory("B", "b", 50.0),
        RecipeCategory("C", "c", 0.0, enabled=False),
    ]
    recipe = DatasetRecipe("test", "Test", total_target_tokens=100_000_000, categories=cats)
    recipe.recalculate_token_projections()

    assert cats[0].tokens_estimated == 50_000_000
    assert cats[1].tokens_estimated == 50_000_000
    assert cats[2].tokens_estimated == 0


def test_save_and_load_from_file(tmp_path: Path) -> None:
    recipe = get_default_recipe()
    out_file = tmp_path / "custom_recipe.json"
    recipe.save_to_file(out_file)

    assert out_file.is_file()
    loaded = DatasetRecipe.load_from_file(out_file)
    assert loaded is not None
    assert loaded.recipe_id == recipe.recipe_id
    assert len(loaded.categories) == len(recipe.categories)
    assert loaded.is_balanced()


def test_recipe_tab_ui_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication, QLabel
    from interface.tabs.dataset_recipe_tab import build_dataset_recipe_tab

    app = QApplication.instance() or QApplication([])

    class DummyWindow:
        def __init__(self):
            self.active_dataset_recipe = get_default_recipe()
            self.theme_name = "dark"
            self.current_project_dir = None

        def _metric_chip(self, text: str, tip: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setToolTip(tip)
            return lbl

        def _tip(self, w: Any, tip: str) -> None:
            w.setToolTip(tip)

    window = DummyWindow()
    page = build_dataset_recipe_tab(window)

    assert page is not None
    assert hasattr(window, "recipe_total_categories_chip")
    assert hasattr(window, "recipe_total_percentage_chip")
    assert hasattr(window, "recipe_dist_bar")
    assert hasattr(window, "recipe_normalize_btn")
    assert hasattr(window, "recipe_add_cat_btn")
    assert len(window._recipe_row_widgets) == len(window.active_dataset_recipe.categories)

    # Test auto-normalize button trigger
    window.active_dataset_recipe.categories[0].target_percentage = 99.0
    window.recipe_normalize_btn.click()
    assert window.active_dataset_recipe.is_balanced()
