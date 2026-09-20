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
    scan_category_disk_stats,
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
    assert cat.files_count == 0

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


def test_frontier_11_pillar_preset_has_all_11_pillars() -> None:
    recipe = create_frontier_11_pillar_recipe()
    assert len(recipe.categories) == 11

    expected_slugs = [
        "code_pretraining",
        "stem_pretraining",
        "algorithms_pretraining",
        "hardware_pretraining",
        "cybersecurity_pretraining",
        "medicine_pretraining",
        "science_pretraining",
        "finance",
        "law_pretraining",
        "multilingual_pretraining",
        "encyclopedic",
    ]
    actual_slugs = [c.slug for c in recipe.categories]
    assert actual_slugs == expected_slugs
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


def test_rebalance_on_category_change_preserves_locked() -> None:
    cats = [
        RecipeCategory("Locked A", "a", 30.0, locked=True),
        RecipeCategory("Unlocked B", "b", 35.0, locked=False),
        RecipeCategory("Unlocked C", "c", 35.0, locked=False),
    ]
    recipe = DatasetRecipe("test", "Test Recipe", total_target_tokens=100_000_000, categories=cats)
    recipe.recalculate_token_projections()

    # Initial tokens
    assert cats[0].tokens_estimated == 30_000_000
    assert cats[1].tokens_estimated == 35_000_000
    assert cats[2].tokens_estimated == 35_000_000

    # User slides category B from 35.0% to 50.0%
    recipe.rebalance_on_category_change("b", 50.0)

    # Locked A MUST NOT change percentage or tokens
    assert cats[0].target_percentage == 30.0
    assert cats[0].tokens_estimated == 30_000_000

    # B became 50%
    assert cats[1].target_percentage == 50.0
    assert cats[1].tokens_estimated == 50_000_000

    # C dynamically compensated: 100 - 30 - 50 = 20%
    assert abs(cats[2].target_percentage - 20.0) < 0.1
    assert cats[2].tokens_estimated == 20_000_000
    assert recipe.is_balanced()


def test_token_projections_locked_remain_stable() -> None:
    cats = [
        RecipeCategory("Locked A", "a", 25.0, locked=True),
        RecipeCategory("Unlocked B", "b", 25.0, locked=False),
    ]
    recipe = DatasetRecipe("test", "Test", total_target_tokens=100_000_000, categories=cats)
    recipe.recalculate_token_projections()
    assert cats[0].tokens_estimated == 25_000_000

    # Even if B is changed without normalization (unbalanced), locked A tokens don't change
    cats[1].target_percentage = 50.0
    recipe.recalculate_token_projections()
    assert cats[0].tokens_estimated == 25_000_000  # Stays exactly 25M (25% of 100M)!


def test_scan_category_disk_stats(tmp_path: Path) -> None:
    cat_dir = tmp_path / "encyclopedic"
    cat_dir.mkdir(parents=True)
    f1 = cat_dir / "part1.jsonl"
    f1.write_text("Hello world " * 100, encoding="utf-8")
    f2 = cat_dir / "part2.jsonl"
    f2.write_text("More encyclopedic knowledge " * 200, encoding="utf-8")

    cat = RecipeCategory("Encyclopedic", "encyclopedic", 5.0, source_paths=["encyclopedic"])
    stats = scan_category_disk_stats(cat, [tmp_path])

    assert stats["files_count"] == 2
    assert stats["disk_bytes"] > 0
    assert stats["disk_tokens"] > 0
    assert cat.files_count == 2
    assert cat.disk_tokens == stats["disk_tokens"]


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
    assert hasattr(window, "recipe_scan_disk_btn")
    assert hasattr(window, "recipe_live_autobalance_check")
    assert len(window._recipe_row_widgets) == 11

    # Verify live auto-balance when moving slider
    rw0 = window._recipe_row_widgets[0]  # code_pretraining (22%)
    rw1 = window._recipe_row_widgets[1]  # stem_pretraining (12%)
    orig_rw1_pct = rw1.category.target_percentage

    # Lock category 2 (algorithms)
    rw2 = window._recipe_row_widgets[2]
    rw2.lock_btn.setChecked(True)
    locked_pct = rw2.category.target_percentage

    # Move slider for category 0 from 22% to 30%
    rw0.slider.setValue(300)

    # Locked category percentage MUST remain unchanged
    assert rw2.category.target_percentage == locked_pct

    # Total must stay balanced at 100%
    assert window.active_dataset_recipe.is_balanced()
