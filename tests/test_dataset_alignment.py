from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from engine.dataset_recipe import (
    create_frontier_11_pillar_recipe,
    scan_category_disk_stats,
)
from interface.tabs.dataset_plan_tab import (
    blueprint_data_root,
    dataset_category_label,
    default_data_root,
    infer_category_type,
    iter_default_data_files,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_frontier_11_pillar_recipe_proportions():
    recipe = create_frontier_11_pillar_recipe()
    assert recipe.recipe_id == "frontier_11_pillar"
    assert len(recipe.categories) == 11
    assert recipe.total_target_tokens == 250_000_000
    assert recipe.is_balanced()
    assert pytest.approx(recipe.total_percentage(), 0.001) == 100.0

    cat_map = {c.slug: c.target_percentage for c in recipe.categories}
    expected = {
        "encyclopedic": 25.0,
        "code_pretraining": 15.0,
        "algorithms_pretraining": 10.0,
        "stem_pretraining": 10.0,
        "medicine_pretraining": 10.0,
        "hardware_pretraining": 8.0,
        "cybersecurity_pretraining": 7.0,
        "finance": 5.0,
        "law_pretraining": 4.0,
        "multilingual_pretraining": 3.0,
        "fine_tune_conversation": 3.0,
    }
    assert cat_map == expected


def test_blueprint_data_root_resolution():
    root = default_data_root()
    dataset_repo = Path(r"E:\AI_Projects\dataset")
    if dataset_repo.is_dir():
        assert root == dataset_repo
        active = blueprint_data_root()
        assert active == dataset_repo


def test_iter_default_data_files_pruning():
    dataset_repo = Path(r"E:\AI_Projects\dataset")
    if not dataset_repo.is_dir():
        pytest.skip("E:\\AI_Projects\\dataset not found on this system.")

    files = iter_default_data_files(dataset_repo)
    assert len(files) > 0

    categories = {cat for _, cat in files}
    assert "curated_2b_base" in categories

    for path, _ in files:
        parts = set(path.parts)
        assert "_quarantine" not in parts
        assert ".git" not in parts
        assert ".idea" not in parts
        assert "dist" not in parts


def test_category_disk_stats_scan():
    dataset_repo = Path(r"E:\AI_Projects\dataset")
    if not dataset_repo.is_dir():
        pytest.skip("E:\\AI_Projects\\dataset not found on this system.")

    recipe = create_frontier_11_pillar_recipe()
    roots = [dataset_repo]

    for cat in recipe.categories:
        stats = scan_category_disk_stats(cat, roots)
        assert len(stats["directories"]) >= 1
        assert stats["files_count"] >= 1
        assert stats["disk_bytes"] > 0
        assert stats["disk_tokens"] > 0
        for f in stats["files"]:
            assert "_quarantine" not in f.parts


def test_category_labels_and_types():
    assert "Pre-Training Mixture" in dataset_category_label("curated_2b_base")
    icon, type_label = infer_category_type("curated_2b_base")
    assert "Base Pretraining" in type_label

    icon, type_label = infer_category_type("code_pretraining")
    assert "Code" in type_label

    icon, type_label = infer_category_type("fine_tune_thinking")
    assert "Reasoning" in type_label
