from __future__ import annotations

from interface.theme import (
    DARK_THEME,
    SYSTEM_THEME,
    THEME_PREFERENCE_VERSION,
    _stylesheet_for_theme,
    project_theme,
)


def test_legacy_project_theme_defaults_to_dark() -> None:
    """Projects saved before theme versioning retain the established Dark UI."""
    assert project_theme({"theme": SYSTEM_THEME}) == DARK_THEME


def test_versioned_system_project_theme_is_preserved() -> None:
    """A deliberate System selection remains available after project reload."""
    assert project_theme(
        {
            "theme": SYSTEM_THEME,
            "theme_preference_version": THEME_PREFERENCE_VERSION,
        }
    ) == SYSTEM_THEME


def test_system_theme_replaces_dark_control_surfaces() -> None:
    """System colors do not retain Dark rail, header, or disabled button backgrounds."""
    stylesheet = _stylesheet_for_theme(SYSTEM_THEME)
    assert "QPushButton#NavButton {\n    background: #f2f2f2;" in stylesheet
    assert "QHeaderView::section {\n    background: #f2f2f2;" in stylesheet
    assert "QPushButton:disabled {\n    background: #e8e8e8;" in stylesheet
