from __future__ import annotations

from interface.theme import DARK_THEME, SYSTEM_THEME, THEME_PREFERENCE_VERSION, project_theme


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
