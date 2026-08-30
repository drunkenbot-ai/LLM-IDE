"""Application-wide theme selection and stylesheet loading."""

from __future__ import annotations

import json
from pathlib import Path
import re

from PySide6.QtWidgets import QApplication

SYSTEM_THEME = "system"
DARK_THEME = "dark"
DEFAULT_THEME = DARK_THEME
THEME_PREFERENCE_VERSION = 1
_THEME_PROPERTY = "application_theme"
_RECENT_PROJECTS_PATH = Path.home() / ".drunkenbot_ide" / "recent_projects.json"
_SYSTEM_COLOR_MAP = {
    "#111111": "#f5f5f5",
    "#eeeeee": "#202020",
    "#1f1f1f": "#ededed",
    "#202020": "#f2f2f2",
    "#3a3a3a": "#b8b8b8",
    "#171717": "#f0f0f0",
    "#181818": "#f7f7f7",
    "#242424": "#ffffff",
    "#3d3d3d": "#c4c4c4",
    "#141414": "#ffffff",
    "#dddddd": "#202020",
    "#f2f2f2": "#151515",
    "#d8eec2": "#29451b",
    "#20231d": "#f2f7ec",
    "#8fbf5a": "#7aa844",
    "#b6d77a": "#476d21",
    "#1b1f18": "#f7fbf3",
    "#6f8f45": "#7aa844",
    "#171a14": "#f7fbf3",
    "#f0f0f0": "#202020",
    "#555555": "#8a8a8a",
    "#6a6a6a": "#8a8a8a",
    "#1d1d1d": "#f0f0f0",
    "#1a1a1a": "#e8e8e8",
    "#777777": "#6a6a6a",
    "#333333": "#b8b8b8",
    "#4a4a4a": "#a0a0a0",
    "#d4d4d4": "#303030",
    "#2a2a2a": "#eeeeee",
    "#3a3326": "#fff7e8",
    "#c98f2e": "#b66a00",
    "#050505": "#fafafa",
    "#2b2b2b": "#c4c4c4",
}


def normalize_theme(theme: object) -> str:
    """Return a supported theme name, defaulting to the application's Dark theme."""
    return SYSTEM_THEME if theme == SYSTEM_THEME else DARK_THEME


def load_startup_theme() -> str:
    """Load the theme selected for the most recently opened project."""
    try:
        recent_projects = json.loads(_RECENT_PROJECTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_THEME
    if not isinstance(recent_projects, list):
        return DEFAULT_THEME
    for entry in recent_projects:
        if not isinstance(entry, dict):
            continue
        project_path = Path(str(entry.get("path", "")))
        try:
            project_state = json.loads(project_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(project_state, dict):
            return project_theme(project_state)
    return DEFAULT_THEME


def project_theme(project_state: object) -> str:
    """Return a project's theme, migrating preferences created before versioning."""
    if not isinstance(project_state, dict):
        return DEFAULT_THEME
    if project_state.get("theme_preference_version") != THEME_PREFERENCE_VERSION:
        return DEFAULT_THEME
    return normalize_theme(project_state.get("theme"))


def current_theme() -> str:
    """Return the QApplication theme currently in effect."""
    app = QApplication.instance()
    return normalize_theme(app.property(_THEME_PROPERTY) if app is not None else DEFAULT_THEME)


def _stylesheet_for_theme(theme: str) -> str:
    """Return the shared widget stylesheet with the requested color palette."""
    stylesheet_path = Path(__file__).with_name("styles.qss")
    dark_stylesheet = stylesheet_path.read_text(encoding="utf-8")
    if theme == DARK_THEME:
        return dark_stylesheet
    return re.sub(
        r"#[0-9a-fA-F]{6}",
        lambda match: _SYSTEM_COLOR_MAP.get(match.group(0).lower(), match.group(0)),
        dark_stylesheet,
    )


def apply_theme(theme: object) -> str:
    """Apply the selected application-wide theme and return its normalized name."""
    selected_theme = normalize_theme(theme)
    app = QApplication.instance()
    if app is None:
        return selected_theme
    app.setStyleSheet(_stylesheet_for_theme(selected_theme))
    app.setProperty(_THEME_PROPERTY, selected_theme)
    for widget in app.allWidgets():
        refresh = getattr(widget, "apply_theme", None)
        if callable(refresh):
            refresh(selected_theme)
    return selected_theme
