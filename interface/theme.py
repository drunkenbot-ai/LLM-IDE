"""Application-wide theme selection and stylesheet loading."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QApplication

SYSTEM_THEME = "system"
DARK_THEME = "dark"
_THEME_PROPERTY = "application_theme"
_RECENT_PROJECTS_PATH = Path.home() / ".drunkenbot_ide" / "recent_projects.json"


def normalize_theme(theme: object) -> str:
    """Return a supported theme name, defaulting to the system theme."""
    return DARK_THEME if theme == DARK_THEME else SYSTEM_THEME


def load_startup_theme() -> str:
    """Load the theme selected for the most recently opened project."""
    try:
        recent_projects = json.loads(_RECENT_PROJECTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return SYSTEM_THEME
    if not isinstance(recent_projects, list):
        return SYSTEM_THEME
    for entry in recent_projects:
        if not isinstance(entry, dict):
            continue
        project_path = Path(str(entry.get("path", "")))
        try:
            project_state = json.loads(project_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(project_state, dict):
            return normalize_theme(project_state.get("theme"))
    return SYSTEM_THEME


def current_theme() -> str:
    """Return the QApplication theme currently in effect."""
    app = QApplication.instance()
    return normalize_theme(app.property(_THEME_PROPERTY) if app is not None else SYSTEM_THEME)


def apply_theme(theme: object) -> str:
    """Apply the selected application-wide theme and return its normalized name."""
    selected_theme = normalize_theme(theme)
    app = QApplication.instance()
    if app is None:
        return selected_theme
    stylesheet = ""
    if selected_theme == DARK_THEME:
        stylesheet_path = Path(__file__).with_name("styles.qss")
        stylesheet = stylesheet_path.read_text(encoding="utf-8")
    app.setStyleSheet(stylesheet)
    app.setProperty(_THEME_PROPERTY, selected_theme)
    return selected_theme
