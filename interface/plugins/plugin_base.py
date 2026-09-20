"""Base plugin specification and interface for DrunkenBot IDE."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtWidgets import QWidget


@dataclass
class PluginNavEntry:
    """Navigation entry definition contributed by a plugin.

    Attributes:
        nav_id: Internal navigation attribute identifier.
        display_name: Human-readable display label and tooltip.
        icon_name: Icon filename located in interface/icons.
        page_index_attr: Attribute name on window storing the stacked page index.
    """

    nav_id: str
    display_name: str
    icon_name: str
    page_index_attr: str


@dataclass
class PluginSpec:
    """Metadata specification for an IDE plugin.

    Attributes:
        plugin_id: Unique string identifier for the plugin.
        name: Human-readable plugin name.
        description: Brief explanation of what the plugin provides.
        version: Version string.
        icon: Display icon or emoji.
        enabled_by_default: Whether the plugin is active by default.
        settings: Current configuration key-values.
    """

    plugin_id: str
    name: str
    description: str
    version: str = "1.0.0"
    icon: str = "🧩"
    enabled_by_default: bool = True
    settings: dict[str, Any] = field(default_factory=dict)


class BasePlugin(ABC):
    """Abstract base class for DrunkenBot IDE modular plugins."""

    def __init__(self, spec: PluginSpec) -> None:
        """Initialize plugin with specification.

        Args:
            spec: Plugin specification metadata.
        """
        self.spec = spec

    @property
    def plugin_id(self) -> str:
        """Unique plugin identifier."""
        return self.spec.plugin_id

    @property
    def name(self) -> str:
        """Human-readable plugin name."""
        return self.spec.name

    @property
    def description(self) -> str:
        """Plugin description."""
        return self.spec.description

    @property
    def is_enabled(self) -> bool:
        """Whether the plugin is currently enabled."""
        return self.spec.settings.get("enabled", self.spec.enabled_by_default)

    def set_enabled(self, enabled: bool) -> None:
        """Set enabled status of the plugin.

        Args:
            enabled: True to activate, False to deactivate.
        """
        self.spec.settings["enabled"] = enabled

    @abstractmethod
    def get_nav_entries(self) -> list[PluginNavEntry]:
        """Return navigation rail entries contributed by this plugin.

        Returns:
            List of navigation entries.
        """
        ...

    @abstractmethod
    def get_screen_builders(self) -> list[tuple[PluginNavEntry, Callable[[Any], QWidget]]]:
        """Return list of (nav_entry, screen_builder) pairs.

        Returns:
            List of tuples linking nav entry to screen builder function.
        """
        ...

    @abstractmethod
    def create_settings_widget(self, parent: QWidget | None = None) -> QWidget:
        """Create and return a settings configuration widget for this plugin.

        Args:
            parent: Optional parent widget.

        Returns:
            Configured QWidget displaying editable settings.
        """
        ...

    @abstractmethod
    def save_settings_from_widget(self, widget: QWidget) -> None:
        """Read values from the settings widget and update plugin settings.

        Args:
            widget: Settings widget created by create_settings_widget.
        """
        ...
