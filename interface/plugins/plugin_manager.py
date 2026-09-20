"""Plugin manager and registry for DrunkenBot IDE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interface.plugins.cluster_plugin import ClusterPlugin
from interface.plugins.inference_plugin import InferencePlugin
from interface.plugins.plugin_base import BasePlugin


class PluginManager:
    """Manages discovery, configuration persistence, and lifecycle of IDE plugins."""

    CONFIG_PATH = Path.home() / ".drunkenbot_ide" / "plugins_config.json"

    def __init__(self, config_file: Path | None = None) -> None:
        """Initialize the plugin manager and register built-in plugins.

        Args:
            config_file: Optional path to configuration file (defaults to home directory).
        """
        self.config_path = config_file or self.CONFIG_PATH
        self._plugins: dict[str, BasePlugin] = {}
        self._load_and_register_defaults()

    def _load_and_register_defaults(self) -> None:
        """Register built-in plugins with persisted settings if available."""
        saved_config = self._read_config()

        inference_settings = saved_config.get("inference", {})
        cluster_settings = saved_config.get("cluster_job_monitor", {})

        self.register_plugin(InferencePlugin(inference_settings))
        self.register_plugin(ClusterPlugin(cluster_settings))

    def _read_config(self) -> dict[str, Any]:
        """Read saved configuration from disk safely.

        Returns:
            Dictionary of saved plugin configurations.
        """
        if not self.config_path.is_file():
            return {}
        try:
            content = self.config_path.read_text(encoding="utf-8")
            data = json.loads(content)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save_config(self) -> None:
        """Save current plugin settings and enabled states to disk."""
        data = {
            plugin_id: plugin.spec.settings
            for plugin_id, plugin in self._plugins.items()
        }
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def register_plugin(self, plugin: BasePlugin) -> None:
        """Register a plugin in the manager.

        Args:
            plugin: Plugin instance to register.
        """
        self._plugins[plugin.plugin_id] = plugin

    def get_plugin(self, plugin_id: str) -> BasePlugin | None:
        """Retrieve a registered plugin by identifier.

        Args:
            plugin_id: Plugin identifier string.

        Returns:
            BasePlugin instance or None if not registered.
        """
        return self._plugins.get(plugin_id)

    def is_plugin_enabled(self, plugin_id: str) -> bool:
        """Return whether a plugin is enabled.

        Args:
            plugin_id: Plugin identifier string.

        Returns:
            True if plugin is registered and enabled, False otherwise.
        """
        plugin = self.get_plugin(plugin_id)
        return plugin.is_enabled if plugin is not None else False

    def set_plugin_enabled(self, plugin_id: str, enabled: bool) -> None:
        """Set a plugin's enabled status and persist configuration.

        Args:
            plugin_id: Plugin identifier.
            enabled: True to enable, False to disable.
        """
        plugin = self.get_plugin(plugin_id)
        if plugin is not None:
            plugin.set_enabled(enabled)
            self.save_config()

    def all_plugins(self) -> list[BasePlugin]:
        """Return all registered plugins.

        Returns:
            List of all registered plugin instances.
        """
        return list(self._plugins.values())

    def active_plugins(self) -> list[BasePlugin]:
        """Return all currently enabled plugins.

        Returns:
            List of enabled plugin instances.
        """
        return [p for p in self._plugins.values() if p.is_enabled]

    @classmethod
    def instance(cls) -> PluginManager:
        """Return the global PluginManager singleton instance.

        Returns:
            Shared PluginManager instance.
        """
        return get_plugin_manager()


_INSTANCE: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Return the global PluginManager singleton instance.

    Returns:
        Shared PluginManager instance.
    """
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = PluginManager()
    return _INSTANCE
