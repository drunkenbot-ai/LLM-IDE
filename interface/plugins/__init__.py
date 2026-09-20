"""DrunkenBot IDE plugin system."""

from interface.plugins.cluster_plugin import ClusterPlugin
from interface.plugins.inference_plugin import InferencePlugin
from interface.plugins.plugin_base import BasePlugin, PluginNavEntry, PluginSpec
from interface.plugins.plugin_dialog import PluginSettingsDialog
from interface.plugins.plugin_manager import PluginManager, get_plugin_manager

__all__ = [
    "BasePlugin",
    "ClusterPlugin",
    "InferencePlugin",
    "PluginManager",
    "PluginNavEntry",
    "PluginSettingsDialog",
    "PluginSpec",
    "get_plugin_manager",
]
