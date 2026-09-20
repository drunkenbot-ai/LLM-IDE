"""Unit tests for DrunkenBot IDE plugin manager, plugins, and settings."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QSpinBox

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from interface.plugins.cluster_plugin import ClusterPlugin
from interface.plugins.inference_plugin import InferencePlugin
from interface.plugins.plugin_dialog import PluginSettingsDialog
from interface.plugins.plugin_manager import PluginManager


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Ensure a QApplication instance is initialized for GUI testing.

    Returns:
        The active QApplication instance.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_plugin_manager_initialization(tmp_path: Path) -> None:
    """Verify default plugins are registered with expected initial states."""
    config_file = tmp_path / "plugins_test.json"
    manager = PluginManager(config_file=config_file)

    all_plugins = manager.all_plugins()
    assert len(all_plugins) >= 2

    inference = manager.get_plugin("inference")
    assert inference is not None
    assert inference.is_enabled is True
    assert inference.name == "Inference & Chat Laboratory"

    cluster = manager.get_plugin("cluster_job_monitor")
    assert cluster is not None
    assert cluster.is_enabled is True
    assert cluster.name == "Cluster Job Monitor"


def test_plugin_toggle_and_persistence(tmp_path: Path) -> None:
    """Verify that toggling plugin enabled state persists to disk."""
    config_file = tmp_path / "plugins_test.json"
    manager = PluginManager(config_file=config_file)

    manager.set_plugin_enabled("cluster_job_monitor", False)
    assert manager.is_plugin_enabled("cluster_job_monitor") is False
    assert config_file.is_file()

    # Create fresh manager from saved file
    reloaded_manager = PluginManager(config_file=config_file)
    assert reloaded_manager.is_plugin_enabled("cluster_job_monitor") is False
    assert reloaded_manager.is_plugin_enabled("inference") is True


def test_inference_plugin_settings_widget(qapp: QApplication) -> None:
    """Verify InferencePlugin settings widget correctly updates spec."""
    plugin = InferencePlugin()
    widget = plugin.create_settings_widget()

    temp_spin = widget.findChild(QDoubleSpinBox, "inference_temperature")
    assert temp_spin is not None
    temp_spin.setValue(0.42)

    chk = widget.findChild(QCheckBox, "inference_benchmarks_enabled")
    assert chk is not None
    chk.setChecked(False)

    plugin.save_settings_from_widget(widget)
    assert plugin.spec.settings["default_temperature"] == 0.42
    assert plugin.spec.settings["enable_benchmarks"] is False


def test_cluster_plugin_settings_widget(qapp: QApplication) -> None:
    """Verify ClusterPlugin settings widget correctly updates spec."""
    plugin = ClusterPlugin()
    widget = plugin.create_settings_widget()

    sync_spin = widget.findChild(QSpinBox, "cluster_sync_interval")
    assert sync_spin is not None
    sync_spin.setValue(45)

    plugin.save_settings_from_widget(widget)
    assert plugin.spec.settings["sync_interval_steps"] == 45


def test_plugin_settings_dialog_save(tmp_path: Path, qapp: QApplication) -> None:
    """Verify PluginSettingsDialog saves modified settings cleanly."""
    config_file = tmp_path / "plugins_test.json"
    manager = PluginManager(config_file=config_file)

    dialog = PluginSettingsDialog(parent=None, manager=manager)
    assert dialog.stack.count() >= 2

    dialog._on_save_applied()
    assert config_file.is_file()
