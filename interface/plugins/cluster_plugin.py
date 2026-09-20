"""Cluster Job Monitor & Fleet Synchronization plugin for DrunkenBot IDE."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from interface.plugins.plugin_base import BasePlugin, PluginNavEntry, PluginSpec
from interface.tabs.job_manager_tab import build_job_manager_tab


class ClusterPlugin(BasePlugin):
    """Plugin providing distributed cluster job monitoring and Local SGD fleet sync."""

    DEFAULT_SETTINGS = {
        "enabled": True,
        "shared_storage_path": "runs/cluster_shared",
        "sync_interval_steps": 250,
        "node_discovery_timeout": 30,
    }

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        """Initialize the Cluster Job Monitor plugin with custom or default settings.

        Args:
            settings: Optional initial configuration settings dictionary.
        """
        merged_settings = dict(self.DEFAULT_SETTINGS)
        if settings:
            merged_settings.update(settings)
        spec = PluginSpec(
            plugin_id="cluster_job_monitor",
            name="Cluster Job Monitor",
            description="Distributed Local SGD fleet coordination, network worker heartbeat, and shared job monitor.",
            version="1.0.0",
            icon="⚡",
            enabled_by_default=True,
            settings=merged_settings,
        )
        super().__init__(spec)

    def get_nav_entries(self) -> list[PluginNavEntry]:
        """Return navigation rail entries for the cluster job manager.

        Returns:
            List containing the job manager navigation entry.
        """
        return [
            PluginNavEntry(
                nav_id="jobs_nav",
                display_name="Job manager",
                icon_name="job_tab_icon.png",
                page_index_attr="job_manager_page_index",
            )
        ]

    def get_screen_builders(self) -> list[tuple[PluginNavEntry, Callable[[Any], QWidget]]]:
        """Return screen builders linked to their nav entries.

        Returns:
            List of (PluginNavEntry, builder_function) tuples.
        """
        return [(
            PluginNavEntry(
                nav_id="jobs_nav",
                display_name="Job manager",
                icon_name="job_tab_icon.png",
                page_index_attr="job_manager_page_index",
            ),
            build_job_manager_tab,
        )]

    def create_settings_widget(self, parent: QWidget | None = None) -> QWidget:
        """Create an options configuration form for cluster monitor settings.

        Args:
            parent: Optional parent widget.

        Returns:
            Widget containing cluster configuration form.
        """
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(8)

        storage_input = QLineEdit(str(self.spec.settings.get("shared_storage_path", "runs/cluster_shared")))
        storage_input.setObjectName("cluster_storage_path")
        widget.storage_input = storage_input
        form.addRow("Shared Storage Path:", storage_input)

        sync_spin = QSpinBox()
        sync_spin.setObjectName("cluster_sync_interval")
        sync_spin.setRange(10, 10_000)
        sync_spin.setSingleStep(25)
        sync_spin.setValue(int(self.spec.settings.get("sync_interval_steps", 250)))
        widget.sync_spin = sync_spin
        form.addRow("Local SGD Sync Steps:", sync_spin)

        timeout_spin = QSpinBox()
        timeout_spin.setObjectName("cluster_node_timeout")
        timeout_spin.setRange(5, 600)
        timeout_spin.setSingleStep(5)
        timeout_spin.setValue(int(self.spec.settings.get("node_discovery_timeout", 30)))
        timeout_spin.setSuffix(" sec")
        widget.timeout_spin = timeout_spin
        form.addRow("Node Heartbeat Timeout:", timeout_spin)

        layout.addLayout(form)
        tip = QLabel("Configures shared directory and heartbeat timeout across cluster nodes.")
        tip.setStyleSheet("color: #94a3b8; font-size: 11px; font-style: italic;")
        layout.addWidget(tip)
        layout.addStretch(1)
        return widget

    def save_settings_from_widget(self, widget: QWidget) -> None:
        """Persist settings values from the settings widget into plugin configuration.

        Args:
            widget: Widget created by create_settings_widget.
        """
        if hasattr(widget, "storage_input"):
            self.spec.settings["shared_storage_path"] = widget.storage_input.text().strip()
        if hasattr(widget, "sync_spin"):
            self.spec.settings["sync_interval_steps"] = widget.sync_spin.value()
        if hasattr(widget, "timeout_spin"):
            self.spec.settings["node_discovery_timeout"] = widget.timeout_spin.value()
