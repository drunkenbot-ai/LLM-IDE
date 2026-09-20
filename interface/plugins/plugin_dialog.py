"""Plugin configuration modal dialog for DrunkenBot IDE."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from interface.plugins.plugin_base import BasePlugin
from interface.plugins.plugin_manager import PluginManager, get_plugin_manager


class PluginSettingsDialog(QDialog):
    """Dialog allowing users to enable/disable and configure IDE plugins."""

    def __init__(self, parent: QWidget | None = None, manager: PluginManager | None = None) -> None:
        """Initialize the plugin settings dialog.

        Args:
            parent: Optional parent widget.
            manager: Optional PluginManager instance.
        """
        super().__init__(parent)
        self.manager = manager or get_plugin_manager()
        self._plugin_widgets: dict[str, QWidget] = {}
        self._checkboxes: dict[str, QCheckBox] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Construct the dialog layout and widgets."""
        self.setWindowTitle("IDE Plugins Configuration")
        self.resize(680, 440)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        header = QLabel("Plugins & Extensions")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #f59e0b;")
        layout.addWidget(header)

        sub = QLabel("Enable or disable optional capabilities and configure runtime parameters.")
        sub.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(12)

        # Left list of plugins
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(230)
        self.list_widget.currentRowChanged.connect(self._on_plugin_selected)
        body.addWidget(self.list_widget)

        # Right settings stack
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background-color: #14141e; border: 1px solid #232336; border-radius: 6px;")
        body.addWidget(self.stack, 1)

        layout.addLayout(body, 1)

        # Populate plugins
        for plugin in self.manager.all_plugins():
            self._add_plugin_entry(plugin)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save & Apply")
        save_btn.setStyleSheet("background-color: #f59e0b; color: #000000; font-weight: bold; padding: 6px 16px; border-radius: 4px;")
        save_btn.clicked.connect(self._on_save_applied)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _add_plugin_entry(self, plugin: BasePlugin) -> None:
        """Add a plugin to the selection list and stack.

        Args:
            plugin: Plugin instance to register in the dialog.
        """
        item = QListWidgetItem(f"{plugin.spec.icon}  {plugin.name}")
        self.list_widget.addItem(item)

        page = QWidget()
        p_layout = QVBoxLayout(page)
        p_layout.setContentsMargins(14, 14, 14, 14)
        p_layout.setSpacing(10)

        top_row = QHBoxLayout()
        p_title = QLabel(f"{plugin.spec.icon} {plugin.name}")
        p_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f8fafc;")
        top_row.addWidget(p_title)
        top_row.addStretch(1)

        enable_chk = QCheckBox("Enabled")
        enable_chk.setChecked(plugin.is_enabled)
        enable_chk.setStyleSheet("font-weight: bold; color: #10b981;")
        self._checkboxes[plugin.plugin_id] = enable_chk
        top_row.addWidget(enable_chk)
        p_layout.addLayout(top_row)

        desc = QLabel(plugin.description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        p_layout.addWidget(desc)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #232336;")
        p_layout.addWidget(sep)

        settings_widget = plugin.create_settings_widget(page)
        self._plugin_widgets[plugin.plugin_id] = settings_widget
        p_layout.addWidget(settings_widget, 1)

        self.stack.addWidget(page)

    def _on_plugin_selected(self, row: int) -> None:
        """Handle row change in plugin list.

        Args:
            row: Newly selected row index.
        """
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)

    def _on_save_applied(self) -> None:
        """Save settings for all plugins and accept the dialog."""
        for plugin in self.manager.all_plugins():
            chk = self._checkboxes.get(plugin.plugin_id)
            if chk is not None:
                plugin.set_enabled(chk.isChecked())
            w = self._plugin_widgets.get(plugin.plugin_id)
            if w is not None:
                plugin.save_settings_from_widget(w)
        self.manager.save_config()
        self.accept()
