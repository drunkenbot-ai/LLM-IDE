"""Inference & Neural Chat Laboratory plugin for DrunkenBot IDE."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from inference.ui.benchmark_tab import build_benchmark_tab
from inference.ui.chat_tab import build_chat_tab
from interface.plugins.plugin_base import BasePlugin, PluginNavEntry, PluginSpec


class InferencePlugin(BasePlugin):
    """Plugin providing Chat Laboratory and Benchmark Suite."""

    DEFAULT_SETTINGS = {
        "enabled": True,
        "default_temperature": 0.70,
        "default_top_p": 0.90,
        "default_context_length": 2048,
        "enable_benchmarks": True,
    }

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        """Initialize the Inference plugin with custom or default settings.

        Args:
            settings: Optional initial configuration settings dictionary.
        """
        merged_settings = dict(self.DEFAULT_SETTINGS)
        if settings:
            merged_settings.update(settings)
        spec = PluginSpec(
            plugin_id="inference",
            name="Inference & Chat Laboratory",
            description="Interactive neural chat testing, prompt playground, and model benchmark evaluation suite.",
            version="1.0.0",
            icon="💬",
            enabled_by_default=True,
            settings=merged_settings,
        )
        super().__init__(spec)

    def get_nav_entries(self) -> list[PluginNavEntry]:
        """Return navigation rail entries for inference and benchmark tools.

        Returns:
            List of navigation entries.
        """
        entries = []
        if self.spec.settings.get("enable_benchmarks", True):
            entries.append(
                PluginNavEntry(
                    nav_id="benchmark_nav",
                    display_name="Benchmarks",
                    icon_name="benchmark_tab_icon.png",
                    page_index_attr="benchmark_page_index",
                )
            )
        entries.append(
            PluginNavEntry(
                nav_id="chat_nav",
                display_name="Chat",
                icon_name="chat_tab_icon.png",
                page_index_attr="chat_page_index",
            )
        )
        return entries

    def get_screen_builders(self) -> list[tuple[PluginNavEntry, Callable[[Any], QWidget]]]:
        """Return screen builders linked to their nav entries.

        Returns:
            List of (PluginNavEntry, builder_function) tuples.
        """
        builders: list[tuple[PluginNavEntry, Callable[[Any], QWidget]]] = []
        if self.spec.settings.get("enable_benchmarks", True):
            builders.append((
                PluginNavEntry(
                    nav_id="benchmark_nav",
                    display_name="Benchmarks",
                    icon_name="benchmark_tab_icon.png",
                    page_index_attr="benchmark_page_index",
                ),
                build_benchmark_tab,
            ))
        builders.append((
            PluginNavEntry(
                nav_id="chat_nav",
                display_name="Chat",
                icon_name="chat_tab_icon.png",
                page_index_attr="chat_page_index",
            ),
            build_chat_tab,
        ))
        return builders

    def create_settings_widget(self, parent: QWidget | None = None) -> QWidget:
        """Create an options configuration form for inference parameters.

        Args:
            parent: Optional parent widget.

        Returns:
            Widget containing configuration form.
        """
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(8)

        temp_spin = QDoubleSpinBox()
        temp_spin.setObjectName("inference_temperature")
        temp_spin.setRange(0.01, 2.0)
        temp_spin.setSingleStep(0.05)
        temp_spin.setValue(float(self.spec.settings.get("default_temperature", 0.70)))
        widget.temp_spin = temp_spin
        form.addRow("Default Temperature:", temp_spin)

        topp_spin = QDoubleSpinBox()
        topp_spin.setObjectName("inference_top_p")
        topp_spin.setRange(0.01, 1.0)
        topp_spin.setSingleStep(0.05)
        topp_spin.setValue(float(self.spec.settings.get("default_top_p", 0.90)))
        widget.topp_spin = topp_spin
        form.addRow("Default Top-P (Nucleus):", topp_spin)

        ctx_spin = QSpinBox()
        ctx_spin.setObjectName("inference_context_length")
        ctx_spin.setRange(128, 1_000_000)
        ctx_spin.setSingleStep(128)
        ctx_spin.setValue(int(self.spec.settings.get("default_context_length", 2048)))
        widget.ctx_spin = ctx_spin
        form.addRow("Default Context Tokens:", ctx_spin)

        bench_chk = QCheckBox("Enable Benchmark Suite tab")
        bench_chk.setObjectName("inference_benchmarks_enabled")
        bench_chk.setChecked(bool(self.spec.settings.get("enable_benchmarks", True)))
        widget.bench_chk = bench_chk
        form.addRow("Benchmarks Tab:", bench_chk)

        layout.addLayout(form)
        tip = QLabel("Changes will take effect on next model loading or tab switch.")
        tip.setStyleSheet("color: #94a3b8; font-size: 11px; font-style: italic;")
        layout.addWidget(tip)
        layout.addStretch(1)
        return widget

    def save_settings_from_widget(self, widget: QWidget) -> None:
        """Persist settings values from the settings widget into plugin configuration.

        Args:
            widget: Widget created by create_settings_widget.
        """
        if hasattr(widget, "temp_spin"):
            self.spec.settings["default_temperature"] = widget.temp_spin.value()
        if hasattr(widget, "topp_spin"):
            self.spec.settings["default_top_p"] = widget.topp_spin.value()
        if hasattr(widget, "ctx_spin"):
            self.spec.settings["default_context_length"] = widget.ctx_spin.value()
        if hasattr(widget, "bench_chk"):
            self.spec.settings["enable_benchmarks"] = widget.bench_chk.isChecked()
