from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart3:
    @staticmethod
    def _metric_pair(value: Optional[int], total: Optional[int]) -> str:
        """Format a metric pair.

        Args:
            value: Current value.
            total: Total value.

        Returns:
            Display text.
        """

        if value is None:
            return "-"
        if total is None:
            return str(value)
        return f"{value}/{total}"

    @staticmethod
    def _metric_float(value: Optional[float], suffix: str = "") -> str:
        """Format a floating-point metric.

        Args:
            value: Metric value.
            suffix: Optional suffix.

        Returns:
            Display text.
        """

        if value is None:
            return "-"
        return f"{value:.4g}{suffix}"

    def _init_telemetry_store(self, model_dir: Path) -> None:
        """Create or reset the SQLite telemetry store for a training run.

        Args:
            model_dir: Model output directory.
        """

        self.telemetry_db_path = initialize_store(model_dir)
        self.telemetry_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_timeline_label.setText("Timeline: live")
        self.live_scrub_active = False

    def _record_live_metric(self, event: dict[str, Any]) -> None:
        """Persist one live training metric event to SQLite.

        Args:
            event: Training progress event.
        """

        if self.telemetry_db_path is None or not self.telemetry_run_id or event.get("step") is None:
            return
        self.telemetry_latest_id = insert_metric(self.telemetry_db_path, self.telemetry_run_id, event)
        self.telemetry_latest_index += 1
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, self.telemetry_latest_index)
        if not self.live_scrub_active:
            self.live_time_slider.setValue(self.telemetry_latest_index)
            self.live_timeline_label.setText("Timeline: live")
        self.live_time_slider.blockSignals(False)

    def _load_existing_telemetry(self, model_dir: Path) -> None:
        """Load the latest saved telemetry run for an opened project.

        Args:
            model_dir: Model output directory that may contain ``training_telemetry.sqlite``.
        """

        db_path = telemetry_db_path(model_dir)
        self.telemetry_db_path = db_path if db_path.exists() else None
        self.telemetry_run_id = ""
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_scrub_active = False
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_time_slider.blockSignals(False)
        self.live_timeline_label.setText("Timeline: no saved telemetry")
        self.live_sample_text.setText("Training text: -")
        if self.telemetry_db_path is None:
            return
        try:
            run_row = latest_run(self.telemetry_db_path)
            if run_row is None:
                self.live_timeline_label.setText("Timeline: no samples")
                return
            self.telemetry_run_id = str(run_row["run_id"])
            self.telemetry_latest_index = int(run_row["sample_count"] or 0)
            self.telemetry_latest_id = int(run_row["latest_id"] or 0)
        except sqlite3.Error as exc:
            self.live_timeline_label.setText("Timeline: could not load")
            self.training_log.append(f"Telemetry load warning: {exc}")
            return
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, self.telemetry_latest_index)
        self.live_time_slider.setValue(self.telemetry_latest_index)
        self.live_time_slider.blockSignals(False)
        if self.telemetry_latest_index:
            rows = self._timeline_rows_until(self.telemetry_latest_index)
            if rows:
                self._apply_timeline_rows(rows)

    def _timeline_rows_until(self, sample_index: int) -> list[sqlite3.Row]:
        """Load telemetry rows up to a selected sample index.

        Args:
            sample_index: Maximum number of samples to load for the active run.

        Returns:
            Ordered telemetry rows for the active run.
        """

        if self.telemetry_db_path is None or not self.telemetry_run_id or sample_index <= 0:
            return []
        return rows_until(self.telemetry_db_path, self.telemetry_run_id, sample_index)

    def _begin_live_scrub(self) -> None:
        """Pause live auto-follow while the timeline slider is being dragged."""

        self.live_scrub_active = True

    def _end_live_scrub(self) -> None:
        """Apply the selected timeline snapshot after slider drag."""

        self._scrub_live_timeline(self.live_time_slider.value())

    def _jump_live_timeline_to_latest(self) -> None:
        """Return timeline display to the latest live point."""

        self.live_scrub_active = False
        self.live_time_slider.setValue(self.telemetry_latest_index)
        self._scrub_live_timeline(self.telemetry_latest_index)
        self.live_timeline_label.setText("Timeline: live")

    def _scrub_live_timeline(self, sample_index: int) -> None:
        """Replay charts and live visual widgets to a selected telemetry point.

        Args:
            sample_index: Timeline sample selected by the slider.
        """

        rows = self._timeline_rows_until(sample_index)
        if not rows:
            return
        self._apply_timeline_rows(rows)

    def _apply_timeline_rows(self, rows: list[sqlite3.Row]) -> None:
        """Apply historical telemetry rows to charts and live widgets.

        Args:
            rows: Ordered SQLite telemetry rows.
        """

        def series(name: str) -> list[tuple[int, float]]:
            return [(int(row["step"]), float(row[name])) for row in rows if row[name] is not None]

        latest = rows[-1]
        self.loss_chart.set_points(series("train_loss"), series("val_loss"))
        self.optimization_chart.set_points(series("learning_rate"), series("grad_norm"))
        self.stability_chart.set_points(series("weight_norm"), series("update_ratio"))
        self.throughput_chart.set_points(series("tokens_per_second"), series("samples_per_second"))
        self.memory_chart.set_points(series("vram_allocated_gb"), series("vram_reserved_gb"))
        snapshot = {key: latest[key] for key in latest.keys()}
        sample_text = str(snapshot.get("sample_text") or "").strip()
        if sample_text:
            self.live_sample_text.setText(f"Training text: {self._compact_preview_text(sample_text, 220)}")
        else:
            self.live_sample_text.setText("Training text: -")
        self._update_live_training_metrics(
            int(latest["step"]),
            snapshot,
            snapshot.get("train_loss"),
            snapshot.get("learning_rate"),
            snapshot.get("grad_norm"),
            snapshot.get("update_ratio"),
            snapshot.get("tokens_per_second"),
            snapshot.get("samples_per_second"),
            snapshot.get("vram_allocated_gb"),
            snapshot.get("vram_reserved_gb"),
            snapshot.get("gpu_memory_percent"),
            snapshot.get("system_cpu_percent"),
            snapshot.get("system_ram_percent"),
            snapshot.get("data_loader_workers"),
        )
        timestamp = datetime.fromtimestamp(float(latest["recorded_at"])).strftime("%H:%M:%S")
        self.live_timeline_label.setText(f"Timeline: step {int(latest['step']):,} @ {timestamp}")

    @staticmethod
    def _compact_preview_text(text: str, limit: int = 220) -> str:
        """Normalize a training preview into a compact single line.

        Args:
            text: Raw decoded preview text.
            limit: Maximum number of displayed characters.

        Returns:
            Single-line text preview.
        """

        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."

    def _build_export_tab(self) -> QWidget:
        """Build the export page.

        Returns:
            Export page widget.
        """

        return build_export_tab(self)

    def _build_benchmark_tab(self) -> QWidget:
        """Build the benchmark prompt page.

        Returns:
            Benchmark page widget.
        """

        return build_benchmark_tab(self)

    def _build_chat_tab(self) -> QWidget:
        """Build the model test chat page.

        Returns:
            Chat page widget.
        """

        return build_chat_tab(self)

    def _panel(self) -> QWidget:
        """Create a base page panel.

        Returns:
            Panel widget.
        """

        page = QWidget()
        page.setObjectName("Panel")
        return page

    def _page_title(self, text: str) -> QLabel:
        """Create a page title label.

        Args:
            text: Title text.

        Returns:
            Label configured as a page title.
        """

        label = QLabel(text)
        label.setObjectName("PageTitle")
        return label

    def _metric_chip(self, text: str, tooltip: str) -> QLabel:
        """Create a compact metric display label.

        Args:
            text: Initial metric text.
            tooltip: User-facing explanation.

        Returns:
            Configured metric label.
        """

        label = QLabel(text)
        label.setObjectName("MetricChip")
        label.setMinimumWidth(150)
        label.setMinimumHeight(28)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(label, tooltip)
        return label

    def _hardware_meter(self, name: str) -> QProgressBar:
        """Create a slider-like hardware utilization meter.

        Args:
            name: Display name for the meter.

        Returns:
            Configured progress bar.
        """

        meter = QProgressBar()
        meter.setObjectName("HardwareMeter")
        meter.setRange(0, 100)
        meter.setValue(0)
        meter.setTextVisible(False)
        meter.setFixedHeight(8)
        meter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(meter, f"Live {name} utilization.")
        return meter

    def _set_meter(self, meter: QProgressBar, name: str, value: Optional[float]) -> None:
        """Update a hardware utilization meter.

        Args:
            meter: Meter to update.
            name: Display name for the meter.
            value: Utilization percentage.
        """

        if value is None:
            meter.setValue(0)
            label = self.hardware_meter_labels.get(id(meter))
            if label is not None:
                label.setText(f"{name}: -")
            return
        bounded = max(0.0, min(100.0, float(value)))
        meter.setValue(int(round(bounded)))
        label = self.hardware_meter_labels.get(id(meter))
        if label is not None:
            label.setText(f"{name}: {bounded:.1f}%")


