from __future__ import annotations

# LiveScreenMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class LiveScreenMixin:
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
        if (
            self.telemetry_latest_index
            and self.pages.currentIndex() == self.live_page_index
        ):
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

    def _update_live_training_metrics(
        self,
        step: int,
        event: dict[str, Any],
        train_loss: Optional[float],
        learning_rate: Optional[float],
        grad_norm: Optional[float],
        update_ratio: Optional[float],
        tokens_per_second: Optional[float],
        samples_per_second: Optional[float],
        vram_allocated: Optional[float],
        vram_reserved: Optional[float],
        gpu_memory: Optional[float],
        system_cpu: Optional[float],
        system_ram: Optional[float],
        data_workers: Optional[int],
    ) -> None:
        """Update live tracker widgets from one training progress event.

        Args:
            step: Current optimizer step.
            event: Progress event emitted by training.
            train_loss: Latest training loss.
            learning_rate: Current learning rate.
            grad_norm: Current gradient norm.
            update_ratio: Current parameter update ratio.
            tokens_per_second: Current token throughput.
            samples_per_second: Current sample throughput.
            vram_allocated: Current CUDA allocated memory in GB.
            vram_reserved: Current CUDA reserved memory in GB.
            gpu_memory: Current GPU memory pressure percentage.
            system_cpu: Current system CPU utilization percentage.
            system_ram: Current system RAM utilization percentage.
            data_workers: CPU data-loader worker count.
        """

        total_steps = event.get("total_steps")
        if "epoch" in event and "total_epochs" in event:
            self.live_epoch_metric.setText(f"Epoch: {event['epoch']}/{event['total_epochs']}")
        if total_steps:
            self.live_step_metric.setText(f"Step: {step:,}/{int(total_steps):,}")
            data_percent = min(100.0, max(0.0, (step / max(1, int(total_steps))) * 100.0))
            self.live_data_metric.setText(f"Data: {data_percent:.1f}%")
            self.live_progress.setValue(int(data_percent))
        else:
            self.live_step_metric.setText(f"Step: {step:,}")
        if tokens_per_second is not None:
            self.live_tokens_metric.setText(f"Tokens/sec: {float(tokens_per_second):,.0f}")
        if train_loss is not None:
            self.live_loss_metric.setText(f"Loss: {float(train_loss):.4f}")
        if learning_rate is not None:
            self.live_lr_metric.setText(f"LR: {float(learning_rate):.2e}")
        sample_text = str(event.get("sample_text") or "").strip()
        if sample_text:
            self.live_sample_text.setText(f"Training text: {self._compact_preview_text(sample_text, 220)}")
        self.live_layer_status.setText(f"Layers: {self.n_layer.value()}")
        self.live_head_status.setText(f"Heads: {self.n_head.value()}")
        self.live_hidden_status.setText(f"Hidden size: {self.n_embd.value()}")
        self.live_batch_status.setText(f"Batch size: {self.batch_size.value()}")
        self.live_context_status.setText(f"Context: {self.train_context_length.value()}")
        self.live_device_status.setText(f"Device: {self.device.currentText()}")
        self.live_worker_status.setText(f"CPU workers: {data_workers if data_workers is not None else self.data_loader_workers.value()}")
        self._set_meter(self.live_cpu_bar, "System CPU", system_cpu if system_cpu is not None else self._system_cpu_value())
        self._set_meter(self.live_gpu_bar, "GPU memory", gpu_memory)
        if vram_allocated is not None or vram_reserved is not None:
            allocated = float(vram_allocated or 0.0)
            reserved = float(vram_reserved or 0.0)
            reserved_percent = None
            if self.device.currentText().startswith("cuda") and torch.cuda.is_available():
                try:
                    _, total_vram = torch.cuda.mem_get_info()
                    reserved_percent = min(100.0, 100.0 * reserved * (1024 ** 3) / max(total_vram, 1))
                except Exception:
                    reserved_percent = None
            self._set_meter(self.live_vram_bar, "VRAM reserved", reserved_percent)
            self.live_vram_label.setText(f"VRAM reserved: {reserved:.2f} GB ({allocated:.2f} GB active)")
        self._set_meter(self.live_ram_bar, "System RAM", system_ram if system_ram is not None else self._system_ram_value())
        latest_loss = float(train_loss) if train_loss is not None else None
        self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), step, latest_loss)
        self.live_prediction_chart.update_distribution(step, latest_loss)
        self.live_attention_chart.update_heatmap(step, grad_norm)
        self.live_activation_chart.update_histogram(step, tokens_per_second)
        self.live_gradient_chart.update_flow(self.n_layer.value(), grad_norm, step)

    def _system_ram_value(self) -> Optional[float]:
        """Read system RAM utilization for live telemetry.

        Returns:
            System RAM percentage, or None when unavailable.
        """

        if psutil is None:
            return None
        return float(psutil.virtual_memory().percent)

    def _system_cpu_value(self) -> Optional[float]:
        """Read system CPU utilization for live telemetry.

        Returns:
            System CPU percentage, or None when unavailable.
        """

        if psutil is None:
            return None
        return float(psutil.cpu_percent(interval=None))
