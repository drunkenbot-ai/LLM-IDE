"""Main-window integration for standalone training supervision."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from interface import app as _app
from interface.training_process_controller import (
    TrainingProcessController,
    TrainingProcessSnapshot,
    TrainingTelemetryBatch,
    TrainingTerminal,
)

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class TrainingProcessMixin:
    """Connect standalone engine worker supervision to the main window."""

    def _initialize_training_controller(self) -> None:
        self.training_process_timer = QTimer(self)
        self.training_controller = TrainingProcessController(
            self.training_process_timer,
            on_state=self._on_training_process_state,
            on_telemetry=self._on_training_telemetry,
            on_terminal=self._on_training_terminal,
            on_error=self._on_training_process_error,
        )
        self.training_force_stop_available = False
        self.training_ui_cpu_percent: float | None = None

    def _launch_local_training(
        self,
        dataset_dir: Path,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        *,
        training_mode: str,
        stage: str,
        log: QTextEdit,
        progress: QProgressBar,
        button: QPushButton,
        stop_button: QPushButton,
        busy_text: str,
    ) -> None:
        if self.thread is not None:
            QMessageBox.information(
                self,
                "Task running",
                "Please wait for the current background task before starting local training.",
            )
            return
        if self.training_controller.active:
            QMessageBox.information(
                self,
                "Training running",
                f"Run {self.training_controller.request.run_id} is already active.",
            )
            return

        job = TrainingJobSpec.local(
            dataset_dir,
            model_config,
            training_config,
            metadata={"training_mode": training_mode, "stage": stage},
        )
        self.active_training_mode = training_mode
        self.active_training_log = log
        self.active_training_progress = progress
        self.active_training_button = button
        self.active_training_output_dir = training_config.output_dir
        self.active_training_final_button_text = (
            "Start Fine-Tune" if training_mode == "fine_tune" else "Start Training"
        )
        button.setEnabled(False)
        button.setText(f"{busy_text}...")
        stop_button.setText("Stop")
        stop_button.setEnabled(True)
        project_dir = self.current_project_file.parent if self.current_project_file else None
        notifier_path = default_notifier_config_path(project_dir)
        try:
            request = self.training_controller.launch(
                job,
                notifier_config_path=notifier_path,
            )
        except Exception as exc:
            log.append(f"Standalone training launch failed: {exc}")
            self._finish_training_controls()
            QMessageBox.warning(self, "Training launch failed", str(exc))
            return

        self.telemetry_db_path = request.telemetry_db_path
        self.telemetry_run_id = request.run_id
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        log.append(f"Standalone worker request: {request.manifest_path.parent / 'request.json'}")
        log.append(f"Training run ID: {request.run_id}")
        try:
            self._remember_training_request(request)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._on_training_process_error(
                f"Could not persist training reattachment metadata: {exc}"
            )

    def discover_training_run(self) -> bool:
        """Discover and safely reattach to the active or latest project run."""
        self.training_controller.detach()
        manifest_path = str(self.persisted_training_process.get("manifest_path") or "")
        if manifest_path and Path(manifest_path).exists():
            try:
                self.training_controller.attach(Path(manifest_path))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._on_training_process_error(
                    f"Could not reattach from saved training metadata: {exc}"
                )
            else:
                request = self.training_controller.request
                expected_run_id = str(
                    self.persisted_training_process.get("run_id") or ""
                )
                if request is not None and request.run_id == expected_run_id:
                    return True
        output_dirs = []
        for value in (self.model_dir.text(), self.fine_tune_output_dir.text()):
            if value.strip():
                output_dirs.append(Path(value))
        discovered = self.training_controller.discover(output_dirs)
        if discovered and self.training_controller.request is not None:
            try:
                self._remember_training_request(self.training_controller.request)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._on_training_process_error(
                    f"Could not persist discovered training metadata: {exc}"
                )
        return discovered

    def _remember_training_request(self, request: Any) -> None:
        metadata = {
            "schema": "drunkenbot.training-process-reference",
            "version": 1,
            "run_id": request.run_id,
            "manifest_path": str(request.manifest_path),
            "control_path": str(request.control_path),
            "telemetry_db_path": str(request.telemetry_db_path),
        }
        self.persisted_training_process = metadata
        project_file = self.current_project_file
        if project_file is None or not project_file.exists():
            return
        data = json.loads(project_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Project file must contain a JSON object")
        data["training_process"] = metadata
        temporary = project_file.with_suffix(f"{project_file.suffix}.tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, project_file)

    def stop_training_process(self) -> None:
        """Request cooperative stop, or verified force stop after its timeout."""
        try:
            if self.training_force_stop_available:
                self.training_controller.force_stop()
            else:
                self.training_controller.request_stop()
        except RuntimeError as exc:
            self._on_training_process_error(str(exc))

    def _on_training_process_state(self, snapshot: TrainingProcessSnapshot) -> None:
        self.training_force_stop_available = snapshot.force_stop_available
        request = self.training_controller.request
        if request is not None:
            self.telemetry_db_path = request.telemetry_db_path
            self.telemetry_run_id = request.run_id
            self.telemetry_latest_index = self.training_controller.total_metric_rows
            self.telemetry_latest_id = self.training_controller.last_metric_row_id
        identity = f"Run: {snapshot.run_id or '-'} | PID: {snapshot.pid or '-'}"
        detail = f" | {snapshot.message}" if snapshot.message else ""
        text = f"Worker: {snapshot.state} | {identity}{detail}"
        self._set_text_if_changed(self.training_process_status, text)
        self._set_text_if_changed(self.fine_tune_process_status, text)

        if snapshot.run_id:
            self.telemetry_run_id = snapshot.run_id
        if snapshot.state in {"starting", "running"}:
            self.project_state.setText("Training")
            self.train_status.setText(f"Training: {snapshot.state}")
        elif snapshot.state == "stopping":
            self.project_state.setText("Stopping")
            self.train_status.setText("Training: stopping")
        elif snapshot.state == "stale":
            self.project_state.setText("Training stale")
            self.train_status.setText("Training: stale metadata")
        elif snapshot.state == "failed":
            self.project_state.setText("Training failed")
            self.train_status.setText("Training: failed")

        for stop_button in (self.stop_training_button, self.stop_fine_tune_button):
            stop_button.setText("Force Stop" if snapshot.force_stop_available else "Stop")
            stop_button.setEnabled(
                snapshot.force_stop_available or snapshot.state in {"starting", "running"}
            )
        mode = self._active_training_mode()
        self.active_training_mode = mode
        self.active_training_final_button_text = (
            "Start Fine-Tune" if mode == "fine_tune" else "Start Training"
        )
        active_button = self.fine_tune_button if mode == "fine_tune" else self.train_button
        idle_text = "Start Fine-Tune" if mode == "fine_tune" else "Start Training"
        if snapshot.state in {"starting", "running", "stopping"}:
            active_button.setEnabled(False)
            active_button.setText("Fine-tuning..." if mode == "fine_tune" else "Training...")
        elif snapshot.state in {"failed", "stale"}:
            active_button.setEnabled(True)
            active_button.setText(idle_text)
            self._finish_training_controls()

    def _on_training_telemetry(self, batch: TrainingTelemetryBatch) -> None:
        self.training_ui_cpu_percent = batch.ui_process_cpu_percent
        self.telemetry_latest_index = batch.total_metric_rows
        display_metric = dict(batch.latest_metric) if batch.latest_metric is not None else None
        if batch.latest_metric is not None:
            self.telemetry_latest_id = int(batch.latest_metric["id"])
        for event in batch.new_events:
            if event.get("event_type") != "validation" or event.get("val_loss") is None:
                continue
            if display_metric is None:
                display_metric = dict(event)
            else:
                display_metric["val_loss"] = event["val_loss"]
            for row in reversed(self.training_controller.metric_history):
                if row.get("step") == event.get("step"):
                    row["val_loss"] = event["val_loss"]
                    break
        if display_metric is not None:
            mode = self._active_training_mode()
            self._update_training_metrics(
                display_metric,
                update_fine_tune=mode == "fine_tune",
                render_live=False,
            )
        self._update_standalone_training_progress(batch)
        log = self._training_log_for_mode(self._active_training_mode())
        for event in batch.new_events:
            message = str(event.get("message") or "").strip()
            if message:
                event_type = str(event.get("event_type") or "event")
                log.append(f"[{event_type.upper()}] {message}")
        if self.pages.currentIndex() == self.live_page_index:
            self._render_live_snapshot(batch.metric_history, display_metric)

    def _update_standalone_training_progress(self, batch: TrainingTelemetryBatch) -> None:
        percent = None
        for event in reversed(batch.new_events):
            if event.get("percent") is not None:
                percent = int(event["percent"])
                break
        if percent is None and batch.latest_metric is not None:
            step = batch.latest_metric.get("step")
            total_steps = batch.latest_metric.get("total_steps")
            if step is not None and total_steps:
                percent = round(100 * int(step) / int(total_steps))
        if percent is None:
            return
        value = max(0, min(100, percent))
        mode = self._active_training_mode()
        progress = (
            self.active_training_progress
            or (self.fine_tune_progress if mode == "fine_tune" else self.training_progress)
        )
        if progress.value() != value:
            progress.setValue(value)
        if self.live_progress.value() != value:
            self.live_progress.setValue(value)

    def _render_live_snapshot(
        self,
        history: tuple[dict[str, Any], ...],
        latest: dict[str, Any] | None,
    ) -> None:
        """Redraw each expensive Live chart once from the bounded history."""
        if latest is None:
            return

        def series(name: str) -> list[tuple[int, float]]:
            values = []
            for row in history:
                value = self._finite_metric(row.get(name))
                if value is not None and row.get("step") is not None:
                    values.append((int(row["step"]), value))
            return values

        self.loss_chart.set_points(series("train_loss"), series("val_loss"))
        self.optimization_chart.set_points(series("learning_rate"), series("grad_norm"))
        self.stability_chart.set_points(series("weight_norm"), series("update_ratio"))
        self.throughput_chart.set_points(
            series("tokens_per_second"),
            series("samples_per_second"),
        )
        self.memory_chart.set_points(
            series("vram_allocated_gb"),
            series("vram_reserved_gb"),
        )
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, self.telemetry_latest_index)
        if not self.live_scrub_active:
            self.live_time_slider.setValue(self.telemetry_latest_index)
        self.live_time_slider.blockSignals(False)
        self._update_live_training_metrics(
            int(latest["step"]),
            latest,
            self._finite_metric(latest.get("train_loss")),
            self._finite_metric(latest.get("learning_rate")),
            self._finite_metric(latest.get("grad_norm")),
            self._finite_metric(latest.get("update_ratio")),
            self._finite_metric(latest.get("tokens_per_second")),
            self._finite_metric(latest.get("samples_per_second")),
            self._finite_metric(latest.get("vram_allocated_gb")),
            self._finite_metric(latest.get("vram_reserved_gb")),
            self._finite_metric(latest.get("gpu_memory_percent")),
            self._finite_metric(latest.get("system_cpu_percent")),
            self._finite_metric(latest.get("system_ram_percent")),
            latest.get("data_loader_workers"),
        )
        self._set_meter(
            self.live_ui_cpu_bar,
            "UI process CPU",
            self.training_ui_cpu_percent,
        )
        self._set_text_if_changed(
            self.live_training_cpu_label,
            "Training process CPU: unavailable",
        )
        self.live_time_slider.setToolTip(f"{self.telemetry_latest_index:,} durable worker samples")
        self._set_text_if_changed(
            self.live_timeline_label,
            f"Timeline: live | {self.telemetry_latest_index:,} samples",
        )

    def _render_current_live_snapshot(self) -> None:
        if not hasattr(self, "training_controller"):
            return
        history = tuple(self.training_controller.metric_history)
        latest = history[-1] if history else None
        if latest is not None:
            self._render_live_snapshot(history, latest)
        elif self.telemetry_latest_index:
            self._scrub_live_timeline(self.telemetry_latest_index)

    def _on_training_terminal(self, terminal: TrainingTerminal) -> None:
        status = str(terminal.manifest.get("status") or "failed")
        mode = "pretrain"
        if terminal.request is not None:
            mode = str(terminal.request.job.metadata.get("training_mode") or mode)
            self.active_training_output_dir = terminal.request.job.artifacts.output_dir
        self.active_training_mode = mode
        self.active_training_log = self._training_log_for_mode(mode)
        self.active_training_progress = (
            self.fine_tune_progress if mode == "fine_tune" else self.training_progress
        )
        self.active_training_final_button_text = (
            "Start Fine-Tune" if mode == "fine_tune" else "Start Training"
        )
        if status == "failed":
            failure = dict(terminal.manifest.get("failure") or {})
            message = str(failure.get("message") or "Standalone training failed")
            details_path = str(failure.get("details_path") or "")
            self.active_training_log.append(f"Training failed: {message}")
            if details_path:
                self.active_training_log.append(f"Failure details: {details_path}")
            self._finish_training_controls()
            return

        output_paths = dict(terminal.manifest.get("output_paths") or {})
        checkpoint_path = output_paths.get("checkpoint")
        summary_path = output_paths.get("summary")
        if not checkpoint_path or not summary_path:
            self._on_training_process_error(
                f"Terminal run {terminal.manifest.get('run_id')} is missing result paths"
            )
            self._finish_training_controls()
            return
        metric = terminal.latest_metric or {}
        event = terminal.latest_event or {}
        train_loss = self._finite_metric(event.get("train_loss", metric.get("train_loss")))
        val_loss = self._finite_metric(event.get("val_loss", metric.get("val_loss")))
        if train_loss is None:
            try:
                summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
                train_loss = self._finite_metric(summary.get("final_train_loss"))
                val_loss = self._finite_metric(summary.get("final_val_loss"))
            except (OSError, ValueError, TypeError):
                pass
        result = SimpleNamespace(
            checkpoint_path=Path(checkpoint_path),
            summary_path=Path(summary_path),
            final_train_loss=train_loss,
            final_val_loss=val_loss,
            stopped=status == "stopped",
            run_id=str(terminal.manifest.get("run_id") or ""),
        )
        self._training_finished(result)

    def _on_training_process_error(self, message: str) -> None:
        LOGGER.error("Standalone training supervision: %s", message)
        self._training_log_for_mode(self._active_training_mode()).append(
            f"Training process warning: {message}"
        )

    def _active_training_mode(self) -> str:
        request = self.training_controller.request
        if request is None:
            return "pretrain"
        return str(request.job.metadata.get("training_mode") or "pretrain")

    def _training_log_for_mode(self, mode: str) -> QTextEdit:
        return self.fine_tune_log if mode == "fine_tune" else self.training_log

    def _finish_training_controls(self) -> None:
        mode = self.active_training_mode
        button = self.active_training_button or (
            self.fine_tune_button if mode == "fine_tune" else self.train_button
        )
        button.setEnabled(True)
        button.setText(self.active_training_final_button_text)
        for stop_button in (self.stop_training_button, self.stop_fine_tune_button):
            stop_button.setText("Stop")
            stop_button.setEnabled(False)
        self.active_training_log = None
        self.active_training_progress = None
        self.active_training_button = None
        self.active_training_output_dir = None

    @staticmethod
    def _set_text_if_changed(widget: QLabel, text: str) -> None:
        if widget.text() != text:
            widget.setText(text)
