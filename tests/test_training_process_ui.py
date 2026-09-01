from __future__ import annotations

import inspect
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QCloseEvent

from interface import app as interface_app
from interface import process_control
from interface.training_artifacts import select_training_artifacts
from interface.training_process_controller import (
    TrainingProcessSnapshot,
    TrainingTelemetryBatch,
)

TaskRunnerMixin = interface_app.TaskRunnerMixin


def fake_request(tmp_path: Path, run_id: str = "run-test"):
    run_dir = tmp_path / "training_runs" / run_id
    return SimpleNamespace(
        run_id=run_id,
        manifest_path=run_dir / "manifest.json",
        control_path=run_dir / "control.json",
        telemetry_db_path=tmp_path / "training_telemetry.sqlite",
        job=SimpleNamespace(
            metadata={"training_mode": "pretrain"},
            artifacts=SimpleNamespace(output_dir=tmp_path),
        ),
    )


def test_checkpoint_selection_separates_lora_inference_and_resume() -> None:
    selection = select_training_artifacts(
        {
            "recommended_checkpoint_path": "checkpoints/checkpoint_best_val.pt",
            "best_checkpoint_path": "checkpoints/checkpoint_best_val.pt",
            "best_resume_checkpoint_path": "checkpoints/checkpoint_best_val_resume.pt",
        },
        Path("final_model.pt"),
    )

    assert selection.inference_path.name == "checkpoint_best_val.pt"
    assert selection.resume_path.name == "checkpoint_best_val_resume.pt"


class QueueDrainHarness(TaskRunnerMixin):
    def __init__(self, events) -> None:
        self.progress_queue = Queue()
        for event in events:
            self.progress_queue.put(event)
        self.active_log = object()
        self.active_progress_bar = SimpleNamespace(
            setValue=lambda value: setattr(self, "percent", value)
        )
        self.active_task_kind = ""
        self.active_button = None
        self.handled = []
        self.notifications = []
        self.percent = None

    def _handle_progress(self, event, _log, _progress) -> None:
        self.handled.append(event)

    def _notify_progress(self, event) -> None:
        self.notifications.append(event)


def test_final_queue_drain_preserves_diagnostics_and_coalesces_metrics() -> None:
    events = [{"step": index, "percent": index} for index in range(20)]
    events.insert(4, {"message": "Invalid record: data.jsonl: line 8: malformed JSON"})
    events.insert(10, {"event_type": "failure", "message": "preparation failed"})
    harness = QueueDrainHarness(events)

    harness._drain_progress_queue(final=True)

    messages = [event.get("message") for event in harness.handled if isinstance(event, dict)]
    assert "Invalid record: data.jsonl: line 8: malformed JSON" in messages
    assert "preparation failed" in messages
    assert sum("step" in event for event in harness.handled if isinstance(event, dict)) == 1
    assert harness.progress_queue.empty()
    assert harness.percent == 19


def test_pretraining_and_fine_tuning_share_standalone_controller() -> None:
    pretraining_source = inspect.getsource(interface_app.TrainingRunMixin.start_training)
    fine_tuning_source = inspect.getsource(interface_app.FineTuningRunMixin.start_fine_tuning)

    assert "_launch_local_training(" in pretraining_source
    assert "_launch_local_training(" in fine_tuning_source
    assert "_run_task(" not in pretraining_source
    assert "_run_task(" not in fine_tuning_source
    assert "publish_remote_training_job" in pretraining_source
    assert "publish_remote_training_job" in fine_tuning_source
    assert "launch_runpod_worker_for_current_training" in pretraining_source
    assert "launch_runpod_worker_for_current_training" in fine_tuning_source


def test_hidden_live_page_skips_expensive_snapshot_render(monkeypatch) -> None:
    app = interface_app.QApplication.instance() or interface_app.QApplication([])
    app.setProperty("license_valid", True)
    window = interface_app.MainWindow()
    renders = []
    monkeypatch.setattr(window, "_render_live_snapshot", lambda *args: renders.append(args))
    window.training_controller.request = fake_request(Path("output"))
    batch = TrainingTelemetryBatch(
        latest_metric={"id": 1, "step": 1, "train_loss": 1.25},
        new_events=(),
        metric_history=({"id": 1, "step": 1, "train_loss": 1.25},),
        ui_process_cpu_percent=1.0,
        total_metric_rows=1,
    )
    try:
        window.pages.setCurrentIndex(2)
        window._on_training_telemetry(batch)
        assert renders == []

        window.pages.setCurrentIndex(window.live_page_index)
        window._on_training_telemetry(batch)
        assert len(renders) == 1
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_durable_metric_updates_training_and_live_progress() -> None:
    app = interface_app.QApplication.instance() or interface_app.QApplication([])
    app.setProperty("license_valid", True)
    window = interface_app.MainWindow()
    try:
        window._on_training_telemetry(
            TrainingTelemetryBatch(
                latest_metric={"id": 1, "step": 5, "total_steps": 20},
                new_events=(),
                metric_history=({"id": 1, "step": 5, "total_steps": 20},),
                ui_process_cpu_percent=1.0,
                total_metric_rows=1,
            )
        )
        assert window.training_progress.value() == 25
        assert window.live_progress.value() == 25
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_validation_event_updates_durable_metric_snapshot() -> None:
    app = interface_app.QApplication.instance() or interface_app.QApplication([])
    app.setProperty("license_valid", True)
    window = interface_app.MainWindow()
    metric = {"id": 1, "step": 5, "total_steps": 20, "train_loss": 1.0}
    window.training_controller.metric_history.append(metric)
    try:
        window._on_training_telemetry(
            TrainingTelemetryBatch(
                latest_metric=metric,
                new_events=({
                    "id": 2,
                    "event_type": "validation",
                    "step": 5,
                    "val_loss": 0.5,
                    "message": "validation complete",
                },),
                metric_history=(metric,),
                ui_process_cpu_percent=1.0,
                total_metric_rows=1,
            )
        )
        assert window.training_val_metric.text() == "Val loss: 0.5000"
        assert window.training_controller.metric_history[-1]["val_loss"] == 0.5
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_reattach_state_rebinds_live_telemetry_database(tmp_path: Path) -> None:
    app = interface_app.QApplication.instance() or interface_app.QApplication([])
    app.setProperty("license_valid", True)
    window = interface_app.MainWindow()
    request = fake_request(tmp_path, "reattached-run")
    window.training_controller.request = request
    window.training_controller.total_metric_rows = 12
    window.training_controller.last_metric_row_id = 34
    try:
        window._on_training_process_state(
            TrainingProcessSnapshot(
                state="running",
                run_id=request.run_id,
                pid=1234,
                reattached=True,
            )
        )
        assert window.telemetry_db_path == request.telemetry_db_path
        assert window.telemetry_run_id == request.run_id
        assert window.telemetry_latest_index == 12
        assert window.telemetry_latest_id == 34
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_training_control_cleanup_does_not_mutate_qthread_task_state() -> None:
    app = interface_app.QApplication.instance() or interface_app.QApplication([])
    app.setProperty("license_valid", True)
    window = interface_app.MainWindow()
    shared_button = object()
    shared_stop_button = object()
    try:
        window.active_task_kind = "dataset"
        window.active_button = shared_button
        window.active_stop_button = shared_stop_button

        window._finish_training_controls()

        assert window.active_task_kind == "dataset"
        assert window.active_button is shared_button
        assert window.active_stop_button is shared_stop_button
    finally:
        window.active_button = None
        window.active_stop_button = None
        window.close()
        window.deleteLater()
        app.processEvents()


def test_fine_tune_eta_uses_eta_payload_without_elapsed_time() -> None:
    app = interface_app.QApplication.instance() or interface_app.QApplication([])
    app.setProperty("license_valid", True)
    window = interface_app.MainWindow()
    try:
        window._update_training_metrics(
            {"step": 3, "eta_seconds": 65},
            update_fine_tune=True,
            render_live=False,
        )
        assert window.fine_tune_eta_metric.text() == "ETA: 1m 05s"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_window_close_detaches_without_stop_or_wait(monkeypatch) -> None:
    app = interface_app.QApplication.instance() or interface_app.QApplication([])
    app.setProperty("license_valid", True)
    window = interface_app.MainWindow()
    detached = []
    stop_calls = []
    monkeypatch.setattr(
        window.training_controller,
        "detach",
        lambda: detached.append(True),
    )
    monkeypatch.setattr(
        window.training_controller,
        "request_stop",
        lambda: stop_calls.append(True),
    )
    event = QCloseEvent()
    try:
        window.closeEvent(event)
        assert detached == [True]
        assert stop_calls == []
        assert event.isAccepted()
    finally:
        window.deleteLater()
        app.processEvents()


def test_force_termination_rechecks_same_process_identity(monkeypatch) -> None:
    terminated = []

    class Process:
        def create_time(self):
            return 123.456

        def terminate(self):
            terminated.append(True)

    monkeypatch.setattr(process_control, "manifest_process_is_current", lambda _manifest: True)
    monkeypatch.setattr(process_control.psutil, "Process", lambda _pid: Process())
    process_control.terminate_verified_process(
        {
            "pid": 42,
            "process_identity": {
                "kind": "psutil-create-time",
                "value": "123.456000",
            },
        }
    )
    assert terminated == [True]


def test_force_termination_refuses_identity_change(monkeypatch) -> None:
    class Process:
        def create_time(self):
            return 999.0

        def terminate(self):
            raise AssertionError("identity mismatch must fail before termination")

    monkeypatch.setattr(process_control, "manifest_process_is_current", lambda _manifest: True)
    monkeypatch.setattr(process_control.psutil, "Process", lambda _pid: Process())
    with pytest.raises(RuntimeError, match="identity changed"):
        process_control.terminate_verified_process(
            {
                "pid": 42,
                "process_identity": {
                    "kind": "psutil-create-time",
                    "value": "123.456000",
                },
            }
        )
