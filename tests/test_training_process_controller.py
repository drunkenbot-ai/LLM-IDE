from __future__ import annotations

import inspect
import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QCloseEvent

from interface import app as interface_app
from interface import process_control
from interface.training_artifacts import select_training_artifacts
from interface.training_process_controller import (
    MAX_EVENT_HISTORY,
    MAX_METRIC_HISTORY,
    TrainingProcessController,
    TrainingProcessSnapshot,
    TrainingTelemetryBatch,
)

TaskRunnerMixin = interface_app.TaskRunnerMixin


class FakeSignal:
    def __init__(self) -> None:
        self.callback = None

    def connect(self, callback) -> None:
        self.callback = callback


class FakeTimer:
    def __init__(self) -> None:
        self.timeout = FakeSignal()
        self.interval = 0
        self.running = False

    def setInterval(self, milliseconds: int) -> None:
        self.interval = milliseconds

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


class FakeProcess:
    def __init__(self, pid: int = 1234, exit_code=None) -> None:
        self.pid = pid
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code


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


def manifest(request, status: str = "running", pid: int = 1234) -> dict:
    return {
        "schema": "drunkenbot.training-run-manifest",
        "version": 1,
        "run_id": request.run_id,
        "status": status,
        "pid": pid,
        "process_identity": {"kind": "test", "value": "identity"},
        "request_path": str(request.manifest_path.parent / "request.json"),
        "output_paths": {
            "output_dir": str(request.job.artifacts.output_dir),
            "telemetry_db": str(request.telemetry_db_path),
            "checkpoint": str(request.job.artifacts.output_dir / "final_model.pt"),
            "summary": str(request.job.artifacts.output_dir / "training_summary.json"),
        },
    }


def make_controller(tmp_path: Path, **overrides):
    request = overrides.pop("request", fake_request(tmp_path))
    timer = FakeTimer()
    states = []
    batches = []
    terminals = []
    errors = []
    process = overrides.pop("process", FakeProcess())
    controller = TrainingProcessController(
        timer,
        on_state=states.append,
        on_telemetry=batches.append,
        on_terminal=terminals.append,
        on_error=errors.append,
        create_request=overrides.pop("create_request", lambda *_args, **_kwargs: request),
        write_request=overrides.pop(
            "write_request",
            lambda path, _request: path,
        ),
        launch_process=overrides.pop("launch_process", lambda _path: process),
        load_request=overrides.pop("load_request", lambda _path: request),
        load_manifest=overrides.pop(
            "load_manifest",
            lambda _path: manifest(request),
        ),
        is_stale=overrides.pop("is_stale", lambda _manifest: False),
        process_is_current=overrides.pop("process_is_current", lambda _manifest: True),
        read_metrics=overrides.pop("read_metrics", lambda *_args, **_kwargs: []),
        read_events=overrides.pop("read_events", lambda *_args, **_kwargs: []),
        write_stop=overrides.pop("write_stop", lambda path, _run_id: path),
        ui_cpu_percent=lambda: 2.5,
        **overrides,
    )
    return controller, timer, states, batches, terminals, errors


def test_launch_persists_run_specific_request_without_qthread(tmp_path: Path) -> None:
    request = fake_request(tmp_path)
    writes = []
    controller, timer, states, *_ = make_controller(
        tmp_path,
        request=request,
        write_request=lambda path, value: writes.append((path, value)) or path,
    )

    launched = controller.launch(object(), notifier_config_path=tmp_path / "notifier.json")

    assert launched is request
    assert writes == [(request.manifest_path.parent / "request.json", request)]
    assert timer.running
    assert timer.interval == 750
    assert states[-1].run_id == request.run_id
    assert states[-1].pid == 1234
    assert "QThread" not in inspect.getsource(TrainingProcessController)


def test_poll_reads_incrementally_and_emits_terminal_once(tmp_path: Path) -> None:
    request = fake_request(tmp_path)
    request.manifest_path.parent.mkdir(parents=True)
    request.manifest_path.write_text("{}", encoding="utf-8")
    statuses = iter(("running", "completed", "completed"))
    metric_last_ids = []
    event_last_ids = []

    def read_metrics(_path, _run_id, last_row_id=0, **_kwargs):
        metric_last_ids.append(last_row_id)
        return (
            [{"id": 1, "step": 1, "train_loss": 2.0}, {"id": 2, "step": 2, "train_loss": 1.0}]
            if last_row_id == 0
            else []
        )

    def read_events(_path, _run_id, last_row_id=0, **_kwargs):
        event_last_ids.append(last_row_id)
        return (
            [{
                "id": 7,
                "event_type": "lifecycle",
                "message": "started",
                "payload_json": json.dumps({"percent": 5}),
            }]
            if last_row_id == 0
            else []
        )

    controller, _, states, batches, terminals, _ = make_controller(
        tmp_path,
        request=request,
        load_manifest=lambda _path: manifest(request, next(statuses)),
        read_metrics=read_metrics,
        read_events=read_events,
    )
    controller.request = request
    controller.manifest_path = request.manifest_path

    controller.poll()
    controller.poll()
    controller.poll()

    assert metric_last_ids == [0, 2, 2]
    assert event_last_ids == [0, 7, 7]
    assert batches[0].latest_metric["id"] == 2
    assert batches[0].new_events[0]["percent"] == 5
    assert states[-1].state == "completed"
    assert len(terminals) == 1


def test_worker_exit_two_is_visible_launch_refusal(tmp_path: Path) -> None:
    controller, timer, states, _, _, errors = make_controller(
        tmp_path,
        process=FakeProcess(exit_code=2),
    )
    controller.launch(object())
    controller.poll()

    assert not timer.running
    assert states[-1].state == "failed"
    assert "refused launch" in states[-1].message
    assert errors == [states[-1].message]


def test_worker_exit_after_manifest_reports_supervision_failure(tmp_path: Path) -> None:
    request = fake_request(tmp_path)
    request.manifest_path.parent.mkdir(parents=True)
    request.manifest_path.write_text("{}", encoding="utf-8")
    controller, timer, states, _, _, errors = make_controller(
        tmp_path,
        request=request,
        process=FakeProcess(exit_code=7),
        load_manifest=lambda _path: manifest(request, "running"),
    )
    controller.launch(object())

    controller.poll()

    assert states[-1].state == "failed"
    assert errors == ["Worker exited with code 7 before a terminal manifest update"]
    assert not timer.running


def test_detach_never_requests_stop(tmp_path: Path) -> None:
    stop_calls = []
    controller, timer, states, *_ = make_controller(
        tmp_path,
        write_stop=lambda *args: stop_calls.append(args),
    )
    controller.launch(object())

    controller.detach()

    assert not timer.running
    assert states[-1].state == "detached"
    assert stop_calls == []


def test_discovery_prefers_healthy_active_run_over_newer_terminal(tmp_path: Path) -> None:
    active = fake_request(tmp_path / "active", "active")
    complete = fake_request(tmp_path / "complete", "complete")
    for request in (active, complete):
        request.manifest_path.parent.mkdir(parents=True)
        request.manifest_path.write_text("{}", encoding="utf-8")
    manifests = {
        active.manifest_path: manifest(active, "running"),
        complete.manifest_path: manifest(complete, "completed"),
    }
    requests = {
        str(active.manifest_path.parent / "request.json"): active,
        str(complete.manifest_path.parent / "request.json"): complete,
    }
    controller, timer, states, *_ = make_controller(
        tmp_path,
        load_manifest=lambda path: manifests[path],
        load_request=lambda path: requests[str(path)],
    )

    assert controller.discover([active.job.artifacts.output_dir, complete.job.artifacts.output_dir])
    assert controller.request is active
    assert states[-1].state == "running"
    assert states[-1].reattached
    assert timer.running


def test_discovery_without_candidates_clears_previous_project_history(tmp_path: Path) -> None:
    controller, _, *_ = make_controller(tmp_path)
    controller.launch(object())
    controller.metric_history.append({"id": 1, "step": 1})
    controller.event_history.append({"id": 2, "event_type": "lifecycle"})

    assert not controller.discover([tmp_path / "other-project"])

    assert controller.request is None
    assert not controller.metric_history
    assert not controller.event_history
    assert controller.last_metric_row_id == 0
    assert controller.last_event_row_id == 0


def test_stale_run_does_not_remain_active(tmp_path: Path) -> None:
    request = fake_request(tmp_path)
    request.manifest_path.parent.mkdir(parents=True)
    request.manifest_path.write_text("{}", encoding="utf-8")
    controller, timer, states, *_ = make_controller(
        tmp_path,
        request=request,
        is_stale=lambda _manifest: True,
    )

    controller.attach(request.manifest_path)

    assert states[-1].state == "stale"
    assert not timer.running
    assert not controller.active


def test_reattached_stopping_run_restarts_guarded_force_timeout(tmp_path: Path) -> None:
    request = fake_request(tmp_path)
    request.manifest_path.parent.mkdir(parents=True)
    request.manifest_path.write_text("{}", encoding="utf-8")
    now = [0.0]
    controller, _, states, *_ = make_controller(
        tmp_path,
        request=request,
        monotonic=lambda: now[0],
        force_stop_timeout_seconds=5,
        load_manifest=lambda _path: manifest(request, "stopping"),
    )

    controller.attach(request.manifest_path)
    now[0] = 6.0
    controller.poll()

    assert states[-1].state == "stopping"
    assert states[-1].force_stop_available


def test_force_stop_fails_closed_for_reused_or_unverifiable_pid(tmp_path: Path) -> None:
    now = [0.0]
    current = [False]
    terminated = []
    controller, *_ = make_controller(
        tmp_path,
        monotonic=lambda: now[0],
        force_stop_timeout_seconds=5,
        process_is_current=lambda _manifest: current[0],
        terminate_process=terminated.append,
    )
    controller.launch(object())
    controller.manifest = manifest(controller.request)
    controller.request_stop()
    now[0] = 6.0

    with pytest.raises(RuntimeError, match="unverifiable"):
        controller.force_stop()
    assert terminated == []

    current[0] = True
    controller.force_stop()
    assert terminated == [controller.manifest]


def test_backlog_is_coalesced_and_histories_are_bounded(tmp_path: Path) -> None:
    metrics = [
        {"id": index, "step": index, "train_loss": 1.0 / index}
        for index in range(1, MAX_METRIC_HISTORY + 106)
    ]
    events = [
        {
            "id": index,
            "event_type": "warning",
            "message": f"warning {index}",
            "payload_json": "{}",
        }
        for index in range(1, MAX_EVENT_HISTORY + 51)
    ]
    controller, _, _, batches, *_ = make_controller(
        tmp_path,
        read_metrics=lambda *_args, **_kwargs: metrics,
        read_events=lambda *_args, **_kwargs: events,
    )
    controller.request = fake_request(tmp_path)

    controller._read_telemetry()

    assert len(batches) == 1
    assert batches[0].latest_metric["id"] == metrics[-1]["id"]
    assert len(batches[0].metric_history) == MAX_METRIC_HISTORY
    assert len(controller.event_history) == MAX_EVENT_HISTORY


def test_metric_payload_json_fields_are_available_to_ui(tmp_path: Path) -> None:
    controller, _, _, batches, *_ = make_controller(
        tmp_path,
        read_metrics=lambda *_args, **_kwargs: [{
            "id": 1,
            "step": 4,
            "payload_json": json.dumps({"eta_seconds": 65}),
        }],
    )
    controller.request = fake_request(tmp_path)

    controller._read_telemetry()

    assert batches[0].latest_metric["eta_seconds"] == 65


def test_terminal_backlog_drains_across_bounded_polls(tmp_path: Path) -> None:
    request = fake_request(tmp_path)
    request.manifest_path.parent.mkdir(parents=True)
    request.manifest_path.write_text("{}", encoding="utf-8")
    rows = [{"id": index, "step": index} for index in range(1, 1002)]

    def read_metrics(_path, _run_id, last_row_id=0, limit=1000):
        return [row for row in rows if row["id"] > last_row_id][:limit]

    controller, timer, _, batches, terminals, _ = make_controller(
        tmp_path,
        request=request,
        load_manifest=lambda _path: manifest(request, "completed"),
        read_metrics=read_metrics,
    )

    controller.attach(request.manifest_path)
    assert timer.running
    assert len(terminals) == 0
    assert batches[-1].latest_metric["id"] == 1000

    controller.poll()

    assert not timer.running
    assert batches[-1].latest_metric["id"] == 1001
    assert len(terminals) == 1


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
