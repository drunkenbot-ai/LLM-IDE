"""Timer-driven supervision for durable standalone training workers."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from engine.contracts import TrainingJobSpec
from engine.telemetry_store import event_rows_after, metric_rows_after
from engine.training_worker_protocol import (
    StandaloneTrainingRequest,
    create_worker_request,
    launch_worker_process,
    load_run_manifest,
    load_worker_request,
    manifest_is_stale,
    manifest_process_is_current,
    write_stop_request,
    write_worker_request,
)
from interface.process_control import terminate_verified_process

try:
    import psutil
except ImportError:
    psutil = None


LOGGER = logging.getLogger("interface.training_process_controller")
POLL_INTERVAL_MS = 750
FORCE_STOP_TIMEOUT_SECONDS = 10.0
MAX_CONSECUTIVE_MANIFEST_ERRORS = 5
MAX_METRIC_HISTORY = 2000
MAX_EVENT_HISTORY = 1000
TERMINAL_STATES = {"completed", "stopped", "failed"}


class SignalLike(Protocol):
    """Minimal Qt signal surface used by the controller."""
    def connect(self, callback: Callable[[], None]) -> None:
        """Connect a timeout callback."""


class Timer(Protocol):
    """Minimal timer surface supported by QTimer and unit-test fakes."""
    timeout: SignalLike

    def setInterval(self, milliseconds: int) -> None:
        """Set the polling interval."""
    def start(self) -> None:
        """Start polling."""

    def stop(self) -> None:
        """Stop polling."""


class WorkerProcess(Protocol):
    """Minimal subprocess handle used for early-exit detection."""
    pid: int

    def poll(self) -> int | None:
        """Return the process exit code when it has exited."""


@dataclass(frozen=True)
class TrainingProcessSnapshot:
    """One coalesced process-state update for the UI."""

    state: str
    run_id: str = ""
    pid: int | None = None
    message: str = ""
    force_stop_available: bool = False
    reattached: bool = False


@dataclass(frozen=True)
class TrainingTelemetryBatch:
    """Incremental telemetry rows plus the latest renderable snapshot."""

    latest_metric: dict[str, Any] | None
    new_events: tuple[dict[str, Any], ...]
    metric_history: tuple[dict[str, Any], ...]
    ui_process_cpu_percent: float | None
    total_metric_rows: int


@dataclass(frozen=True)
class TrainingTerminal:
    """Durable terminal state and its latest persisted telemetry."""

    manifest: dict[str, Any]
    request: StandaloneTrainingRequest | None
    latest_metric: dict[str, Any] | None
    latest_event: dict[str, Any] | None


class TrainingProcessController:
    """Supervise a standalone engine worker from the Qt event loop."""

    def __init__(
        self,
        timer: Timer,
        *,
        on_state: Callable[[TrainingProcessSnapshot], None],
        on_telemetry: Callable[[TrainingTelemetryBatch], None],
        on_terminal: Callable[[TrainingTerminal], None],
        on_error: Callable[[str], None],
        launch_process: Callable[[Path], WorkerProcess] = launch_worker_process,
        create_request: Callable[..., StandaloneTrainingRequest] = create_worker_request,
        write_request: Callable[[Path, StandaloneTrainingRequest], Path] = write_worker_request,
        load_request: Callable[[Path], StandaloneTrainingRequest] = load_worker_request,
        load_manifest: Callable[[Path], dict[str, Any]] = load_run_manifest,
        is_stale: Callable[[dict[str, Any]], bool] = manifest_is_stale,
        process_is_current: Callable[[dict[str, Any]], bool] = manifest_process_is_current,
        read_metrics: Callable[..., list[Any]] = metric_rows_after,
        read_events: Callable[..., list[Any]] = event_rows_after,
        write_stop: Callable[[Path, str], Path] = write_stop_request,
        terminate_process: Callable[[dict[str, Any]], None] = terminate_verified_process,
        monotonic: Callable[[], float] = time.monotonic,
        ui_cpu_percent: Callable[[], float | None] | None = None,
        force_stop_timeout_seconds: float = FORCE_STOP_TIMEOUT_SECONDS,
        max_manifest_retries: int = MAX_CONSECUTIVE_MANIFEST_ERRORS,
    ) -> None:
        """Configure protocol functions, callbacks, and bounded state."""
        self.timer = timer
        self.on_state = on_state
        self.on_telemetry = on_telemetry
        self.on_terminal = on_terminal
        self.on_error = on_error
        self._launch_process = launch_process
        self._create_request = create_request
        self._write_request = write_request
        self._load_request = load_request
        self._load_manifest = load_manifest
        self._is_stale = is_stale
        self._process_is_current = process_is_current
        self._read_metrics = read_metrics
        self._read_events = read_events
        self._write_stop = write_stop
        self._terminate_process = terminate_process
        self._monotonic = monotonic
        self._ui_cpu_percent = ui_cpu_percent or self._default_ui_cpu_percent
        self.force_stop_timeout_seconds = max(0.0, force_stop_timeout_seconds)
        self.max_manifest_retries = max(1, int(max_manifest_retries))
        self._consecutive_manifest_errors = 0

        self.request: StandaloneTrainingRequest | None = None
        self.request_path: Path | None = None
        self.manifest_path: Path | None = None
        self.process: WorkerProcess | None = None
        self.manifest: dict[str, Any] | None = None
        self.state = "detached"
        self.reattached = False
        self.last_metric_row_id = 0
        self.last_event_row_id = 0
        self.total_metric_rows = 0
        self.metric_history: deque[dict[str, Any]] = deque(maxlen=MAX_METRIC_HISTORY)
        self.event_history: deque[dict[str, Any]] = deque(maxlen=MAX_EVENT_HISTORY)
        self.stop_requested_at: float | None = None
        self._terminal_emitted = False
        self._last_snapshot: TrainingProcessSnapshot | None = None
        self._ui_process = psutil.Process(os.getpid()) if psutil is not None else None

        self.timer.setInterval(POLL_INTERVAL_MS)
        self.timer.timeout.connect(self.poll)

    @property
    def active(self) -> bool:
        """Return whether an attached worker still needs supervision."""
        return bool(self.request and self.state not in TERMINAL_STATES | {"stale", "detached"})

    def launch(
        self,
        job: TrainingJobSpec,
        *,
        notifier_config_path: Path | None = None,
        run_id: str | None = None,
    ) -> StandaloneTrainingRequest:
        """Persist and launch a new worker request."""
        if self.active:
            raise RuntimeError(f"Training run {self.request.run_id} is already active")
        request = self._create_request(
            job,
            run_id=run_id,
            notifier_config_path=notifier_config_path,
        )
        request_path = request.manifest_path.parent / "request.json"
        self._write_request(request_path, request)
        self._reset_for_request(request, request_path, reattached=False)
        self._set_state("starting", message="Launching standalone training worker")
        try:
            self.process = self._launch_process(request_path)
        except Exception:
            self._set_state("failed", message="Standalone training worker could not be launched")
            raise
        self._emit_state(message="Waiting for worker manifest")
        self.timer.start()
        return request

    def discover(self, output_dirs: Iterable[Path]) -> bool:
        """Attach to the active run, or otherwise the latest durable run."""
        self._reset_for_request(None, None, reattached=False)
        self._set_state("detached", message="No training run attached")
        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        errors: list[str] = []
        for output_dir in output_dirs:
            runs_dir = Path(output_dir) / "training_runs"
            if not runs_dir.exists():
                continue
            for manifest_path in runs_dir.glob("*/manifest.json"):
                try:
                    manifest = self._load_manifest(manifest_path)
                    timestamp = manifest_path.stat().st_mtime
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{manifest_path}: {exc}")
                    continue
                candidates.append((timestamp, manifest_path, manifest))
        if not candidates:
            if errors:
                self._set_state(
                    "failed", message=f"Could not read training run metadata: {errors[0]}"
                )
                self.on_error(f"Malformed or incompatible training manifest: {errors[0]}")
            return False

        healthy_active = [
            item
            for item in candidates
            if item[2].get("status") not in TERMINAL_STATES and not self._is_stale(item[2])
        ]
        _, manifest_path, manifest = max(healthy_active or candidates, key=lambda item: item[0])
        self.attach(manifest_path, manifest=manifest)
        return True

    def attach(
        self,
        manifest_path: Path,
        *,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        """Safely reattach to one durable run manifest."""
        manifest_path = Path(manifest_path)
        durable_manifest = manifest or self._load_manifest(manifest_path)
        request_path_text = str(durable_manifest.get("request_path") or "").strip()
        if not request_path_text:
            self._set_state("failed", message="Training manifest has no request path")
            self.on_error("Training manifest has no request path; safe reattach is unavailable")
            return
        request_path = Path(request_path_text)
        try:
            request = self._load_request(request_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._set_state("failed", message=f"Could not load training request: {exc}")
            self.on_error(f"Could not load training request for reattach: {exc}")
            return
        if request is not None and request.run_id != durable_manifest.get("run_id"):
            raise ValueError("Training request and manifest run IDs do not match")

        self._reset_for_request(request, request_path, reattached=True)
        self.manifest_path = manifest_path
        self.manifest = durable_manifest
        status = str(durable_manifest.get("status") or "failed")
        if status == "stopping":
            self.stop_requested_at = self._monotonic()
        if status not in TERMINAL_STATES and self._is_stale(durable_manifest):
            self._set_state("stale", message="Heartbeat or process identity is stale")
            return
        self._set_state(status, message="Reattached to durable training run")
        backlog_possible = self._read_telemetry()
        if status in TERMINAL_STATES:
            if backlog_possible:
                self.timer.start()
            else:
                self._emit_terminal()
        else:
            self.timer.start()

    def poll(self) -> None:
        """Poll manifest and telemetry once at the bounded UI cadence."""
        if self.manifest_path is None:
            return
        if self.manifest_path.exists():
            try:
                self.manifest = self._load_manifest(self.manifest_path)
                self._consecutive_manifest_errors = 0
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._consecutive_manifest_errors += 1
                if self._consecutive_manifest_errors < self.max_manifest_retries:
                    LOGGER.warning(
                        "Transient error reading training manifest (%d/%d): %s",
                        self._consecutive_manifest_errors,
                        self.max_manifest_retries,
                        exc,
                    )
                    return
                self._set_state("failed", message=f"Malformed or incompatible manifest: {exc}")
                self.on_error(f"Malformed or incompatible training manifest: {exc}")
                self.timer.stop()
                return
        else:
            exit_code = self.process.poll() if self.process is not None else None
            if exit_code is not None:
                message = (
                    "Worker refused launch because this durable run is already active or terminal"
                    if exit_code == 2
                    else f"Worker exited with code {exit_code} before creating its manifest"
                )
                self._set_state("failed", message=message)
                self.on_error(message)
                self.timer.stop()
            return

        status = str(self.manifest.get("status") or "failed")
        if status not in TERMINAL_STATES and self._is_stale(self.manifest):
            self._set_state("stale", message="Heartbeat or process identity is stale")
            self.timer.stop()
            return

        backlog_possible = self._read_telemetry()
        message = ""
        force_available = False
        if self.stop_requested_at is not None and status not in TERMINAL_STATES:
            elapsed = self._monotonic() - self.stop_requested_at
            force_available = (
                elapsed >= self.force_stop_timeout_seconds
                and self._process_is_current(self.manifest)
            )
            if elapsed >= self.force_stop_timeout_seconds and not force_available:
                message = "Force stop unavailable because process identity cannot be verified"
        self._set_state(status, message=message, force_stop_available=force_available)

        if status in TERMINAL_STATES:
            if not backlog_possible:
                self.timer.stop()
                self._emit_terminal()
            return
        if self.process is not None:
            exit_code = self.process.poll()
            if exit_code is not None:
                message = (
                    f"Worker exited with code {exit_code} before a terminal manifest update"
                )
                self._set_state("failed", message=message)
                self.on_error(message)
                self.timer.stop()

    def request_stop(self) -> Path:
        """Write the cooperative, run-specific durable stop request."""
        if self.request is None:
            raise RuntimeError("No verified training request is attached")
        if self.state in TERMINAL_STATES | {"stale", "detached"}:
            raise RuntimeError(f"Cannot stop training while state is {self.state}")
        path = self._write_stop(self.request.control_path, self.request.run_id)
        self.stop_requested_at = self._monotonic()
        self._set_state("stopping", message="Cooperative stop requested")
        return path

    def force_stop(self) -> None:
        """Terminate after timeout only when process creation identity still matches."""
        if self.stop_requested_at is None:
            raise RuntimeError("Cooperative stop must be requested before force stop")
        if self._monotonic() - self.stop_requested_at < self.force_stop_timeout_seconds:
            raise RuntimeError("Force stop timeout has not elapsed")
        if self.manifest is None or not self._process_is_current(self.manifest):
            raise RuntimeError("Training process identity is unverifiable; refusing force stop")
        self._terminate_process(self.manifest)
        self._set_state("stopping", message="Verified training process termination requested")

    def detach(self) -> None:
        """Stop UI supervision without stopping or waiting for the worker."""
        self.timer.stop()
        if self.request is not None and self.state not in TERMINAL_STATES:
            self._set_state("detached", message="UI detached; standalone training continues")

    def _reset_for_request(
        self,
        request: StandaloneTrainingRequest | None,
        request_path: Path | None,
        *,
        reattached: bool,
    ) -> None:
        self.timer.stop()
        self.request = request
        self.request_path = Path(request_path) if request_path else None
        self.manifest_path = request.manifest_path if request is not None else None
        self.process = None
        self.manifest = None
        self.reattached = reattached
        self.last_metric_row_id = 0
        self.last_event_row_id = 0
        self.total_metric_rows = 0
        self.metric_history.clear()
        self.event_history.clear()
        self.stop_requested_at = None
        self._terminal_emitted = False
        self._last_snapshot = None
        self._consecutive_manifest_errors = 0

    def _read_telemetry(self) -> bool:
        if self.request is None:
            return False
        try:
            metric_rows = self._read_metrics(
                self.request.telemetry_db_path,
                self.request.run_id,
                last_row_id=self.last_metric_row_id,
                limit=1000,
            )
            event_rows = self._read_events(
                self.request.telemetry_db_path,
                self.request.run_id,
                last_row_id=self.last_event_row_id,
                limit=1000,
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            self.on_error(f"Could not read training telemetry: {exc}")
            return False

        metrics = [self._telemetry_payload(dict(row)) for row in metric_rows]
        events = [self._event_payload(dict(row)) for row in event_rows]
        if metrics:
            self.last_metric_row_id = int(metrics[-1]["id"])
            self.total_metric_rows += len(metrics)
            self.metric_history.extend(metrics)
        if events:
            self.last_event_row_id = int(events[-1]["id"])
            self.event_history.extend(events)
        if metrics or events:
            self.on_telemetry(
                TrainingTelemetryBatch(
                    latest_metric=metrics[-1] if metrics else None,
                    new_events=tuple(events),
                    metric_history=tuple(self.metric_history),
                    ui_process_cpu_percent=self._ui_cpu_percent(),
                    total_metric_rows=self.total_metric_rows,
                )
            )
        return len(metrics) == 1000 or len(events) == 1000

    @staticmethod
    def _telemetry_payload(row: dict[str, Any]) -> dict[str, Any]:
        try:
            payload = json.loads(str(row.get("payload_json") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return {**payload, **row}

    _event_payload = _telemetry_payload

    def _emit_terminal(self) -> None:
        if self._terminal_emitted or self.manifest is None:
            return
        self._terminal_emitted = True
        self.on_terminal(
            TrainingTerminal(
                manifest=dict(self.manifest),
                request=self.request,
                latest_metric=self.metric_history[-1] if self.metric_history else None,
                latest_event=self.event_history[-1] if self.event_history else None,
            )
        )

    def _set_state(
        self,
        state: str,
        *,
        message: str = "",
        force_stop_available: bool = False,
    ) -> None:
        self.state = state
        self._emit_state(message=message, force_stop_available=force_stop_available)

    def _emit_state(
        self,
        *,
        message: str = "",
        force_stop_available: bool = False,
    ) -> None:
        run_id = (
            self.request.run_id
            if self.request is not None
            else str((self.manifest or {}).get("run_id") or "")
        )
        pid = (self.manifest or {}).get("pid")
        if not isinstance(pid, int) and self.process is not None:
            pid = self.process.pid
        snapshot = TrainingProcessSnapshot(
            state=self.state,
            run_id=run_id,
            pid=pid if isinstance(pid, int) else None,
            message=message,
            force_stop_available=force_stop_available,
            reattached=self.reattached,
        )
        if snapshot != self._last_snapshot:
            self._last_snapshot = snapshot
            self.on_state(snapshot)

    def _default_ui_cpu_percent(self) -> float | None:
        if self._ui_process is None:
            return None
        try:
            return float(self._ui_process.cpu_percent(interval=None))
        except psutil.Error:
            return None
