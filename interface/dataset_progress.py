"""Bounded presentation helpers for dataset preparation progress events."""

from __future__ import annotations

from typing import Any


def apply_dataset_progress(
    owner: Any,
    event: dict[str, Any],
    log: Any,
    progress_bar: Any,
) -> bool:
    """Render a dataset-specific progress event when recognized."""

    event_type = str(event.get("event_type", "")).lower()
    if event_type == "dataset_diagnostic":
        diagnostic = event.get("diagnostic")
        diagnostic = diagnostic if isinstance(diagnostic, dict) else {}
        source = str(
            event.get("source_path")
            or diagnostic.get("path")
            or diagnostic.get("filename")
            or event.get("message")
            or "dataset source"
        )
        if source not in owner.dataset_diagnostic_sources:
            owner.dataset_diagnostic_sources.add(source)
            message = str(
                event.get("message")
                or diagnostic.get("summary")
                or f"{source}: invalid records detected."
            )
            level = (
                "ERROR"
                if str(event.get("level", "")).lower() == "error"
                else "WARN"
            )
            log.append(f"[{level}] {_bounded_message(message)}")
        _set_progress(progress_bar, event.get("percent"))
        return True

    message = str(event.get("message") or "")
    legacy_source = legacy_dataset_diagnostic_source(message)
    if legacy_source:
        if legacy_source not in owner.dataset_diagnostic_sources:
            owner.dataset_diagnostic_sources.add(legacy_source)
            log.append(
                f"[WARN] {legacy_source}: invalid records detected; "
                "additional per-record diagnostics suppressed."
            )
        _set_progress(progress_bar, event.get("percent"))
        return True

    outcome = str(event.get("outcome", "")).lower()
    if event_type == "completion" and outcome in {
        "completed",
        "completed_with_warnings",
    }:
        owner.active_task_terminal_event = dict(event)
        progress_bar.setRange(0, 100)
        progress_bar.setValue(100)
        if message:
            log.append(_bounded_message(message))
        if not owner.dataset_result_applied:
            owner.dataset_status.setText(
                "Dataset: prepared with warnings"
                if outcome == "completed_with_warnings"
                else "Dataset: prepared"
            )
        return True

    return False


def dataset_event_is_ui_only(event: dict[str, Any]) -> bool:
    """Return whether an event is rendered locally without external notification."""

    event_type = str(event.get("event_type", "")).lower()
    return event_type in {"dataset_diagnostic", "completion"} or bool(
        legacy_dataset_diagnostic_source(str(event.get("message") or ""))
    )


def dataset_failure_was_stopped(message: str) -> bool:
    """Return whether a worker failure message represents user cancellation."""

    lowered_message = message.lower()
    return any(
        marker in lowered_message
        for marker in (
            "stopped by user",
            "cancelled by user",
            "canceled by user",
            "task cancelled",
            "task canceled",
        )
    )


def dataset_terminal_percent(event: dict[str, Any] | None) -> int | None:
    """Return the authoritative progress value for a dataset terminal event."""

    if not event:
        return None
    event_type = str(event.get("event_type", "")).lower()
    outcome = str(event.get("outcome", "")).lower()
    if event_type == "completion" and outcome in {
        "completed",
        "completed_with_warnings",
    }:
        return 100
    if event_type in {"cancelled", "failure"}:
        return 0
    return None


def legacy_dataset_diagnostic_source(message: str) -> str:
    """Return the source prefix for legacy per-record ingestion messages."""

    text = message.removeprefix("Invalid record: ").strip()
    for marker in (", line ", ": line ", ", record ", ": record "):
        if marker in text:
            return text.split(marker, 1)[0].strip()
    return ""


def _bounded_message(message: str) -> str:
    """Normalize and bound a diagnostic message for one UI log entry."""

    return " ".join(message.split())[:800]


def _set_progress(progress_bar: Any, percent: object) -> None:
    """Apply an optional bounded dataset preparation percentage."""

    if percent is not None:
        progress_bar.setRange(0, 100)
        progress_bar.setValue(max(0, min(100, int(percent))))
