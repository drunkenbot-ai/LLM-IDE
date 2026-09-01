from __future__ import annotations

from queue import Queue
from types import SimpleNamespace

import pytest

from interface import app as interface_app


class FakeLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(self, message: str) -> None:
        self.lines.append(message)


class FakeProgress:
    def __init__(self) -> None:
        self.minimum = 0
        self.maximum = 0
        self.current = 0

    def setRange(self, minimum: int, maximum: int) -> None:
        self.minimum = minimum
        self.maximum = maximum

    def setValue(self, value: int) -> None:
        self.current = value


class FakeLabel:
    def __init__(self) -> None:
        self.value = ""

    def setText(self, value: str) -> None:
        self.value = value


class FakeTimer:
    def __init__(self) -> None:
        self.stopped = False

    def isActive(self) -> bool:
        return True

    def stop(self) -> None:
        self.stopped = True


class DatasetTaskHarness(interface_app.TaskRunnerMixin):
    def __init__(self, events=()) -> None:
        self.progress_queue = Queue()
        for event in events:
            self.progress_queue.put(event)
        self.active_log = FakeLog()
        self.active_progress_bar = FakeProgress()
        self.training_log = object()
        self.active_task_kind = "dataset"
        self.active_task_terminal_event = None
        self.dataset_diagnostic_sources: set[str] = set()
        self.dataset_result_applied = False
        self.dataset_status = FakeLabel()
        self.project_state = FakeLabel()
        self.notification_manager = None
        self.progress_timer = FakeTimer()
        self.thread = object()
        self.worker = object()
        self.result_bridge = None
        self.stop_event = object()
        self.active_stop_button = None
        self.active_button = object()
        self.button_cleared = False

    def _clear_button_busy(self, final_text=None) -> None:
        self.button_cleared = True
        self.active_button = None


class DatasetResultHarness:
    def __init__(self) -> None:
        self.dataset_progress = FakeProgress()
        self.auto_vocab_label = FakeLabel()
        self.dataset_log = FakeLog()
        self.train_data_dir = FakeLabel()
        self.project_state = FakeLabel()
        self.dataset_status = FakeLabel()
        self.quality_summary = None
        self.notifications = []
        self.button_text = ""

    def _update_dataset_quality_report(self, summary) -> None:
        self.quality_summary = summary

    def refresh_model_estimate(self) -> None:
        pass

    def refresh_fine_tune_workflow(self) -> None:
        pass

    def _notify_complete(self, stage, title, lines) -> None:
        self.notifications.append((stage, title, lines))

    def _clear_button_busy(self, final_text=None) -> None:
        self.button_text = final_text


def grouped_diagnostic(invalid_record_count: int = 30_000) -> dict:
    return {
        "event_type": "dataset_diagnostic",
        "level": "warning",
        "outcome": "partial",
        "source_path": "/data/educational_instructions.jsonl",
        "message": (
            "educational_instructions.jsonl: 30,000 invalid records were skipped; "
            "valid records were retained."
        ),
        "percent": 45,
        "diagnostic": {
            "filename": "educational_instructions.jsonl",
            "location_kind": "line",
            "invalid_record_count": invalid_record_count,
            "location_ranges": [
                {"start": index * 10 + 1, "end": index * 10 + 1}
                for index in range(12)
            ],
            "omitted_location_count": invalid_record_count - 12,
            "summary": "30,000 invalid rows in 12 shown ranges.",
        },
    }


def completed_with_warnings() -> dict:
    return {
        "event_type": "completion",
        "outcome": "completed_with_warnings",
        "percent": 100,
        "partial_file_count": 1,
        "failed_file_count": 0,
        "invalid_record_count": 30_000,
        "message": "Dataset preparation completed with warnings.",
    }


def dataset_result() -> SimpleNamespace:
    return SimpleNamespace(
        output_dir="/project/dataset",
        tokenizer_path="/project/dataset/tokenizer.json",
        document_count=120,
        token_count=45_000,
        vocab_size=8_000,
        character_count=180_000,
        suggested_vocab_size=8_000,
        train_window_count=350,
        val_window_count=40,
        sequence_token_stats={},
        warning=None,
        code_sample_count=0,
        prose_sample_count=120,
        conversation_sample_count=0,
        cached_file_count=0,
        processed_file_count=1,
        partial_file_count=1,
        skipped_file_count=0,
        failed_file_count=0,
        invalid_record_count=30_000,
        preparation_outcome="completed_with_warnings",
        dataset_version_id="dataset-v7",
        dataset_version_number=7,
        duplicate_block_count=0,
        unique_block_count=120,
        corpus_block_count=120,
        duplicate_block_ratio=0.0,
        unique_block_ratio=1.0,
    )


def test_grouped_large_diagnostic_logs_once_and_thread_finish_keeps_terminal_state() -> None:
    diagnostic = grouped_diagnostic()
    harness = DatasetTaskHarness(
        [
            {"message": "Finalizing dataset.", "percent": 98},
            diagnostic,
            diagnostic,
            completed_with_warnings(),
        ]
    )
    log = harness.active_log
    progress = harness.active_progress_bar

    harness._thread_finished()

    diagnostic_lines = [
        line for line in log.lines
        if "educational_instructions.jsonl" in line
    ]
    assert harness.active_log is None
    assert len(harness.dataset_diagnostic_sources) == 0
    assert harness.active_progress_bar is None
    assert harness.active_task_terminal_event["outcome"] == "completed_with_warnings"
    assert harness.dataset_status.value == "Dataset: prepared with warnings"
    assert progress.current == 100
    assert len(diagnostic_lines) == 1
    assert len(log.lines) == 3
    assert log.lines[-1] == "Dataset preparation completed with warnings."
    assert harness.progress_timer.stopped
    assert harness.button_cleared


def test_grouped_large_diagnostic_is_bounded_without_expanding_ranges() -> None:
    harness = DatasetTaskHarness()
    event = grouped_diagnostic()

    harness._handle_progress(event, harness.active_log, harness.active_progress_bar)
    harness._handle_progress(event, harness.active_log, harness.active_progress_bar)
    harness._handle_progress(
        completed_with_warnings(), harness.active_log, harness.active_progress_bar
    )

    assert len(harness.active_log.lines) == 2
    assert "30,000 invalid records" in harness.active_log.lines[0]
    assert "location_ranges" not in "\n".join(harness.active_log.lines)
    assert "line 121" not in "\n".join(harness.active_log.lines)
    assert len(harness.active_log.lines[0]) <= 807
    assert harness.active_progress_bar.current == 100
    assert harness.active_progress_bar.maximum == 100


def test_unusable_source_diagnostic_is_visibly_an_error() -> None:
    harness = DatasetTaskHarness()
    event = grouped_diagnostic()
    event.update(level="error", outcome="failed")

    harness._handle_progress(event, harness.active_log, harness.active_progress_bar)

    assert harness.active_log.lines[0].startswith("[ERROR]")


def test_late_completion_event_does_not_overwrite_applied_result_status() -> None:
    harness = DatasetTaskHarness()
    harness.dataset_result_applied = True
    harness.dataset_status.setText("Dataset: 120 valid, 30,000 invalid record(s)")

    harness._handle_progress(
        completed_with_warnings(), harness.active_log, harness.active_progress_bar
    )

    assert harness.dataset_status.value == "Dataset: 120 valid, 30,000 invalid record(s)"
    assert harness.active_progress_bar.current == 100


def test_completed_with_warnings_applies_result_and_returns_idle() -> None:
    harness = DatasetResultHarness()

    interface_app.DatasetScreenMixin._dataset_finished(harness, dataset_result())

    assert harness.dataset_progress.current == 100
    assert harness.dataset_progress.maximum == 100
    assert harness.train_data_dir.value == "/project/dataset"
    assert harness.project_state.value == "Dataset ready with warnings"
    assert "30,000 invalid record(s)" in harness.dataset_status.value
    assert any("Dataset version: dataset-v7" in line for line in harness.dataset_log.lines)
    warning_lines = [
        line for line in harness.dataset_log.lines
        if line.startswith("[WARN] Dataset preparation completed")
    ]
    assert len(warning_lines) == 1
    assert len(warning_lines[0]) < 240
    assert harness.quality_summary["invalid_record_count"] == 30_000
    assert harness.quality_summary["preparation_outcome"] == "completed_with_warnings"
    assert harness.button_text == "DataSet Prepared"


@pytest.mark.parametrize(
    ("message", "expected_status", "expected_event", "expected_prefix"),
    [
        (
            "Malformed JSON in source.jsonl at line 4",
            "Dataset: preparation failed",
            "failure",
            "Error:",
        ),
        (
            "Dataset preparation stopped by user",
            "Dataset: preparation stopped",
            "cancelled",
            "Stopped:",
        ),
    ],
)
def test_dataset_failure_and_cancel_restore_determinate_idle_state(
    message: str,
    expected_status: str,
    expected_event: str,
    expected_prefix: str,
) -> None:
    harness = DatasetTaskHarness()
    harness.progress_queue.put(grouped_diagnostic())
    progress = harness.active_progress_bar
    log = harness.active_log

    harness._task_failed_from_worker(message)
    harness._thread_finished()

    assert harness.dataset_status.value == expected_status
    assert harness.active_task_terminal_event["event_type"] == expected_event
    assert progress.minimum == 0
    assert progress.maximum == 100
    assert progress.current == 0
    assert any(line.startswith(expected_prefix) for line in log.lines)
    assert harness.button_cleared
