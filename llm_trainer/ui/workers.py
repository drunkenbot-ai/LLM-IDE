from __future__ import annotations

import logging
import multiprocessing as mp
import time
from queue import Queue
from threading import Event
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal


LOGGER = logging.getLogger(__name__)


def _process_worker_entry(
    fn: Any,
    args: tuple[Any, ...],
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
    stop_event: mp.Event,
    with_progress: bool,
) -> None:
    """Run a worker function inside a child process.

    Args:
        fn: Callable to run.
        args: Positional arguments.
        result_queue: Queue receiving the final result or failure.
        progress_queue: Queue receiving progress events.
        stop_event: Cross-process cooperative cancellation event.
        with_progress: Whether to pass progress and stop callbacks to ``fn``.
    """

    try:
        if with_progress:
            result = fn(
                *args,
                progress=lambda event: progress_queue.put(event),
                should_stop=stop_event.is_set,
            )
        else:
            result = fn(*args)
        result_queue.put(("finished", result))
    except Exception as exc:
        LOGGER.exception("Process worker failed while running %s", getattr(fn, "__name__", fn))
        result_queue.put(("failed", str(exc)))


class TaskWorker(QObject):
    """Background worker used for long-running UI tasks."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        fn: Any,
        *args: Any,
        progress_queue: Optional[Queue] = None,
        with_progress: bool = False,
        stop_event: Optional[Event] = None,
    ) -> None:
        """Create a worker.

        Args:
            fn: Callable to execute in the worker thread.
            *args: Positional arguments passed to ``fn``.
            progress_queue: Optional queue for progress events.
            with_progress: Whether to pass a progress callback to ``fn``.
            stop_event: Optional event used for cooperative cancellation.
        """

        super().__init__()
        self.fn = fn
        self.args = args
        self.progress_queue = progress_queue
        self.with_progress = with_progress
        self.stop_event = stop_event

    def run(self) -> None:
        """Execute the worker function and emit completion or failure."""

        try:
            if self.with_progress:
                self.finished.emit(self.fn(*self.args, progress=self._queue_progress, should_stop=self._should_stop))
            else:
                self.finished.emit(self.fn(*self.args))
        except Exception as exc:
            LOGGER.exception("Background worker failed while running %s", getattr(self.fn, "__name__", self.fn))
            self.failed.emit(str(exc))

    def _queue_progress(self, event: Any) -> None:
        if self.progress_queue is not None:
            self.progress_queue.put(event)

    def _should_stop(self) -> bool:
        """Return whether the active task has been asked to stop.

        Returns:
            True when the cooperative stop event is set.
        """

        return bool(self.stop_event and self.stop_event.is_set())


class ProcessTaskWorker(QObject):
    """Background worker that isolates heavy tasks in a child process."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        fn: Any,
        *args: Any,
        progress_queue: Optional[Queue] = None,
        with_progress: bool = False,
        stop_event: Optional[Event] = None,
    ) -> None:
        """Create a process-backed worker.

        Args:
            fn: Callable to execute in the child process.
            *args: Positional arguments passed to ``fn``.
            progress_queue: UI-thread progress queue.
            with_progress: Whether to pass progress callbacks to ``fn``.
            stop_event: Thread event used to request cancellation.
        """

        super().__init__()
        self.fn = fn
        self.args = args
        self.progress_queue = progress_queue
        self.with_progress = with_progress
        self.stop_event = stop_event

    def run(self) -> None:
        """Execute the worker function in a separate process."""

        context = mp.get_context("spawn")
        child_progress_queue: mp.Queue = context.Queue()
        result_queue: mp.Queue = context.Queue()
        child_stop_event: mp.Event = context.Event()
        process = context.Process(
            target=_process_worker_entry,
            args=(self.fn, self.args, result_queue, child_progress_queue, child_stop_event, self.with_progress),
            daemon=True,
        )
        process.start()
        stop_requested_at: Optional[float] = None
        try:
            while process.is_alive():
                if self.stop_event is not None and self.stop_event.is_set():
                    child_stop_event.set()
                    if stop_requested_at is None:
                        stop_requested_at = time.monotonic()
                    elif time.monotonic() - stop_requested_at > 5.0:
                        process.terminate()
                        break
                self._drain_child_progress(child_progress_queue)
                process.join(0.05)
            self._drain_child_progress(child_progress_queue)
            process.join()
            if stop_requested_at is not None and process.exitcode not in {0, None} and result_queue.empty():
                self.failed.emit("Dataset preparation stopped by user.")
                return
            if result_queue.empty():
                if process.exitcode == 0:
                    self.failed.emit("Process finished without returning a result.")
                else:
                    self.failed.emit(f"Process exited unexpectedly with code {process.exitcode}.")
                return
            status, payload = result_queue.get()
            if status == "finished":
                self.finished.emit(payload)
            else:
                self.failed.emit(str(payload))
        finally:
            if process.is_alive():
                child_stop_event.set()
                process.terminate()
                process.join(2)
            child_progress_queue.close()
            result_queue.close()

    def _drain_child_progress(self, child_progress_queue: mp.Queue) -> None:
        """Move child-process progress events into the UI progress queue.

        Args:
            child_progress_queue: Queue owned by the child process.
        """

        if self.progress_queue is None:
            return
        while True:
            try:
                event = child_progress_queue.get_nowait()
            except Exception:
                break
            self.progress_queue.put(event)
