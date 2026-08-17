from __future__ import annotations

# TaskRunnerMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class TaskRunnerMixin:
    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        """Set combo text when the value exists.

        Args:
            combo: Combo box to update.
            text: Display text to select.
        """

        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
        elif combo.isEditable():
            combo.setEditText(text)

    def _set_combo_by_data(self, combo: QComboBox, value: str, labels: dict[str, str]) -> None:
        """Set a combo by internal saved value.

        Args:
            combo: Combo box to update.
            value: Internal saved value.
            labels: Mapping from saved value to display label.
        """

        self._set_combo_text(combo, labels.get(value, value))

    def _run_task(
        self,
        fn,
        args,
        on_finished,
        log: QTextEdit,
        progress_bar: QProgressBar,
        with_progress: bool = False,
        button: Optional[QPushButton] = None,
        stop_button: Optional[QPushButton] = None,
        busy_text: str = "Working",
        task_kind: str = "",
        isolate_process: bool = False,
    ) -> None:
        """Run a long task on a background thread.

        Args:
            fn: Callable to execute.
            args: Positional arguments for the callable.
            on_finished: Slot called with the task result.
            log: Log widget receiving progress messages.
            progress_bar: Progress bar receiving percent updates.
            with_progress: Whether to pass a progress callback to the task.
            button: Optional button to disable while running.
            stop_button: Optional stop button to enable while running.
            busy_text: Button text shown while running.
            task_kind: Optional notification stage key.
            isolate_process: Run the task inside a child process.
        """

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please wait for the current task to finish.")
            return

        LOGGER.info("Starting background task: %s", getattr(fn, "__name__", str(fn)))
        self.active_task_kind = task_kind
        if button:
            self._set_button_busy(button, busy_text)
        if stop_button:
            stop_button.setEnabled(True)
            self.active_stop_button = stop_button

        self.stop_event = Event()
        self.progress_queue = Queue()
        self.active_log = log
        self.active_progress_bar = progress_bar
        self.thread = QThread(self)
        worker_class = ProcessTaskWorker if isolate_process else TaskWorker
        self.worker = worker_class(
            fn,
            *args,
            progress_queue=self.progress_queue,
            with_progress=with_progress,
            stop_event=self.stop_event,
        )
        self.result_bridge = WorkerSignalBridge(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.result_bridge.finished)
        self.result_bridge.finished.connect(on_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.result_bridge.failed)
        self.result_bridge.failed.connect(self._task_failed_from_worker)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.progress_timer.start(100)
        self.thread.start()

    @Slot(str)
    def _task_failed_from_worker(self, message: str) -> None:
        """Handle a worker failure on the UI thread.

        Args:
            message: Error message emitted by the worker.
        """

        if self.active_log is None or self.active_progress_bar is None:
            return
        LOGGER.error("Background task failed: %s", message)
        if self.active_task_kind == "chat":
            self.chat_status.setText(f"Chat: load failed - {message}")
        elif self.active_task_kind == "dataset_download":
            self.external_dataset_version.setText(f"Download failed: {message}")
            self.dataset_plan_progress.setVisible(False)
        self._task_failed(message, self.active_log, self.active_progress_bar)

    def stop_active_task(self) -> None:
        """Request a graceful stop for the active background task."""

        if self.stop_event is None:
            return
        LOGGER.info("Stop requested for active background task")
        self.stop_event.set()
        self._notify_failure("Stop requested", "The task is stopping at the next safe point.")
        if self.active_log is not None:
            self.active_log.append("Stop requested. Finishing the current safe point...")
        if self.active_stop_button is not None:
            self.active_stop_button.setEnabled(False)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                LOGGER.exception(
                    "Failed to empty CUDA cache in _thread_finished")

    @Slot()
    def request_shutdown_from_signal(self) -> None:
        """Handle Ctrl+C from a terminal without leaving Qt threads wedged."""

        self.interrupt_count += 1
        if self.interrupt_count > 1:
            os._exit(130)
        if self.stop_event is not None:
            self.stop_event.set()
        if self.active_log is not None:
            self.active_log.append("Interrupt received. Requesting stop...")
        self.project_state.setText("Stopping")
        if self.thread is None:
            QApplication.quit()
            return
        QTimer.singleShot(3000, lambda: os._exit(130) if self.thread is not None else QApplication.quit())

    def closeEvent(self, event: Any) -> None:
        """Clean up background services before the window closes.

        Args:
            event: Qt close event.
        """

        if self.thread is not None:
            if self.stop_event is not None:
                self.stop_event.set()
            if self.active_log is not None:
                self.active_log.append("Close requested. Stopping active task first...")
            self.project_state.setText("Stopping")
            LOGGER.info("Close requested while background task is running; waiting for task shutdown")
            event.ignore()
            QTimer.singleShot(500, self.close)
            return
        if self.coordinator_server is not None:
            self.stop_coordinator_server()
        super().closeEvent(event)

    def _handle_progress(self, event: object, log: QTextEdit, progress_bar: QProgressBar) -> None:
        """Apply one progress event to UI widgets.

        Args:
            event: Progress dictionary or message.
            log: Log widget to append messages to.
            progress_bar: Progress bar to update.
        """

        if isinstance(event, dict):
            if event.get("type") == "chat_delta":
                self._apply_chat_delta(event)
                return
            message = event.get("message")
            percent = event.get("percent")
            if log in (self.training_log, getattr(self, "fine_tune_log", None)):
                self._update_training_metrics(event, update_fine_tune=log is getattr(self, "fine_tune_log", None))
            if message:
                log.append(str(message))
                if log in (self.training_log, getattr(self, "fine_tune_log", None)) and hasattr(self, "live_log"):
                    self.live_log.append(str(message))
            if percent is not None:
                progress_bar.setValue(max(0, min(100, int(percent))))
                if log in (self.training_log, getattr(self, "fine_tune_log", None)) and hasattr(self, "live_progress"):
                    self.live_progress.setValue(max(0, min(100, int(percent))))
        else:
            log.append(str(event))

    def _notify_progress(self, event: dict[str, Any]) -> None:
        """Send throttled external progress notifications for long tasks.

        Args:
            event: Progress event emitted by a worker.
        """

        if not self.active_task_kind or self.notification_manager is None:
            return
        if self.active_task_kind not in {"dataset", "training", "fine_tune"}:
            return
        title = {
            "dataset": "Dataset preparation",
            "training": "Model training",
            "fine_tune": "Fine-tuning",
        }[self.active_task_kind]
        percent = event.get("percent")
        self.notification_manager.notify_progress(
            self.active_task_kind,
            title,
            self._notification_lines_from_event(event),
            int(percent) if percent is not None else None,
        )

    def _notify_complete(self, stage_key: str, title: str, lines: list[str]) -> None:
        """Send an external completion notification when configured.

        Args:
            stage_key: Notification stage key.
            title: User-facing title.
            lines: Plain-text summary lines.
        """

        if self.notification_manager is not None:
            self.notification_manager.notify_complete(stage_key, title, lines)

    def _notify_failure(self, title: str, message: str) -> None:
        """Send an external failure or stop notification for the active task.

        Args:
            title: User-facing title.
            message: Failure details.
        """

        if self.active_task_kind and self.notification_manager is not None:
            self.notification_manager.notify_failure(self.active_task_kind, title, message)

    def _notification_lines_from_event(self, event: dict[str, Any]) -> list[str]:
        """Build compact notification text from a worker progress event.

        Args:
            event: Progress event emitted by a worker.

        Returns:
            Body lines for the notification message.
        """

        lines: list[str] = []
        if event.get("message"):
            lines.append(str(event["message"]))
        if "epoch" in event and "total_epochs" in event:
            lines.append(f"Epoch: {event['epoch']}/{event['total_epochs']}")
        if "step" in event and "total_steps" in event:
            lines.append(f"Step: {event['step']}/{event['total_steps']}")
        train_loss = self._finite_metric(event.get("train_loss"))
        if train_loss is not None:
            lines.append(f"Train loss: {float(train_loss):.4f}")
        val_loss = self._finite_metric(event.get("val_loss"))
        if val_loss is not None:
            lines.append(f"Validation loss: {float(val_loss):.4f}")
        learning_rate = self._finite_metric(event.get("learning_rate"))
        if learning_rate is not None:
            lines.append(f"Learning rate: {float(learning_rate):.2e}")
        tokens_per_second = self._finite_metric(event.get("tokens_per_second"))
        if tokens_per_second is not None:
            lines.append(f"Speed: {float(tokens_per_second):.0f} tokens/sec")
        eta_seconds = self._finite_metric(event.get("eta_seconds"))
        if eta_seconds is not None:
            lines.append(f"ETA: {self._format_duration(float(eta_seconds))}")
        vram_allocated = self._finite_metric(event.get("vram_allocated_gb"))
        vram_reserved = self._finite_metric(event.get("vram_reserved_gb"))
        if vram_allocated is not None or vram_reserved is not None:
            allocated = "-" if vram_allocated is None else f"{float(vram_allocated):.2f} GB"
            reserved = "-" if vram_reserved is None else f"{float(vram_reserved):.2f} GB"
            lines.append(f"VRAM: {allocated} allocated, {reserved} reserved")
        return lines[:10]

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration for compact UI display.

        Args:
            seconds: Duration in seconds.

        Returns:
            Human-readable compact duration.
        """

        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def _drain_progress_queue(self) -> None:
        """Drain queued worker progress events on the UI thread."""

        if self.progress_queue is None or self.active_log is None or self.active_progress_bar is None:
            return
        drained = 0
        last_percent = None
        while drained < 12:
            try:
                event = self.progress_queue.get_nowait()
            except Empty:
                break
            notification_event = event
            if isinstance(event, dict) and event.get("percent") is not None:
                last_percent = event.get("percent")
                event = {**event, "percent": None}
            if (
                isinstance(notification_event, dict)
                and self.active_task_kind == "dataset_download"
                and notification_event.get("button_text")
                and self.active_button is not None
            ):
                self.active_button_text = str(notification_event["button_text"])
                self.active_button.setText(self.active_button_text)
            self._handle_progress(event, self.active_log, self.active_progress_bar)
            if isinstance(notification_event, dict):
                self._notify_progress(notification_event)
            drained += 1
        if last_percent is not None:
            self.active_progress_bar.setValue(max(0, min(100, int(last_percent))))

    def _thread_finished(self) -> None:
        """Clean up thread bookkeeping after a worker finishes."""

        LOGGER.info("Background task thread finished")
        self._drain_progress_queue()
        if self.progress_timer.isActive():
            self.progress_timer.stop()
        self.thread = None
        self.worker = None
        if self.result_bridge is not None:
            self.result_bridge.deleteLater()
        self.result_bridge = None
        self.stop_event = None
        self.progress_queue = None
        self.active_log = None
        self.active_progress_bar = None
        if self.active_stop_button is not None:
            self.active_stop_button.setEnabled(False)
        self.active_stop_button = None
        if self.active_button is not None:
            self._clear_button_busy()
        self.active_task_kind = ""

    def _task_failed(self, message: str, log: QTextEdit, progress_bar: QProgressBar) -> None:
        """Handle background task failure.

        Args:
            message: Error message.
            log: Log widget to append to.
            progress_bar: Progress bar to reset.
        """

        stopped_by_user = "stopped by user" in message.lower()
        if stopped_by_user:
            LOGGER.info("Background task stopped by user: %s", message)
        else:
            LOGGER.error("Background task error: %s", message)
        log.append(f"Stopped: {message}" if stopped_by_user else f"Error: {message}")
        self._notify_failure("Task stopped" if stopped_by_user else "Task failed", message)
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        if stopped_by_user:
            self.project_state.setText("Stopped")
        self._clear_button_busy()

    def _set_button_busy(self, button: QPushButton, text: str) -> None:
        """Disable a button and start its spinner text.

        Args:
            button: Button to mark busy.
            text: Busy label.
        """

        self.active_button = button
        self.active_button_text = text
        self.active_button_restore_text = button.text()
        self.spinner_index = 0
        button.setEnabled(False)
        button.setText(f"| {text}")
        self.spinner_timer.start(150)

    def _clear_button_busy(self, final_text: Optional[str] = None) -> None:
        """Restore the active busy button.

        Args:
            final_text: Optional final button text.
        """

        if self.spinner_timer.isActive():
            self.spinner_timer.stop()
        if self.active_button:
            self.active_button.setEnabled(True)
            self.active_button.setText(final_text or self.active_button_restore_text)
        if self.active_stop_button:
            self.active_stop_button.setEnabled(False)
        self.active_button = None
        self.active_button_text = ""
        self.active_button_restore_text = ""

    def _tick_spinner(self) -> None:
        """Advance the active button spinner frame."""

        if not self.active_button:
            return
        frames = "|/-\\"
        self.spinner_index = (self.spinner_index + 1) % len(frames)
        self.active_button.setText(f"{frames[self.spinner_index]} {self.active_button_text}")
