from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart10:
    def _system_cpu_value(self) -> Optional[float]:
        """Read system CPU utilization for live telemetry.

        Returns:
            System CPU percentage, or None when unavailable.
        """

        if psutil is None:
            return None
        return float(psutil.cpu_percent(interval=None))

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

    def _apply_chat_delta(self, event: dict[str, Any]) -> None:
        """Apply one streamed chat chunk to the rendered conversation.

        Args:
            event: Chat stream progress event.
        """

        self.chat_stream_reply += str(event.get("content", ""))
        should_follow = self._is_chat_near_bottom()
        self._render_chat_markdown(self.chat_stream_reply)
        if should_follow:
            self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum())
        self._set_chat_stats(
            float(event.get("elapsed_seconds", 0.0)),
            int(event.get("token_count", 0)),
            float(event.get("tokens_per_second", 0.0)),
        )

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

    def _dataset_config_from_ui(self) -> DatasetConfig:
        """Collect dataset options from the current UI controls.

        Returns:
            Dataset preparation configuration.
        """

        conversation_paths: list[Path] = []
        instruction_paths: list[Path] = []
        dataset_stage = self._dataset_stage_value()
        return DatasetConfig(
            input_dir=Path(self.input_dir.text()),
            output_dir=Path(self.dataset_dir.text()),
            vocab_size=None if self.auto_vocab.isChecked() else self.manual_vocab_size.value(),
            conversation_datasets=self._selected_conversation_datasets(),
            conversation_sample_limit=self.conversation_sample_limit.value(),
            conversation_dataset_path=conversation_paths[0] if conversation_paths else None,
            instruction_dataset_path=instruction_paths[0] if instruction_paths else None,
            conversation_dataset_paths=conversation_paths,
            instruction_dataset_paths=instruction_paths,
            default_data_paths=self._selected_default_data_paths_for_stage(dataset_stage),
            mixture_weights=self._mixture_weights_from_ui(),
            min_frequency=self.min_frequency.value(),
            context_length=self.context_length.value(),
            validation_split=self.validation_split.value(),
            lowercase=False,
            max_workers=self.max_workers.value(),
            code_training_mode=self.code_training_mode.isChecked(),
            include_prose=self.include_prose.isChecked(),
            include_source_code=self.include_source_code.isChecked(),
            extract_code_blocks=self.extract_code_blocks.isChecked(),
            preserve_indentation=self.preserve_indentation.isChecked(),
            generate_instruction_samples=self.instruction_samples.isChecked(),
            reasoning_sample_mode=self._reasoning_sample_mode_value(),
            prepare_mode=self._prepare_mode_value(),
            tokenizer_strategy=self._tokenizer_strategy_value(),
            tokenizer_path=Path(self.tokenizer_path.text()) if self.tokenizer_path.text().strip() else None,
            dataset_stage=dataset_stage,
            tokenizer_training_max_gb=self.tokenizer_training_max_gb.value(),
        )

    def _selected_default_data_paths_for_stage(self, stage: str) -> list[Path]:
        """Return selected bundled files that match the dataset purpose.

        Args:
            stage: Dataset preparation stage.

        Returns:
            Selected paths suitable for the requested stage.
        """

        # Folder selection is the workflow configuration.  Do not apply a
        # second hardcoded stage filter here; the Dataset Sources tree already
        # contains exactly the files selected by the user.
        return self._selected_default_data_paths()

    @staticmethod
    def _split_path_list(text: str) -> list[Path]:
        """Split a semicolon-delimited path field.

        Args:
            text: Raw path field text.

        Returns:
            Parsed paths.
        """

        return [Path(item.strip().strip('"')) for item in text.split(";") if item.strip()]

    def check_project_health(self) -> None:
        """Run a project health check in the background."""

        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self.dataset_log.append("Checking project health...")
        self.project_state.setText("Checking health")
        self._run_task(
            check_project_health,
            (
                Path(self.input_dir.text()),
                Path(self.dataset_dir.text()),
                Path(self.model_dir.text()),
                Path(self.export_dir.text()),
                Path(self.gguf_path.text()) if self.gguf_path.text().strip() else None,
                Path(self.llama_cpp_dir.text()) if self.llama_cpp_dir.text().strip() else None,
                self.device.currentText(),
            ),
            self._health_check_finished,
            self.dataset_log,
            self.dataset_progress,
            with_progress=True,
            button=self.health_check_button,
            stop_button=self.stop_dataset_button,
            busy_text="Checking Health",
            isolate_process=True,
        )

    @Slot(object)
    def _health_check_finished(self, result: Any) -> None:
        """Display project health check results.

        Args:
            result: Project health result.
        """

        self.dataset_progress.setValue(100)
        self.dataset_log.append("")
        self.dataset_log.append(f"Project health: {result.status.upper()} ({result.summary})")
        for check in result.checks:
            marker = {"ok": "OK", "warning": "WARN", "error": "ERROR"}.get(check.get("status"), "INFO")
            self.dataset_log.append(f"[{marker}] {check.get('name')}: {check.get('detail')}")
        self.project_state.setText("Health checked")
        self._clear_button_busy("Check Health")

    def preview_dataset(self) -> None:
        """Run a dataset preview and quality scan in the background."""

        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self.dataset_log.append("Previewing dataset...")
        self.project_state.setText("Previewing dataset")
        self._run_task(
            scan_dataset_preview,
            (self._dataset_config_from_ui(),),
            self._dataset_preview_finished,
            self.dataset_log,
            self.dataset_progress,
            with_progress=True,
            button=self.preview_dataset_button,
            stop_button=self.stop_dataset_button,
            busy_text="Previewing Dataset",
            isolate_process=True,
        )

    @Slot(object)
    def _dataset_preview_finished(self, result: Any) -> None:
        """Finish preview UI cleanup even when rendering a result fails."""
        try:
            self._render_dataset_preview_result(result)
        finally:
            # Keep the action available after both successful and malformed
            # worker results; otherwise the spinner can leave it disabled.
            self._clear_button_busy("Preview Dataset")

    def _render_dataset_preview_result(self, result: Any) -> None:
        """Display dataset preview and quality scan results.

        Args:
            result: Dataset preview result.
        """

        self.dataset_progress.setValue(100)
        suffix_text = ", ".join(f"{suffix}: {count}" for suffix, count in
                                result.suffix_counts.items()) or "none"
        self.dataset_log.append("")
        self.dataset_log.append(
            f"Source files: {result.source_file_count:,}; size: {result.total_bytes / (1024 * 1024):.2f} MB")
        self.dataset_log.append(f"File types: {suffix_text}")
        self.dataset_log.append(
            f"Prepared dataset artifacts: {'found' if result.prepared else 'not complete'}")
        self.dataset_log.append(
            f"Duplicate scan: {result.duplicate_count:,} file entries in {len(result.duplicate_groups):,} likely group(s).")
        self.dataset_log.append(
            f"Bad extraction scan: {result.bad_extraction_count:,} suspicious file(s).")
        self.dataset_log.append(
            f"Code/prose balance: {result.balance_label} ({result.code_preview_count:,}/{result.prose_preview_count:,}).")
        self.dataset_log.append(
            f"Training readiness: {result.readiness_label} ({result.readiness_score}/100).")
        for reason in result.readiness_reasons[:8]:
            self.dataset_log.append(f"- {reason}")
        self.dataset_quality_duplicates.setText(
            f"Duplicates: {result.duplicate_count:,}")
        self.dataset_quality_extraction.setText(
            f"Extraction: {result.bad_extraction_count:,} flagged")
        self.dataset_quality_balance.setText(
            f"Balance: {result.balance_label}")
        self.dataset_quality_readiness.setText(
            f"Readiness: {result.readiness_label} {result.readiness_score}/100")
        if result.summary:
            self._update_dataset_quality_report(result.summary)
            # dataset_quality_duplicates is intentionally left alone here:
            # _update_dataset_quality_report() just set it to the block-level
            # duplication percentage from the prepared corpus (the more useful,
            # actionable metric). Re-setting it to result.duplicate_count (a
            # raw duplicate *file* count from the earlier preview scan) would
            # silently discard that and always show the old metric instead.
            self.dataset_quality_extraction.setText(
                f"Extraction: {result.bad_extraction_count:,} flagged")
            self.dataset_quality_balance.setText(
                f"Balance: {result.balance_label}")
            self.dataset_quality_readiness.setText(
                f"Readiness: {result.readiness_label} {result.readiness_score}/100")
            tokens = int(result.summary.get("token_count", 0) or 0)
            vocab = int(result.summary.get("tokenizer_vocab_size", 0) or 0)
            self.dataset_log.append(
                f"Prepared summary: {tokens:,} tokens, vocab {vocab:,}.")
        else:
            self.dataset_quality_samples.setText(
                f"Preview: {len(result.sample_previews):,} shown")
            self.dataset_quality_tokens.setText("Tokens: not prepared")
            self.dataset_quality_windows.setText("Windows: not prepared")
            self.dataset_quality_vocab.setText("Vocab: not prepared")
            self.dataset_quality_code.setText(
                f"Code/prose: {result.code_preview_count:,}/{result.prose_preview_count:,}")
            self.dataset_quality_cache.setText(
                f"Files: {result.source_file_count:,} source")
        if result.duplicate_groups:
            self.dataset_log.append("")
            self.dataset_log.append("Likely duplicates:")
            for group in result.duplicate_groups[:8]:
                self.dataset_log.append(
                    f"- {group.get('type')}: {group.get('count')} file(s)")
                for path in group.get("files", [])[:4]:
                    self.dataset_log.append(f"    {Path(path).name}")
        if result.bad_extraction_files:
            self.dataset_log.append("")
            self.dataset_log.append("Suspicious extraction files:")
            for item in result.bad_extraction_files[:12]:
                self.dataset_log.append(
                    f"- {Path(item.get('path', '')).name}: {item.get('reasons')}")
        suggestions: list[str] = []
        if result.duplicate_groups:
            suggestions.append(
                "Remove or move duplicate files before preparing the final dataset.")
        if result.bad_extraction_files:
            suggestions.append(
                "Replace flagged PDFs with text/source versions, or remove files with bad extraction.")
        if result.balance_label == "Prose heavy" and self.code_training_mode.isChecked():
            suggestions.append(
                "Add real source-code folders or enable source-file inclusion for a stronger coding model.")
        if result.balance_label == "Code heavy":
            suggestions.append(
                "Add README/tutorial/prose explanations if you want the model to explain code well.")
        if result.readiness_label in {"Needs cleanup", "Not ready"}:
            suggestions.append(
                "Run Preview Dataset again after cleanup and only train once readiness improves.")
        if hasattr(self, "dataset_advisor"):
            if suggestions:
                self.dataset_advisor.setPlainText(
                    "\n".join(f"- {suggestion}" for suggestion in suggestions))
            else:
                self.dataset_advisor.setPlainText(
                    "No immediate cleanup suggestions. Dataset looks acceptable for the current preview.")
        if suggestions:
            self.dataset_log.append("")
            self.dataset_log.append("Cleanup suggestions:")
            for suggestion in suggestions:
                self.dataset_log.append(f"- {suggestion}")
        if result.issues:
            self.dataset_quality_warning.setText(
                f"Warnings: {len(result.issues)}")
            self.dataset_log.append("")
            self.dataset_log.append("Quality notes:")
            for issue in result.issues[:12]:
                self.dataset_log.append(f"- {issue}")
        else:
            self.dataset_quality_warning.setText("Warnings: none")
        if result.sample_previews:
            self.dataset_log.append("")
            self.dataset_log.append("Preview samples:")
            for index, sample in enumerate(result.sample_previews, start=1):
                label = sample.get("language") or sample.get("kind") or "text"
                self.dataset_log.append(
                    f"\n[{index}] {Path(sample.get('path', '')).name} ({label}, {sample.get('characters')} chars)")
                self.dataset_log.append(
                    sample.get("preview", "").replace("\n", "\n    ")[:1400])
        self.project_state.setText("Dataset previewed")
