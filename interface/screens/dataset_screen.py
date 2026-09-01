from __future__ import annotations

# DatasetScreenMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class DatasetScreenMixin:
    def _dataset_config_from_ui(self) -> DatasetConfig:
        """Collect dataset options from the current UI controls.

        Returns:
            Dataset preparation configuration.
        """

        conversation_paths: list[Path] = []
        instruction_paths: list[Path] = []
        tool_call_paths: list[Path] = []
        dataset_stage = self._dataset_stage_value()
        selected_local_paths = self._selected_default_data_paths_for_stage(dataset_stage)
        structured_paths = [
            path for path in selected_local_paths if path.suffix.lower() in {".json", ".jsonl"}
        ]
        if dataset_stage == "conversation":
            conversation_paths = structured_paths
        elif dataset_stage == "instruction":
            instruction_paths = structured_paths
        elif dataset_stage == "tool_call":
            tool_call_paths = structured_paths
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
            tool_call_dataset_paths=tool_call_paths,
            default_data_paths=selected_local_paths,
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

    def prepare_dataset(self) -> None:
        if not bool(QApplication.instance().property("license_valid")):
            self.context_length.setValue(min(self.context_length.value(), 1000))
        """Collect dataset options and start dataset preparation."""

        config = self._dataset_config_from_ui()
        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self._reset_dataset_quality_report()
        self.dataset_log.append("Preparing dataset...")
        self.dataset_log.append(f"App log file: {self.log_file_path}")
        self.dataset_log.append(f"Dataset purpose: {dataset_stage_label(config.dataset_stage)}")
        if config.conversation_dataset_paths:
            self.dataset_log.append(f"Local conversation JSON/JSONL: {len(config.conversation_dataset_paths)} path(s)")
            LOGGER.info("Local conversation JSON/JSONL datasets: %s", "; ".join(str(path) for path in config.conversation_dataset_paths))
        if config.instruction_dataset_paths:
            self.dataset_log.append(f"Local instruction JSON/JSONL: {len(config.instruction_dataset_paths)} path(s)")
            LOGGER.info("Local instruction JSON/JSONL datasets: %s", "; ".join(str(path) for path in config.instruction_dataset_paths))
        if config.tool_call_dataset_paths:
            self.dataset_log.append(f"Local tool-call JSON/JSONL: {len(config.tool_call_dataset_paths)} path(s)")
            LOGGER.info("Local tool-call JSON/JSONL datasets: %s", "; ".join(str(path) for path in config.tool_call_dataset_paths))
        if config.default_data_paths:
            self.dataset_log.append(f"Bundled default data: {len(config.default_data_paths)} file(s)")
            LOGGER.info("Bundled default data files: %s", "; ".join(str(path) for path in config.default_data_paths))
        if self.include_conversation_datasets.isChecked():
            selected_labels = [
                action.text()
                for action in getattr(self, "conversation_dataset_actions", {}).values()
                if action.isChecked() and action.isVisible()
            ]
            if selected_labels:
                hf_cache = config.output_dir / "cache" / "huggingface"
                self.dataset_log.append(f"Online training datasets: {', '.join(selected_labels)}")
                self.dataset_log.append(f"Downloading/loading online data at: {hf_cache}")
                LOGGER.info("Online training datasets: %s", ", ".join(selected_labels))
                LOGGER.info("Downloading/loading online data at: %s", hf_cache)
            else:
                self.dataset_log.append("Online training datasets are enabled, but no dataset is selected for this purpose.")
                LOGGER.warning("Online training datasets enabled, but no dataset is selected")
        else:
            self.dataset_log.append("Online training datasets: off. Local source files only.")
            LOGGER.info("Online training datasets: off. Local source files only.")
            checked_count = sum(
                1
                for action in getattr(self, "conversation_dataset_actions", {}).values()
                if action.isChecked()
            )
            if checked_count:
                self.dataset_log.append("Checked online dataset choices are ignored until the master checkbox is enabled.")
                LOGGER.info("Checked online dataset choices are ignored until the master checkbox is enabled")
        LOGGER.info(
            "Preparing dataset: input=%s output=%s stage=%s online_datasets=%s conversation_json=%s instruction_json=%s tool_call_json=%s",
            config.input_dir,
            config.output_dir,
            config.dataset_stage,
            ",".join(config.conversation_datasets) or "off",
            ";".join(str(path) for path in config.conversation_dataset_paths) or "off",
            ";".join(str(path) for path in config.instruction_dataset_paths) or "off",
            ";".join(str(path) for path in config.tool_call_dataset_paths) or "off",
        )
        self.project_state.setText("Preparing dataset")
        self.dataset_status.setText("Dataset: preparing")
        self.auto_vocab_label.setText("Calculating...")
        self._run_task(
            build_dataset,
            (config,),
            self._dataset_finished,
            self.dataset_log,
            self.dataset_progress,
            with_progress=True,
            button=self.prepare_button,
            stop_button=self.stop_dataset_button,
            busy_text="Preparing Dataset",
            task_kind="dataset",
            isolate_process=True,
        )

    @Slot(object)
    def _dataset_finished(self, result: Any) -> None:
        """Update UI after dataset preparation finishes.

        Args:
            result: Dataset build result.
        """

        self.dataset_progress.setRange(0, 100)
        self.dataset_progress.setValue(100)
        self.dataset_result_applied = True
        self.auto_vocab_label.setText(f"{result.vocab_size:,}")
        partial_file_count = int(getattr(result, "partial_file_count", 0) or 0)
        failed_file_count = int(getattr(result, "failed_file_count", 0) or 0)
        invalid_record_count = int(getattr(result, "invalid_record_count", 0) or 0)
        preparation_outcome = str(
            getattr(result, "preparation_outcome", "")
            or (
                "completed_with_warnings"
                if failed_file_count
                else "completed"
            )
        )
        has_warnings = (
            preparation_outcome == "completed_with_warnings"
            or partial_file_count > 0
            or failed_file_count > 0
            or invalid_record_count > 0
        )

        LOGGER.info(
            "Dataset prepared: documents=%s tokens=%s vocab=%s code=%s prose=%s conversation=%s output=%s",
            result.document_count,
            result.token_count,
            result.vocab_size,
            result.code_sample_count,
            result.prose_sample_count,
            getattr(result, "conversation_sample_count", 0),
            result.output_dir,
        )

        self.dataset_log.append(
            f"Prepared {result.document_count} documents, "
            f"{result.character_count:,} characters, "
            f"{result.token_count:,} tokens, "
            f"vocab {result.vocab_size:,}."
        )

        if getattr(result, "train_window_count", 0) or getattr(result,
                                                               "val_window_count",
                                                               0):
            self.dataset_log.append(
                f"Training windows: {result.train_window_count:,}; "
                f"validation windows: {result.val_window_count:,}."
            )

        self.dataset_log.append(
            f"Cache summary: reused {result.cached_file_count:,} file(s), "
            f"processed {result.processed_file_count:,} file(s), "
            f"skipped {result.skipped_file_count:,}, "
            f"partial {partial_file_count:,}, "
            f"failed {failed_file_count:,}."
        )
        if has_warnings:
            self.dataset_log.append(
                "[WARN] Dataset preparation completed with warnings: "
                f"{partial_file_count:,} partial source(s), "
                f"{failed_file_count:,} unusable source(s), and "
                f"{invalid_record_count:,} invalid record(s)."
            )

        if getattr(result, "dataset_version_id", ""):
            self.dataset_log.append(
                f"Dataset version: {result.dataset_version_id}"
            )

        if result.warning:
            self.dataset_log.append(f"Recommendation: {result.warning}")

        self._update_dataset_quality_report(
            {
                "document_count": result.document_count,
                "token_count": result.token_count,
                "train_window_count": getattr(result, "train_window_count", 0),
                "val_window_count": getattr(result, "val_window_count", 0),
                "character_count": result.character_count,
                "tokenizer_vocab_size": result.vocab_size,
                "code_sample_count": result.code_sample_count,
                "prose_sample_count": result.prose_sample_count,
                "conversation_sample_count": getattr(result,
                                                     "conversation_sample_count",
                                                     0),
                "cached_file_count": result.cached_file_count,
                "processed_file_count": result.processed_file_count,
                "skipped_file_count": result.skipped_file_count,
                "partial_file_count": partial_file_count,
                "failed_file_count": failed_file_count,
                "invalid_record_count": invalid_record_count,
                "preparation_outcome": preparation_outcome,
                "warning": result.warning,
                "sequence_token_stats": getattr(result, "sequence_token_stats",
                                                {}),
                "duplicate_block_count": getattr(result,
                                                 "duplicate_block_count", 0),
                "unique_block_count": getattr(result, "unique_block_count", 0),
                "corpus_block_count": getattr(result, "corpus_block_count", 0),
                "duplicate_block_ratio": getattr(result,
                                                 "duplicate_block_ratio", 0.0),
                "unique_block_ratio": getattr(result, "unique_block_ratio",
                                              1.0),
            }
        )

        self.train_data_dir.setText(str(result.output_dir))
        self.project_state.setText(
            "Dataset ready with warnings"
            if has_warnings
            else "Dataset ready"
        )

        self.dataset_status.setText(
            f"Dataset: {result.document_count} files, {result.token_count:,} tokens"
        )
        if has_warnings:
            self.dataset_status.setText(
                f"Dataset: {result.document_count:,} valid, "
                f"{invalid_record_count:,} invalid record(s), "
                f"{result.token_count:,} tokens"
            )

        if result.code_sample_count and not has_warnings:
            self.dataset_status.setText(
                f"Dataset: {result.code_sample_count:,} code, "
                f"{result.prose_sample_count:,} prose, "
                f"{result.token_count:,} tokens"
            )

        self.refresh_model_estimate()
        self.refresh_fine_tune_workflow()

        self._notify_complete(
            "dataset",
            "Dataset preparation complete",
            [
                f"Output: {result.output_dir}",
                f"Documents: {result.document_count:,}",
                f"Characters: {result.character_count:,}",
                f"Tokens: {result.token_count:,}",
                f"Vocabulary: {result.vocab_size:,}",
                (
                    "Windows: "
                    f"{getattr(result, 'train_window_count', 0):,} training, "
                    f"{getattr(result, 'val_window_count', 0):,} validation"
                ),
                (
                    "Content mix: "
                    f"{result.code_sample_count:,} code, "
                    f"{result.prose_sample_count:,} prose, "
                    f"{getattr(result, 'conversation_sample_count', 0):,} conversation"
                ),
                (
                    "Files: "
                    f"{result.processed_file_count:,} processed, "
                    f"{result.cached_file_count:,} cached, "
                    f"{result.skipped_file_count:,} skipped, "
                    f"{partial_file_count:,} partial, "
                    f"{failed_file_count:,} failed"
                ),
                f"Invalid records: {invalid_record_count:,}",
                f"Dataset version: {getattr(result, 'dataset_version_id', '') or '-'}",
                (
                    "Preparation: completed with warnings"
                    if has_warnings
                    else "Preparation: completed"
                ),
                f"Health: {'warning - ' + result.warning if result.warning else 'ready'}",
            ],
        )

        self._clear_button_busy("DataSet Prepared")
