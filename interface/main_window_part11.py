from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart11:
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
            "Preparing dataset: input=%s output=%s stage=%s online_datasets=%s conversation_json=%s instruction_json=%s",
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

        self.dataset_progress.setValue(100)
        self.auto_vocab_label.setText(f"{result.vocab_size:,}")

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
            f"processed {result.processed_file_count:,} file(s)."
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
                "failed_file_count": result.failed_file_count,
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
        self.project_state.setText("Dataset ready")

        self.dataset_status.setText(
            f"Dataset: {result.document_count} files, {result.token_count:,} tokens"
        )

        if result.code_sample_count:
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
                    f"{result.failed_file_count:,} failed"
                ),
                f"Dataset version: {getattr(result, 'dataset_version_id', '') or '-'}",
                f"Health: {'warning - ' + result.warning if result.warning else 'ready'}",
            ],
        )

        self._clear_button_busy("DataSet Prepared")

    def _prepare_mode_value(self) -> str:
        """Return the selected dataset preparation mode.

        Returns:
            Internal mode value.
        """

        label = self.prepare_mode.currentText()
        if label == "Full rebuild":
            return "full_rebuild"
        if label == "Force reprocess":
            return "force_reprocess"
        return "incremental"

    def _tokenizer_strategy_value(self) -> str:
        """Return the selected tokenizer strategy.

        Returns:
            Internal tokenizer strategy value.
        """

        label = self.tokenizer_strategy.currentText()
        if label == "Train new tokenizer":
            return "train_new"
        if label == "Reuse dataset tokenizer":
            return "reuse_dataset"
        if label == "Import tokenizer.json":
            return "import_tokenizer"
        return "auto"

    def _reasoning_sample_mode_value(self) -> str:
        """Return the selected reasoning sample mode.

        Returns:
            Internal reasoning sample mode.
        """

        label = self.reasoning_sample_mode.currentText()
        if label == "Detailed code reasoning":
            return "detailed"
        if label == "No reasoning wrapper":
            return "none"
        return "scaffold"

    def _dataset_stage_value(self) -> str:
        """Return the selected dataset preparation stage.

        Returns:
            Dataset stage identifier.
        """

        return self.dataset_stage.currentText().strip().lower().replace(" ", "_") or "base"

    def _set_dataset_stage(self, stage: str) -> None:
        """Set the dataset stage combo from an internal stage value.

        Args:
            stage: Dataset stage identifier.
        """

        index = self.dataset_stage.findText(stage, Qt.MatchFixedString)
        if index < 0:
            self.dataset_stage.addItem(stage)
            index = self.dataset_stage.count() - 1
        self.dataset_stage.setCurrentIndex(index)
        self._update_online_dataset_stage_controls()

    def _update_online_dataset_stage_controls(self) -> None:
        """Show and enable online datasets for the selected training stage."""

        if not hasattr(self, "dataset_stage"):
            return
        self._apply_dataset_license_gating()
        stage = self._dataset_stage_value()
        allowed = set(CONVERSATION_DATASET_PRESETS)
        include_online = self.include_conversation_datasets.isChecked()
        for dataset_id, action in getattr(self, "conversation_dataset_actions", {}).items():
            visible = dataset_id in allowed
            action.setVisible(visible)
            action.setEnabled(include_online and visible)
            if not visible:
                action.setChecked(False)
        if hasattr(self, "conversation_dataset_button"):
            self.conversation_dataset_button.setEnabled(include_online)
        self.conversation_sample_limit.setEnabled(include_online)
        self._update_conversation_dataset_button_text()
        self.conversation_datasets_status.setText(
            f"{self.dataset_stage.currentText()}: choose optional online datasets."
            if include_online else "Choose optional online datasets, or use local folders only."
        )

    def _apply_dataset_license_gating(self) -> None:
        """Keep trial restrictions applied after Dataset Sources widgets change."""

        licensed = bool(QApplication.instance().property("license_valid"))
        if hasattr(self, "include_conversation_datasets"):
            if not licensed:
                self.include_conversation_datasets.setChecked(False)
            self.include_conversation_datasets.setEnabled(licensed)
        for name in ("external_dataset_download_button", "custom_huggingface_download"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(licensed)

    def _selected_conversation_datasets(self) -> list[str]:
        """Return selected built-in conversation dataset IDs.

        Returns:
            Selected dataset identifiers.
        """

        allowed = set(CONVERSATION_DATASET_PRESETS)
        selected = [
            dataset_id
            for dataset_id, action in getattr(self, "conversation_dataset_actions", {}).items()
            if dataset_id in allowed and action.isChecked() and self.include_conversation_datasets.isChecked()
        ]
        custom = self.custom_huggingface_dataset.text().strip()
        if custom and self.include_conversation_datasets.isChecked():
            selected.append(f"hf_custom:{custom}")
        return selected

    def _download_custom_huggingface_dataset(self) -> None:
        """Enable the entered Hugging Face dataset for the next preparation run."""
        value = self.custom_huggingface_dataset.text().strip()
        if not value:
            self.conversation_datasets_status.setText("Enter a Hugging Face dataset ID or URL first.")
            return
        self.include_conversation_datasets.setChecked(True)
        self.conversation_datasets_status.setText(
            f"Custom dataset queued: {value}. It will download during dataset preparation."
        )
        self._update_conversation_dataset_button_text()

    def _set_selected_conversation_datasets(self, dataset_ids: list[str]) -> None:
        """Restore selected conversation dataset actions.

        Args:
            dataset_ids: Dataset IDs to select.
        """

        selected = set(dataset_ids)
        allowed = set(dataset_ids_for_stage(self._dataset_stage_value()))
        for dataset_id, action in getattr(self, "conversation_dataset_actions", {}).items():
            action.setChecked(dataset_id in selected and dataset_id in allowed)
            action.setEnabled(self.include_conversation_datasets.isChecked() and dataset_id in allowed)
        if hasattr(self, "custom_huggingface_dataset"):
            self.custom_huggingface_dataset.setText(
                next((value[10:] for value in dataset_ids if value.startswith("hf_custom:")), "")
            )
        if hasattr(self, "conversation_sample_limit"):
            self.conversation_sample_limit.setEnabled(self.include_conversation_datasets.isChecked())
        self._update_conversation_dataset_button_text()
        if hasattr(self, "conversation_datasets_status"):
            self._update_online_dataset_stage_controls()

    def _update_conversation_dataset_button_text(self) -> None:
        """Refresh the compact online dataset selector label."""

        if not hasattr(self, "conversation_dataset_button"):
            return
        allowed = set(dataset_ids_for_stage(self._dataset_stage_value())) if hasattr(self, "dataset_stage") else set()
        selected_labels = [
            action.text()
            for dataset_id, action in getattr(self, "conversation_dataset_actions", {}).items()
            if dataset_id in allowed and action.isChecked()
        ]
        if not self.include_conversation_datasets.isChecked():
            self.conversation_dataset_button.setText("Online datasets off")
        elif not selected_labels:
            self.conversation_dataset_button.setText("Choose online datasets")
        elif len(selected_labels) == 1:
            self.conversation_dataset_button.setText(selected_labels[0])
        else:
            self.conversation_dataset_button.setText(f"{len(selected_labels)} online datasets selected")

    def configure_fine_tune_dataset_builder(self) -> None:
        """Configure the Ingest tab for the selected fine-tune dataset type."""

        stage_label = self.fine_tune_dataset_builder_stage.currentText()
        stage = {
            "Instruction fine-tune": "instruction",
            "Conversation fine-tune": "conversation",
            "Tool-call fine-tune": "tool_call",
            "Code fine-tune": "code",
        }.get(stage_label, "instruction")
        starter_datasets = {
            "instruction": ["alpaca_52k"],
            "conversation": ["dailydialog"],
            "tool_call": [],
            "code": ["codealpaca_20k"],
        }
        self._set_dataset_stage(stage)
        self.include_conversation_datasets.setChecked(bool(starter_datasets.get(stage)))
        self._set_selected_conversation_datasets(starter_datasets.get(stage, []))
        if stage == "code":
            self.code_training_mode.setChecked(True)
            self.include_source_code.setChecked(True)
            self.extract_code_blocks.setChecked(True)
            self.preserve_indentation.setChecked(True)
        self._set_mixture_weights({})
        self._switch_page(0)
        self.dataset_log.append(f"Configured Ingest for {dataset_stage_label(stage)}. Import the base tokenizer before preparing.")
        self.project_state.setText(f"Configured {dataset_stage_label(stage)} data")

    def _dataset_plan_from_ui(self) -> dict[str, float]:
        """Return dataset blueprint state.

        Returns:
            Empty mapping because category percentages are disabled.
        """

        return {}

    def _selected_default_data_paths(self) -> list[Path]:
        """Return bundled default data files selected in the Dataset Blueprint.

        Returns:
            Selected bundled data paths.
        """

        if not hasattr(self, "default_data_actions"):
            return [path for path, _category in iter_default_data_files()]
        return [
            Path(path)
            for path, item in self.default_data_actions.items()
            if item.checkState(0) == Qt.Checked
        ]
