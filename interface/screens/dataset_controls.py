from __future__ import annotations

# DatasetControlsMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class DatasetControlsMixin:
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

    def _tokenizer_strategy_reuses(self) -> bool:
        """Return whether current tokenizer strategy ignores vocabulary controls.

        Returns:
            True when an existing tokenizer is selected directly.
        """

        return self.tokenizer_strategy.currentText() in {"Reuse dataset tokenizer", "Import tokenizer.json"}

    def _update_tokenizer_strategy_controls(self) -> None:
        """Enable only the tokenizer inputs relevant to the selected strategy."""

        imports_tokenizer = self.tokenizer_strategy.currentText() == "Import tokenizer.json"
        reuses_tokenizer = self._tokenizer_strategy_reuses()
        if hasattr(self, "tokenizer_path_row"):
            self.tokenizer_path_row.setEnabled(imports_tokenizer)
        self.tokenizer_path.setEnabled(imports_tokenizer)
        self.auto_vocab.setEnabled(not reuses_tokenizer)
        self.manual_vocab_size.setEnabled(not reuses_tokenizer and not self.auto_vocab.isChecked())
        self.min_frequency.setEnabled(not reuses_tokenizer)
