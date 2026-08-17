from __future__ import annotations

# FineTuningScreenMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class FineTuningScreenMixin:
    def _training_mode_value(self) -> str:
        """Return the selected training mode identifier.

        Returns:
            Stable training mode used by the trainer.
        """

        return {
            "Pretrain from scratch": "pretrain",
            "Fine-tune checkpoint": "fine_tune",
            "Instruction fine-tune": "fine_tune",
            "Conversation fine-tune": "fine_tune",
            "Tool-call fine-tune": "fine_tune",
            "Code fine-tune": "fine_tune",
        }.get(self.training_mode.currentText(), "pretrain")

    def _training_stage_value(self) -> str:
        """Return the higher-level training stage selected in the UI.

        Returns:
            Training stage identifier.
        """

        return {
            "Pretrain from scratch": "base",
            "Fine-tune checkpoint": "domain",
            "Instruction fine-tune": "instruction",
            "Conversation fine-tune": "conversation",
            "Tool-call fine-tune": "tool_call",
            "Code fine-tune": "code",
        }.get(self.training_mode.currentText(), "base")

    def _peft_method_value(self) -> str:
        """Return the selected PEFT method identifier.

        Returns:
            Stable PEFT method used by the trainer.
        """

        return {
            "Full fine-tune": "none",
            "LoRA adapters": "lora",
        }.get(self.peft_method.currentText(), "none")

    def _lora_target_value(self) -> str:
        """Return selected LoRA target groups.

        Returns:
            Comma-separated target group string.
        """

        return {
            "Attention projections": "attention",
            "MLP projections": "mlp",
            "Attention + MLP": "attention,mlp",
        }.get(self.lora_targets.currentText(), "attention")

    def _update_training_mode_controls(self) -> None:
        """Enable fine-tune controls only when fine-tuning is selected."""

        enabled = self._training_mode_value() == "fine_tune"
        lora_enabled = enabled and self._peft_method_value() == "lora"
        self.fine_tune_checkpoint.setEnabled(enabled)
        self.peft_method.setEnabled(enabled)
        self.fine_tune_check_button.setEnabled(enabled)
        self.lora_rank.setEnabled(lora_enabled)
        self.lora_alpha.setEnabled(lora_enabled)
        self.lora_dropout.setEnabled(lora_enabled)
        self.lora_targets.setEnabled(lora_enabled)
        if not lora_enabled:
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.lora_dropout.setValue(0.05)
            self._set_combo_text(self.lora_targets, "Attention projections")
        self.refresh_fine_tune_workflow()

    def _current_dataset_summary(self) -> dict[str, Any]:
        """Read the active prepared dataset summary.

        Returns:
            Dataset summary dictionary, or an empty dictionary.
        """

        summary_path = Path(self.train_data_dir.text()) / "dataset_summary.json"
        if not summary_path.exists():
            summary_path = Path(self.dataset_dir.text()) / "dataset_summary.json"
        if not summary_path.exists():
            return {}
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            LOGGER.warning("Could not read dataset summary %s: %s", summary_path, exc)
            return {}

    def _fine_tune_dataset_stage_status(self) -> tuple[bool, str]:
        """Check whether the prepared dataset matches the fine-tune type.

        Returns:
            Tuple containing whether the workflow may proceed and a user-facing message.
        """

        expected_stage = self._training_stage_value()
        summary = self._current_dataset_summary()
        if not summary:
            return False, "Dataset: not prepared. Prepare the fine-tune dataset first."
        dataset_stage = str(summary.get("dataset_stage") or self._dataset_stage_value())
        tokens = int(summary.get("token_count", 0) or 0)
        vocab = int(summary.get("tokenizer_vocab_size", 0) or 0)
        stage_name = dataset_stage_label(dataset_stage) if dataset_stage in {"base", "instruction", "conversation", "tool_call", "code"} else dataset_stage
        details = f"{stage_name}, {tokens:,} tokens, vocab {vocab:,}"
        if expected_stage == "instruction" and dataset_stage != "instruction":
            return False, f"Dataset mismatch: selected Instruction fine-tune, but prepared dataset is {details}."
        if expected_stage == "conversation" and dataset_stage != "conversation":
            return False, f"Dataset mismatch: selected Conversation fine-tune, but prepared dataset is {details}."
        if expected_stage == "tool_call" and dataset_stage != "tool_call":
            return False, f"Dataset mismatch: selected Tool-call fine-tune, but prepared dataset is {details}."
        if expected_stage == "code" and dataset_stage != "code":
            return False, f"Dataset mismatch: selected Code fine-tune, but prepared dataset is {details}."
        if expected_stage == "domain" and dataset_stage == "base":
            return True, f"Dataset warning: {details}. Base datasets usually belong to pretraining; continue only for domain adaptation."
        return True, f"Dataset ready: {details}."

    def refresh_fine_tune_workflow(self) -> None:
        """Refresh fine-tune workflow guidance in the Fine-Tuning tab."""

        if not hasattr(self, "fine_tune_dataset_status"):
            return
        self._refresh_fine_tune_default_output()
        ok, message = self._fine_tune_dataset_stage_status()
        self.fine_tune_dataset_status.setText(message)
        self.fine_tune_dataset_status.setProperty("state", "ok" if ok else "warning")
        self.fine_tune_dataset_status.style().unpolish(self.fine_tune_dataset_status)
        self.fine_tune_dataset_status.style().polish(self.fine_tune_dataset_status)

    def apply_recommended_fine_tune_settings(self) -> None:
        """Apply conservative fine-tuning defaults for the selected workflow."""

        stage = self._training_stage_value()
        synced = self._sync_architecture_from_fine_tune_base()
        self._set_combo_text(self.peft_method, "LoRA adapters")
        self.lora_dropout.setValue(0.05)
        self._set_combo_text(self.lora_targets, "Attention projections")
        self.max_grad_norm.setValue(0.5)
        self.weight_decay.setValue(0.05)
        self._set_combo_by_data(self.scheduler_name, "cosine", {
            "warmup_linear": "Warmup linear",
            "cosine": "Cosine decay",
            "polynomial": "Polynomial decay",
            "one_cycle": "One-cycle",
            "constant": "Constant",
        })
        if stage == "conversation":
            self.lora_rank.setValue(16)
            self.lora_alpha.setValue(32.0)
            self.learning_rate.setValue(0.00003)
            self.epochs.setValue(max(1, min(self.epochs.value(), 2)))
        elif stage == "tool_call":
            self.lora_rank.setValue(16)
            self.lora_alpha.setValue(32.0)
            self.learning_rate.setValue(0.00003)
            self.epochs.setValue(max(1, min(self.epochs.value(), 3)))
        elif stage == "code":
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.lora_dropout.setValue(0.05)
            self.learning_rate.setValue(0.00005)
            self.max_grad_norm.setValue(0.5)
            self.epochs.setValue(max(1, min(self.epochs.value(), 3)))
        elif stage == "instruction":
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.learning_rate.setValue(0.00005)
            self.epochs.setValue(max(1, min(self.epochs.value(), 3)))
        else:
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.learning_rate.setValue(0.00005)
        self._update_training_mode_controls()
        message = "Recommended LoRA settings applied."
        if synced:
            message += "\nArchitecture was synced from the selected base checkpoint."
        message += "\nUse Check Fine-tune before starting so checkpoint and tokenizer compatibility are verified."
        self.fine_tune_preview.setText(message)

    def _sync_architecture_from_fine_tune_base(self) -> bool:
        """Sync architecture controls from the selected fine-tune base checkpoint.

        Returns:
            True when a checkpoint was read and architecture controls were updated.
        """

        if not hasattr(self, "fine_tune_checkpoint"):
            return False
        checkpoint_text = self.fine_tune_checkpoint.text().strip()
        if not checkpoint_text:
            return False
        checkpoint_path = Path(checkpoint_text)
        if not checkpoint_path.exists():
            return False
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            LOGGER.warning("Could not read fine-tune base checkpoint %s: %s", checkpoint_path, exc)
            return False
        model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
        if not isinstance(model_config, dict):
            return False
        mappings = {
            "embedding_size": self.n_embd,
            "head_count": self.n_head,
            "layer_count": self.n_layer,
            # NOT self.context_length -- that is the Dataset tab's tokenizer
            # window-size setting (DatasetConfig.context_length), an
            # unrelated dataset-preparation parameter. _current_model_config()
            # reads self.train_context_length for ModelConfig.context_length,
            # which is the field resume-compatibility actually checks.
            "context_length": self.train_context_length,
        }
        for key, widget in mappings.items():
            if key in model_config:
                try:
                    widget.setValue(int(model_config[key]))
                except (TypeError, ValueError):
                    LOGGER.warning("Invalid %s in checkpoint %s: %r", key, checkpoint_path, model_config[key])
        if "dropout" in model_config:
            try:
                self.dropout.setValue(float(model_config["dropout"]))
            except (TypeError, ValueError):
                LOGGER.warning("Invalid dropout in checkpoint %s: %r", checkpoint_path, model_config["dropout"])
        if "rope_theta" in model_config:
            try:
                self.rope_theta.setValue(float(model_config["rope_theta"]))
            except (TypeError, ValueError):
                LOGGER.warning("Invalid rope_theta in checkpoint %s: %r", checkpoint_path, model_config["rope_theta"])
        if "bias" in model_config:
            self.use_bias.setChecked(bool(model_config["bias"]))
        norm_type = str(model_config.get("norm_type", "layernorm")).lower()
        position_encoding = str(model_config.get("position_encoding", "learned")).lower()
        mlp_type = str(model_config.get("mlp_type", "gelu")).lower()
        if norm_type == "rmsnorm" or position_encoding == "rope" or mlp_type == "swiglu":
            # Must match training_tab.py's actual combo item text exactly
            # ("Llama-like") -- _set_combo_text() silently no-ops on a
            # non-editable combo when the text doesn't match any item, so a
            # wrong string here does not raise or log anything. It used to
            # say "Modern LLM", which does not exist as an option: this
            # left architecture_style un-synced while every other field
            # (n_embd, n_head, n_layer, ...) synced correctly, guaranteeing
            # a resume-compatibility mismatch on norm_type/position_encoding
            # /mlp_type with no indication of why.
            self._set_combo_text(self.architecture_style, "Llama-like")
        else:
            self._set_combo_text(self.architecture_style, "Classic GPT")
        attention_type = str(model_config.get("attention_type", "mha")).lower()
        self._set_combo_by_data(
            self.attention_type,
            attention_type,
            {
                "mha": "Multi-head",
                "mqa": "Multi-query",
                "gqa": "Grouped-query",
            },
        )
        if "kv_head_count" in model_config:
            try:
                self.kv_head_count.setValue(int(model_config["kv_head_count"]))
            except (TypeError, ValueError):
                LOGGER.warning("Invalid kv_head_count in checkpoint %s: %r", checkpoint_path, model_config["kv_head_count"])
        backend = str(model_config.get("attention_backend", "sdpa")).lower()
        self._set_combo_by_data(
            self.attention_backend,
            backend,
            {
                "sdpa": "SDPA / Flash when available",
                "eager": "PyTorch eager",
            },
        )
        if "attention_window" in model_config:
            try:
                self.attention_window.setValue(int(model_config["attention_window"]))
            except (TypeError, ValueError):
                LOGGER.warning("Invalid attention_window in checkpoint %s: %r", checkpoint_path, model_config["attention_window"])
        LOGGER.info("Fine-tune architecture synced from base checkpoint: %s", checkpoint_path)
        return True
