from __future__ import annotations

import os

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart13:
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
        stage_name = dataset_stage_label(dataset_stage) if dataset_stage in {"base", "instruction", "conversation", "code"} else dataset_stage
        details = f"{stage_name}, {tokens:,} tokens, vocab {vocab:,}"
        if expected_stage == "instruction" and dataset_stage != "instruction":
            return False, f"Dataset mismatch: selected Instruction fine-tune, but prepared dataset is {details}."
        if expected_stage == "conversation" and dataset_stage != "conversation":
            return False, f"Dataset mismatch: selected Conversation fine-tune, but prepared dataset is {details}."
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

    def _attention_type_value(self) -> str:
        """Return the selected attention layout identifier.

        Returns:
            Stable attention type used by the model.
        """

        return {
            "Multi-head": "mha",
            "Grouped-query": "gqa",
            "Multi-query": "mqa",
        }.get(self.attention_type.currentText(), "mha")

    def _attention_backend_value(self) -> str:
        """Return the selected attention backend identifier.

        Returns:
            Stable attention backend used by the model.
        """

        return {
            "SDPA / Flash when available": "sdpa",
            "Manual": "manual",
        }.get(self.attention_backend.currentText(), "sdpa")

    def _profile_architecture_scale(self) -> float:
        context = max(8, self.context_length.value())
        embedding = max(1, self.embedding_size.value())
        layers = max(1, self.n_layer.value())
        return max(0.5, min(8.0, (context / 512) * (embedding / 256) * (layers / 6)))

    def _apply_profile_runtime_defaults(self, profile: str) -> None:
        scale = self._profile_architecture_scale()
        cpu_count = max(1, os.cpu_count() or 1)
        workers = min(8, max(1, cpu_count // 2))
        profile_defaults = {
            "Low-memory": (128, 100, 16, 1000, 2),
            "Code fine-tune": (256, 50, 32, 500, 2),
            "Experimental Lion": (256, 100, 50, 500, 2),
            "Stable LLM": (128, 100, 50, 500, 2),
        }
        stride, eval_interval, eval_batches, save_interval, worker_divisor = profile_defaults.get(profile, profile_defaults["Stable LLM"])
        if scale >= 2.0:
            stride *= 2
            eval_interval *= 2
            save_interval *= 2
            eval_batches = max(8, eval_batches // 2)
        elif scale <= 0.75:
            eval_interval = max(25, eval_interval // 2)
            save_interval = max(100, save_interval // 2)
        target_effective_batch = {
            "Low-memory": 16,
            "Code fine-tune": 16,
            "Experimental Lion": 32,
            "Stable LLM": 32,
        }.get(profile, 32)
        if scale >= 4.0:
            batch_size = 2
        elif scale >= 2.0:
            batch_size = 4
        elif scale >= 1.25:
            batch_size = 8
        elif scale <= 0.75:
            batch_size = 32
        else:
            batch_size = 16
        batch_size = min(self.batch_size.maximum(), batch_size)
        self.batch_size.setValue(batch_size)
        accumulation = max(1, (target_effective_batch + batch_size - 1) // batch_size)
        self.gradient_accumulation.setValue(min(self.gradient_accumulation.maximum(), accumulation))
        self.sample_stride.setValue(min(self.sample_stride.maximum(), max(1, stride)))
        self.eval_interval.setValue(min(self.eval_interval.maximum(), max(0, eval_interval)))
        self.max_eval_batches.setValue(min(self.max_eval_batches.maximum(), max(0, eval_batches)))
        self.save_interval.setValue(min(self.save_interval.maximum(), max(1, save_interval)))
        self.data_loader_workers.setValue(min(self.data_loader_workers.maximum(), workers // worker_divisor))
        patience = {"Low-memory": 4, "Code fine-tune": 2, "Experimental Lion": 3, "Stable LLM": 3}.get(profile, 3)
        if eval_interval >= 200:
            patience += 1
        if eval_batches <= 16:
            patience += 1
        self.early_stopping_patience.setValue(min(self.early_stopping_patience.maximum(), patience))
        epochs = {"Low-memory": 5, "Code fine-tune": 3, "Experimental Lion": 4, "Stable LLM": 5}.get(profile, 5)
        self.epochs.setValue(min(self.epochs.maximum(), epochs))

    def apply_training_profile(self) -> None:
        """Apply the selected optimizer/scheduler/regularization profile.

        Each branch below explicitly sets every field it conceptually owns
        (optimizer, scheduler, LR/regularization, precision/memory knobs,
        batch shape, and early-stopping patience), even fields that happen
        to match the previous profile's value. This is deliberate: profiles
        must be idempotent when switched between, or a field set by a
        previously applied profile (e.g. activation_checkpointing=True from
        Low-memory) can silently survive into a later profile that never
        mentions it, producing a configuration no single profile actually
        intended.

        Two categories of fields are deliberately NOT touched here:
          - attention_type / kv_head_count: an architecture choice, not a
            training-strategy choice. Low-memory sets these to
            Grouped-query because that specific profile is about reducing
            memory end-to-end; the other profiles leave whatever the user
            has selected alone rather than silently reverting it.
          - training_mode / peft_method / lora_* (Code fine-tune only):
            these belong to the fine-tuning tab's widgets, not this tab's.
        """

        profile = self.training_profile.currentText()
        if profile == "Low-memory":
            self._set_combo_text(self.optimizer_name, "Adafactor")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.0002)
            self.weight_decay.setValue(0.05)
            self.min_lr_ratio.setValue(0.05)
            self.polynomial_power.setValue(1.0)
            self.max_grad_norm.setValue(1.0)
            self._set_combo_text(self.precision, "BF16" if torch.cuda.is_available() else "FP32")
            self.use_amp.setChecked(True)
            self._set_combo_text(self.attention_type, "Grouped-query")
            self.kv_head_count.setValue(max(1, self.n_head.value() // 2))
            self.activation_checkpointing.setChecked(True)
            # The two knobs that most directly control peak memory: shrink
            # the batch and make up the lost effective batch size with
            # gradient accumulation, and avoid extra data-loader worker
            # processes competing for memory.
            self.batch_size.setValue(4)
            self.gradient_accumulation.setValue(4)
            self.data_loader_workers.setValue(0)
            self.warmup_steps.setValue(100)
            self.dropout.setValue(0.1)
            self.early_stopping_patience.setValue(3)
        elif profile == "Code fine-tune":
            self._set_combo_text(self.optimizer_name, "AdamW")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.00005)
            self.weight_decay.setValue(0.05)
            self.min_lr_ratio.setValue(0.1)
            self.polynomial_power.setValue(1.0)
            self.max_grad_norm.setValue(0.5)
            self._set_combo_text(self.precision, "FP16")
            self.use_amp.setChecked(True)
            self.activation_checkpointing.setChecked(False)
            self.batch_size.setValue(16)
            self.gradient_accumulation.setValue(1)
            self.data_loader_workers.setValue(0)
            self.warmup_steps.setValue(50)
            self.dropout.setValue(0.05)
            # Fine-tuning generally needs less patience than a full
            # pretraining run before validation loss plateaus meaningfully.
            self.early_stopping_patience.setValue(2)
            self._set_combo_text(self.training_mode, "Fine-tune checkpoint")
            self._set_combo_text(self.peft_method, "LoRA adapters")
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.lora_dropout.setValue(0.05)
            self._set_combo_text(self.lora_targets, "Attention projections")
        elif profile == "Experimental Lion":
            self._set_combo_text(self.optimizer_name, "Lion")
            self._set_combo_text(self.scheduler_name, "One-cycle")
            self.learning_rate.setValue(0.0001)
            self.weight_decay.setValue(0.1)
            self.min_lr_ratio.setValue(0.01)
            self.polynomial_power.setValue(1.0)
            self.max_grad_norm.setValue(1.0)
            # Lion is reported to be more sensitive to fp16 under/overflow
            # than AdamW; prefer bf16 where available, fp32 otherwise.
            self._set_combo_text(self.precision, "BF16" if torch.cuda.is_available() else "FP32")
            self.use_amp.setChecked(True)
            self.activation_checkpointing.setChecked(False)
            self.batch_size.setValue(16)
            self.gradient_accumulation.setValue(1)
            self.data_loader_workers.setValue(0)
            self.warmup_steps.setValue(100)
            self.dropout.setValue(0.1)
            self.early_stopping_patience.setValue(3)
        else:
            self._set_combo_text(self.optimizer_name, "AdamW")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.0003)
            self.weight_decay.setValue(0.1)
            self.min_lr_ratio.setValue(0.1)
            self.polynomial_power.setValue(1.0)
            self.max_grad_norm.setValue(1.0)
            self._set_combo_text(self.precision, "FP16")
            self.use_amp.setChecked(True)
            self.activation_checkpointing.setChecked(False)
            self.batch_size.setValue(16)
            self.gradient_accumulation.setValue(1)
            self.data_loader_workers.setValue(0)
            self.warmup_steps.setValue(100)
            self.dropout.setValue(0.1)
            self.early_stopping_patience.setValue(3)
        self._apply_profile_runtime_defaults(profile)
        self._update_training_mode_controls()
        self.refresh_model_estimate()
        self.training_log.append(f"Applied training profile: {profile}")

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
