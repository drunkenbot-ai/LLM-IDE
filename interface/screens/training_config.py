from __future__ import annotations

# TrainingConfigMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class TrainingConfigMixin:
    def _current_model_config(self, vocab_size: int = 1) -> ModelConfig:
        """Build a model config from the current AI tab settings.

        Args:
            vocab_size: Tokenizer vocabulary size to use.

        Returns:
            Current model configuration.
        """

        return ModelConfig(
            vocab_size=vocab_size,
            context_length=self.train_context_length.value(),
            embedding_size=self.n_embd.value(),
            head_count=self.n_head.value(),
            layer_count=self.n_layer.value(),
            dropout=self.dropout.value(),
            bias=self.use_bias.isChecked(),
            attention_type=self._attention_type_value(),
            kv_head_count=self.kv_head_count.value(),
            attention_backend=self._attention_backend_value(),
            attention_window=self.attention_window.value(),
            **self._architecture_style_config(),
        )

    def _current_training_config(
        self,
        resume_path: Optional[Path] = None,
        training_mode: Optional[str] = None,
    ) -> TrainingConfig:
        """Build a training config from the current AI tab settings.

        Args:
            resume_path: Optional specific checkpoint to resume from.
            training_mode: Optional explicit training mode override.

        Returns:
            Current training configuration.
        """

        return TrainingConfig(
            output_dir=self._training_output_dir_for_mode(training_mode),
            epochs=self.epochs.value(),
            batch_size=self.batch_size.value(),
            learning_rate=self.learning_rate.value(),
            weight_decay=self.weight_decay.value(),
            optimizer_name=self._optimizer_value(),
            scheduler_name=self._scheduler_value(),
            scheduler_min_lr_ratio=self.min_lr_ratio.value(),
            polynomial_power=self.polynomial_power.value(),
            gradient_accumulation=self.gradient_accumulation.value(),
            sample_stride=self.sample_stride.value(),
            warmup_steps=self.warmup_steps.value(),
            eval_interval=self.eval_interval.value(),
            max_eval_batches=self.max_eval_batches.value(),
            save_interval=self.save_interval.value(),
            data_loader_workers=self.data_loader_workers.value(),
            max_grad_norm=self.max_grad_norm.value(),
            activation_checkpointing=self.activation_checkpointing.isChecked(),
            device=self.device.currentText(),
            use_amp=self.use_amp.isChecked(),
            precision=self._precision_value(),
            seed=self.seed.value(),
            training_mode=training_mode or self._training_mode_value(),
            fine_tune_from_checkpoint=(
                Path(self.fine_tune_checkpoint.text())
                if training_mode != "pretrain" and self.fine_tune_checkpoint.text().strip()
                else None
            ),
            peft_method="none" if training_mode == "pretrain" else self._peft_method_value(),
            lora_rank=self.lora_rank.value(),
            lora_alpha=self.lora_alpha.value(),
            lora_dropout=self.lora_dropout.value(),
            lora_target_modules=self._lora_target_value(),
            resume=self.resume_training.isChecked(),
            resume_from_checkpoint=resume_path if self.resume_training.isChecked() else None,
            require_compatible_resume=self.resume_safety.isChecked(),
            early_stopping=self.early_stopping.isChecked(),
            early_stopping_patience=self.early_stopping_patience.value(),
        )

    def _current_training_vocab_size(self, data_dir: Path) -> int:
        """Return the tokenizer vocabulary size for the current training dataset.

        Args:
            data_dir: Prepared dataset folder.

        Returns:
            Vocabulary size, or zero if unavailable.
        """

        summary_path = data_dir / "dataset_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
            if vocab_size > 0:
                return vocab_size
        tokenizer_path = data_dir / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
            vocab = tokenizer_data.get("model", {}).get("vocab", {})
            if isinstance(vocab, dict):
                return len(vocab)
        return 0

    def _checkpoint_vocab_size(self, checkpoint_path: Path) -> int:
        """Return the tokenizer vocabulary size saved in a checkpoint.

        Args:
            checkpoint_path: Checkpoint file to inspect.

        Returns:
            Saved vocabulary size, or zero when unavailable.
        """

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model_config = checkpoint.get("model_config", {})
            if isinstance(model_config, dict):
                return int(model_config.get("vocab_size", 0) or 0)
        except Exception as exc:
            LOGGER.warning("Could not inspect checkpoint vocab size for %s: %s", checkpoint_path, exc)
        return 0

    @staticmethod
    def _tokenizer_mismatch_help(checkpoint_vocab: int, dataset_vocab: int) -> str:
        """Return user-facing help for tokenizer mismatch errors.

        Args:
            checkpoint_vocab: Vocabulary size saved in the checkpoint.
            dataset_vocab: Vocabulary size in the prepared dataset.

        Returns:
            Help text.
        """

        return (
            f"Tokenizer mismatch: base checkpoint vocab is {checkpoint_vocab:,}, "
            f"but prepared dataset vocab is {dataset_vocab:,}.\n"
            "Fix: rebuild the fine-tune dataset using the exact tokenizer from the base model. "
            "In Ingest, set Tokenizer policy to Import tokenizer.json and choose the tokenizer.json "
            "beside the base checkpoint, then prepare the fine-tune dataset again."
        )
