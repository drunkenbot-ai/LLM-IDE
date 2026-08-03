from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart14:
    def _update_model_estimate_chips(
        self,
        estimate: dict[str, Any],
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        train_tokens: int = 0,
    ) -> None:
        """Update model and VRAM estimate chips.

        Args:
            estimate: Estimate dictionary from the training planning service.
            model_config: Model architecture used for the estimate.
            training_config: Training options used for the estimate.
            train_tokens: Number of available training tokens.
        """

        params = int(estimate.get("parameters", 0))
        checkpoint_bytes = float(estimate.get("checkpoint_bytes", 0))
        vram_bytes = float(estimate.get("vram_bytes", 0))
        self.model_size_metric.setText(f"Model: {params / 1_000_000:.2f}M, ckpt {format_bytes(checkpoint_bytes)}")
        self.vram_estimate_metric.setText(f"VRAM est: {format_bytes(vram_bytes)}")
        parameter_breakdown = estimate.get("parameter_breakdown", {}) or {}
        memory_breakdown = estimate.get("memory_breakdown", {}) or {}
        embedding_params = int(parameter_breakdown.get("token_embedding", 0)) + int(
            parameter_breakdown.get("position_embedding", 0)
        )
        attention_params = int(parameter_breakdown.get("attention", 0))
        mlp_params = int(parameter_breakdown.get("mlp", 0))
        norm_params = int(parameter_breakdown.get("norms", 0))
        self.parameter_breakdown_metric.setText(
            "Params: "
            f"emb {self._compact_number(embedding_params)}, "
            f"attn {self._compact_number(attention_params)}, "
            f"mlp {self._compact_number(mlp_params)}"
        )
        self._tip(
            self.parameter_breakdown_metric,
            (
                f"Embedding: {embedding_params:,}\n"
                f"Attention: {attention_params:,}\n"
                f"MLP: {mlp_params:,}\n"
                f"Norms/output: {norm_params:,}\n"
                f"Total: {params:,}"
            ),
        )
        weights = float(memory_breakdown.get("weights", 0))
        optimizer = float(memory_breakdown.get("optimizer", 0))
        activations = float(memory_breakdown.get("activations", 0))
        kv_cache = float(memory_breakdown.get("kv_cache", 0))
        self.memory_breakdown_metric.setText(
            f"Memory: w {format_bytes(weights)}, opt {format_bytes(optimizer)}, act {format_bytes(activations)}"
        )
        self._tip(
            self.memory_breakdown_metric,
            (
                f"Weights: {format_bytes(weights)}\n"
                f"Optimizer state: {format_bytes(optimizer)}\n"
                f"Activations: {format_bytes(activations)}\n"
                f"KV cache estimate: {format_bytes(kv_cache)}\n"
                f"Total training estimate: {format_bytes(vram_bytes)}"
            ),
        )
        self._update_architecture_advisor(estimate, model_config, training_config, train_tokens)

    def _update_architecture_advisor(
        self,
        estimate: dict[str, Any],
        model_config: Optional[ModelConfig],
        training_config: Optional[TrainingConfig],
        train_tokens: int,
    ) -> None:
        """Update the compact architecture advisor chip.

        Args:
            estimate: Estimate dictionary from the training planning service.
            model_config: Model architecture used for the estimate.
            training_config: Training options used for the estimate.
            train_tokens: Number of available training tokens.
        """

        params = max(int(estimate.get("parameters", 0) or 0), 1)
        tokens_per_param = float(train_tokens) / float(params) if train_tokens > 0 else 0.0
        vram_bytes = float(estimate.get("vram_bytes", 0) or 0)
        notes: list[str] = []
        if tokens_per_param <= 0:
            label = "Advisor: prepare data"
            notes.append("Prepare a dataset to compare token budget against model size.")
        elif tokens_per_param < 20:
            label = "Advisor: data-light"
            notes.append(
                f"Token budget is about {tokens_per_param:.1f} tokens per parameter. More data or fewer epochs may reduce overfitting."
            )
        elif tokens_per_param > 150:
            label = "Advisor: data-rich"
            notes.append(
                f"Token budget is about {tokens_per_param:.1f} tokens per parameter. The model may be small for this much data."
            )
        else:
            label = "Advisor: balanced"
            notes.append(f"Token budget is about {tokens_per_param:.1f} tokens per parameter.")
        if model_config is not None:
            if model_config.context_length >= 2048 and model_config.embedding_size <= 256:
                notes.append("Long context with a small embedding can be memory-heavy without adding much capacity.")
            if model_config.attention_type in {"grouped_query", "multi_query"}:
                notes.append("Grouped/multi-query attention reduces KV memory and is useful for longer contexts.")
            if model_config.mlp_type == "swiglu" and model_config.norm_type == "rmsnorm":
                notes.append("Llama-like blocks improve modern compatibility but must match checkpoints when resuming.")
        if training_config is not None and training_config.device == "cuda" and vram_bytes > 3.5 * 1024**3:
            notes.append("Estimated VRAM is high for 4 GB GPUs. Try lower batch, context, embedding, or layers.")
            if label == "Advisor: balanced":
                label = "Advisor: memory check"
        self.architecture_advisor_metric.setText(label)
        self._tip(self.architecture_advisor_metric, "\n".join(notes))

    @staticmethod
    def _compact_number(value: int) -> str:
        """Format a large count for tight metric chips.

        Args:
            value: Count to format.

        Returns:
            Compact display string.
        """

        magnitude = abs(value)
        if magnitude >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"
        if magnitude >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if magnitude >= 1_000:
            return f"{value / 1_000:.1f}K"
        return str(value)

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

    def _training_run_artifacts(self, output_dir: Path) -> list[Path]:
        """Return training-run artifacts in a model output folder.

        Args:
            output_dir: Model output folder to inspect.

        Returns:
            Existing training-run artifact paths.
        """

        candidates = [
            output_dir / "checkpoints",
            output_dir / "final_model.pt",
            output_dir / "final_adapter.pt",
            output_dir / "training_summary.json",
            output_dir / "training_history.json",
            output_dir / "model_lineage.json",
        ]
        return [path for path in candidates if path.exists()]

    def _clear_training_run_artifacts(self, output_dir: Path) -> list[Path]:
        """Delete resumable training artifacts from a model output folder.

        Args:
            output_dir: Model output folder to clean.

        Returns:
            Paths that were removed.
        """

        output_dir = output_dir.resolve()
        removed: list[Path] = []
        candidates = self._training_run_artifacts(output_dir)
        for path in candidates:
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                resolved = path
            if output_dir not in resolved.parents and resolved != output_dir:
                LOGGER.warning("Skipped training cleanup outside model output folder: %s", path)
                continue
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(path)
            elif path.exists():
                path.unlink()
                removed.append(path)
        return removed

    def _selected_resume_path(self) -> Optional[Path]:
        """Return the selected or latest checkpoint path.

        Returns:
            Checkpoint path, or ``None`` when no checkpoint exists.
        """

        if self.resume_checkpoint.text().strip():
            return Path(self.resume_checkpoint.text())
        return latest_checkpoint(Path(self.model_dir.text()) / "checkpoints")

    def preview_resume_compatibility(self) -> None:
        """Preview whether the selected checkpoint can resume safely."""

        if not self.resume_training.isChecked():
            self.resume_training_preview.setText("[INFO] Resume latest is off. Enable resume to continue from a checkpoint.")
            return
        resume_path = self._selected_resume_path()
        if resume_path is None:
            self.resume_training_preview.setText("[INFO] No checkpoint found in the current model folder.")
            return
        if not resume_path.exists():
            self.resume_training_preview.setText(f"[BLOCK] Checkpoint does not exist:\n{resume_path}")
            return
        try:
            vocab_size = self._current_training_vocab_size(Path(self.train_data_dir.text()))
            if vocab_size <= 0:
                self.resume_training_preview.setText("[BLOCK] Could not determine current dataset tokenizer vocabulary size.")
                return
            model_config = self._current_model_config(vocab_size=vocab_size)
            # Explicit override: this is the AI/Training tab's own "Check
            # Resume" button, checking a pretrain checkpoint. Without this,
            # training_mode falls back to reading the separate Fine-Tuning
            # tab's mode combo (self.training_mode), which defaults to
            # "Instruction fine-tune" on a fresh session -- resolving to
            # "fine_tune" and making training_config.validate() below raise
            # "fine_tune_from_checkpoint is required for fine_tune mode",
            # a confusing error unrelated to what the user is checking.
            training_config = self._current_training_config(resume_path, training_mode="pretrain")
            model_config.validate()
            training_config.validate()
            report = check_resume_compatibility(resume_path, model_config, training_config)
            errors = list(report.errors)
            if training_config.require_compatible_resume:
                if not report.can_load_optimizer_state:
                    errors.append("Safe resume requires matching optimizer state.")
                if not report.can_load_scheduler_state:
                    errors.append("Safe resume requires matching scheduler state.")
                if not report.can_load_scaler_state:
                    errors.append("Safe resume requires matching AMP scaler state.")
            lines: list[str] = []
            if errors:
                lines.append("[BLOCK] Resume is not safe with the current settings.")
            elif report.warnings:
                lines.append("[WARN] Resume is possible, but settings changed.")
            else:
                lines.append("[OK] Checkpoint can resume with the current settings.")
            lines.extend(f"[OK] {line}" for line in report.info)
            lines.extend(f"[WARN] {line}" for line in report.warnings)
            lines.extend(f"[BLOCK] {line}" for line in errors)
            if not training_config.require_compatible_resume and not errors:
                lines.append("[INFO] Safe resume is off. Compatible weights will load; incompatible optimizer state may be skipped.")
            self.resume_training_preview.setText("\n".join(lines))
        except Exception as exc:
            self.resume_training_preview.setText(f"[BLOCK] Could not check resume compatibility:\n{exc}")


