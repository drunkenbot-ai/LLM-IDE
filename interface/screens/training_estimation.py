from __future__ import annotations

# TrainingEstimationMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class TrainingEstimationMixin:
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

    def _training_history_path(self) -> Path:
        """Return the training history path for the selected model folder.

        Returns:
            Path to ``training_history.json``.
        """

        output_dir = getattr(self, "active_training_output_dir", None)
        if output_dir is None:
            output_dir = Path(self.model_dir.text())
        return Path(output_dir) / "training_history.json"

    def _load_training_history(self) -> list[dict[str, Any]]:
        """Load training run history.

        Returns:
            List of training run entries.
        """

        path = self._training_history_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def refresh_model_estimate(self) -> None:
        """Refresh model size, rough VRAM, and run history widgets."""

        model_config = self._current_model_config()
        # Same reasoning as preview_resume_compatibility: this is the
        # AI/Training tab's shared "Model Estimate" card, not the
        # Fine-Tuning tab's; pass an explicit override rather than
        # inheriting the Fine-Tuning tab's mode combo by fallback.
        training_config = self._current_training_config(training_mode="pretrain")
        data_dir = Path(self.train_data_dir.text())
        train_tokens = max(model_config.context_length * training_config.batch_size, 1)
        try:
            summary_path = data_dir / "dataset_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
                train_tokens = int(summary.get("train_token_count", summary.get("token_count", train_tokens)) or train_tokens)
                if vocab_size > 0:
                    model_config.vocab_size = vocab_size
            elif (data_dir / "tokenizer.json").exists():
                tokenizer_data = json.loads((data_dir / "tokenizer.json").read_text(encoding="utf-8"))
                vocab = tokenizer_data.get("model", {}).get("vocab", {})
                if vocab:
                    model_config.vocab_size = len(vocab)
        except Exception as exc:
            self.training_log.append(f"[WARN] Could not refresh dataset-based estimate: {exc}")
        estimate = estimate_training_resources(model_config, training_config, train_tokens)
        self.last_training_estimate = estimate
        self._update_model_estimate_chips(estimate, model_config, training_config, train_tokens)
        self.history_metric.setText(f"Runs: {len(self._load_training_history())}")
        self.training_log.append(
            "Model estimate refreshed: "
            f"{int(estimate['parameters']):,} params, "
            f"checkpoint {format_bytes(float(estimate['checkpoint_bytes']))}, "
            f"VRAM {format_bytes(float(estimate['vram_bytes']))}."
        )

    def _append_training_history(self, result: Any) -> None:
        """Persist a training run entry to ``training_history.json``.

        Args:
            result: Training result object.
        """

        history_path = self._training_history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = self._load_training_history()
        summary = {}
        try:
            if Path(result.summary_path).exists():
                summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        estimate = getattr(self, "last_training_estimate", {}) or {}
        entry = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_path": str(result.checkpoint_path),
            "summary_path": str(result.summary_path),
            "stopped": bool(getattr(result, "stopped", False)),
            "final_train_loss": result.final_train_loss,
            "final_val_loss": result.final_val_loss,
            "best_val_loss": summary.get("best_val_loss"),
            "recommended_checkpoint_path": summary.get("recommended_checkpoint_path"),
            "best_checkpoint_path": summary.get("best_checkpoint_path"),
            "dataset_dir": self.train_data_dir.text(),
            "dataset_version": (summary.get("model_lineage") or {}).get("dataset_version"),
            "training_run_id": summary.get("training_run_id"),
            "parameters": estimate.get("parameters") or summary.get("parameters"),
            "model_config": summary.get("model_config"),
            "training_config": summary.get("training_config"),
        }
        history.append(entry)
        history_path.write_text(json.dumps(history[-200:], indent=2), encoding="utf-8")
        self.history_metric.setText(f"Runs: {len(history[-200:])}")
        (self.active_training_log or self.training_log).append(f"Training history updated: {history_path}")
