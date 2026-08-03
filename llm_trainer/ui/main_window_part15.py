from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart15:
    def preview_fine_tune_compatibility(self) -> None:
        """Preview whether the selected checkpoint can be used for fine-tuning."""

        stage_ok, stage_message = self._fine_tune_dataset_stage_status()
        if not stage_ok:
            self.fine_tune_preview.setText(f"[BLOCK] {stage_message}")
            return
        base_path = Path(self.fine_tune_checkpoint.text()) if self.fine_tune_checkpoint.text().strip() else None
        if base_path is None:
            self.fine_tune_preview.setText("[BLOCK] Choose a base checkpoint for fine-tuning.")
            return
        if not base_path.exists():
            self.fine_tune_preview.setText(f"[BLOCK] Fine-tune base checkpoint does not exist:\n{base_path}")
            return
        try:
            vocab_size = self._current_training_vocab_size(Path(self.train_data_dir.text()))
            if vocab_size <= 0:
                self.fine_tune_preview.setText("[BLOCK] Could not determine current dataset tokenizer vocabulary size.")
                return
            model_config = self._current_model_config(vocab_size=vocab_size)
            training_config = self._current_training_config()
            model_config.validate()
            report = check_resume_compatibility(base_path, model_config, training_config)
            lines: list[str] = []
            if report.errors:
                lines.append("[BLOCK] Base checkpoint cannot be fine-tuned with the current model/dataset settings.")
            else:
                lines.append("[OK] Base checkpoint weights can be used for fine-tuning.")
            lines.append(f"[OK] {stage_message}" if stage_ok else f"[BLOCK] {stage_message}")
            lines.extend(self._fine_tune_lineage_advice(base_path))
            lines.extend(f"[OK] {line}" for line in report.info)
            behavior_warnings = [
                warning for warning in report.warnings
                if not warning.startswith("Optimizer changed:") and not warning.startswith("LR scheduler changed:")
            ]
            lines.extend(f"[WARN] {line}" for line in behavior_warnings)
            lines.extend(f"[BLOCK] {line}" for line in report.errors)
            checkpoint_vocab = self._checkpoint_vocab_size(base_path)
            if checkpoint_vocab and checkpoint_vocab != vocab_size:
                lines.append(f"[FIX] {self._tokenizer_mismatch_help(checkpoint_vocab, vocab_size)}")
            if not report.errors:
                lines.append("[INFO] Fine-tuning starts fresh optimizer, scheduler, and scaler state.")
            self.fine_tune_preview.setText("\n".join(lines))
        except Exception as exc:
            self.fine_tune_preview.setText(f"[BLOCK] Could not check fine-tune compatibility:\n{exc}")

    def _fine_tune_lineage_advice(self, base_path: Path) -> list[str]:
        """Return guidance about the selected fine-tune base checkpoint.

        Args:
            base_path: Selected checkpoint path.

        Returns:
            Lines for the fine-tune compatibility report.
        """

        lines: list[str] = []
        try:
            output_dir = self._fine_tune_output_path().resolve()
            base_resolved = base_path.resolve()
            if output_dir == base_resolved.parent or output_dir in base_resolved.parents:
                return [
                    "[BLOCK] Selected base checkpoint is inside the current fine-tune output folder.",
                    "[FIX] Choose the original pretrained model or a completed earlier fine-tune from another folder.",
                ]
        except OSError:
            pass
        lineage_path = base_path.parent / "model_lineage.json"
        summary_path = base_path.parent / "training_summary.json"
        lineage = read_json(lineage_path, default={}) or {}
        summary = read_json(summary_path, default={}) or {}
        training_mode = str(lineage.get("training_mode") or (summary.get("training_config") or {}).get("training_mode") or "")
        stage = str((summary.get("model_lineage") or lineage).get("fine_tune_stage") or "")
        if training_mode == "fine_tune":
            stage_text = f" ({stage})" if stage else ""
            lines.append(f"[INFO] Selected base is a previous fine-tuned checkpoint{stage_text}.")
            lines.append("[INFO] This is correct for cumulative tuning, such as conversation -> instruction -> code.")
        elif training_mode == "pretrain":
            lines.append("[OK] Selected base is the pretrained model checkpoint.")
            lines.append("[INFO] This is correct when starting a new independent fine-tune branch.")
        else:
            lines.append("[INFO] Could not read model lineage; compatibility check will still validate tensor shapes.")
        project_base = self.current_project_file.parent / "models" / "final_model.pt" if self.current_project_file else None
        if project_base and project_base.exists():
            try:
                if base_path.resolve() != project_base.resolve() and training_mode != "fine_tune":
                    lines.append(f"[HINT] Project pretrained model is: {project_base}")
            except OSError:
                pass
        return lines

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

