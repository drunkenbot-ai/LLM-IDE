from __future__ import annotations

# FineTuningRunMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class FineTuningRunMixin:
    def _fine_tune_launch_target_value(self) -> str:
        """Return whether fine-tuning should launch locally or remotely.

        Returns:
            ``local`` or ``remote``.
        """

        if not hasattr(self, "fine_tune_launch_target"):
            return "local"
        if self.fine_tune_launch_target.currentText() == "RunPod cloud":
            return "runpod"
        return "remote" if self.fine_tune_launch_target.currentText() == "Remote workers" else "local"

    def _fine_tune_output_path(self) -> Path:
        """Return the selected fine-tune output folder.

        Returns:
            Folder where fine-tuned artifacts should be written.
        """

        text = self.fine_tune_output_dir.text().strip() if hasattr(self, "fine_tune_output_dir") else ""
        if text:
            path = Path(text)
        elif self.current_project_file is not None:
            path = self.current_project_file.parent / "fine_tunes" / "latest"
        else:
            path = Path(self.model_dir.text()) / "fine_tuned"
        try:
            if path.resolve() == Path(self.model_dir.text()).resolve():
                path = Path(self.model_dir.text()) / "fine_tuned"
        except OSError:
            pass
        if hasattr(self, "fine_tune_output_dir"):
            self.fine_tune_output_dir.setText(str(path))
        return path

    def _refresh_fine_tune_default_output(self, *_args: Any) -> None:
        """Keep the fine-tune output folder stage-specific unless a custom folder was chosen."""

        if not hasattr(self, "fine_tune_output_dir") or self.current_project_file is None:
            return
        project_dir = self.current_project_file.parent
        fine_tunes_dir = project_dir / "fine_tunes"
        stage = self._training_stage_value()
        stage_folder = {
            "instruction": "instruction_latest",
            "conversation": "conversation_latest",
            "tool_call": "tool_call_latest",
            "code": "code_latest",
            "domain": "domain_latest",
        }.get(stage, "fine_tune_latest")
        desired = fine_tunes_dir / stage_folder
        current_text = self.fine_tune_output_dir.text().strip()
        if not current_text:
            self.fine_tune_output_dir.setText(str(desired))
            return
        try:
            current = Path(current_text)
            current_resolved = current.resolve()
            fine_tunes_resolved = fine_tunes_dir.resolve()
        except OSError:
            return
        managed_names = {
            "latest",
            "fine_tune",
            "fine_tuned",
            "instruction",
            "conversation",
            "tool_call",
            "code",
            "domain",
            "instruction_latest",
            "conversation_latest",
            "tool_call_latest",
            "code_latest",
            "domain_latest",
            "fine_tune_latest",
        }
        if current_resolved.parent == fine_tunes_resolved and current.name in managed_names:
            self.fine_tune_output_dir.setText(str(desired))

    def _prepare_fine_tune_run_folder(self, training_config: TrainingConfig) -> None:
        """Create fine-tune folders and snapshot the base checkpoint.

        Args:
            training_config: Fine-tune training configuration.
        """

        output_dir = Path(training_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        base_checkpoint = training_config.fine_tune_from_checkpoint
        if base_checkpoint is None:
            return
        base_checkpoint = Path(base_checkpoint)
        if not base_checkpoint.exists():
            return
        try:
            base_resolved = base_checkpoint.resolve()
            output_resolved = output_dir.resolve()
            if base_resolved == (output_resolved / base_checkpoint.name) or output_resolved in base_resolved.parents:
                raise ValueError(
                    "Fine-tune base checkpoint must be outside the selected fine-tune output folder. "
                    "Choose the original pretrained model checkpoint instead."
                )
        except RuntimeError as exc:
            raise ValueError(f"Could not validate fine-tune base checkpoint path: {exc}") from exc
        snapshot_dir = output_dir / "base_model"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        copied_checkpoint = snapshot_dir / base_checkpoint.name
        if not copied_checkpoint.exists() or copied_checkpoint.stat().st_size != base_checkpoint.stat().st_size:
            shutil.copy2(base_checkpoint, copied_checkpoint)
        base_parent = base_checkpoint.parent
        for file_name in ("tokenizer.json", "training_summary.json", "model_lineage.json"):
            source = base_parent / file_name
            if source.exists():
                target = snapshot_dir / file_name
                if not target.exists() or target.stat().st_size != source.stat().st_size:
                    shutil.copy2(source, target)
        manifest = {
            "base_checkpoint": str(base_checkpoint),
            "copied_checkpoint": str(copied_checkpoint),
            "fine_tune_output": str(output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (snapshot_dir / "base_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.fine_tune_log.append(f"Base model snapshot: {copied_checkpoint}")

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

    def start_fine_tuning(self) -> None:
        """Collect fine-tuning options and start adaptation training."""

        fine_tune_launch = self._fine_tune_launch_target_value()
        if fine_tune_launch in {"remote", "runpod"}:
            stage_ok, stage_message = self._fine_tune_dataset_stage_status()
            self.refresh_fine_tune_workflow()
            if not stage_ok:
                self.fine_tune_log.append(stage_message)
                QMessageBox.warning(self, "Fine-tune blocked", stage_message)
                return
            if fine_tune_launch == "runpod":
                self.launch_runpod_worker_for_current_training(training_mode="fine_tune", stage=self._training_stage_value())
                self.fine_tune_log.append("RunPod fine-tune job launched. Watch Job Manager for worker assignment and progress.")
            else:
                self.publish_remote_training_job(training_mode="fine_tune", stage=self._training_stage_value())
                self.fine_tune_log.append("Remote fine-tune job queued. Watch Job Manager for worker assignment and progress.")
            return
        self.active_training_log = self.fine_tune_log
        self.active_training_progress = self.fine_tune_progress
        self.active_training_final_button_text = "Start Fine-Tune"
        stage_ok, stage_message = self._fine_tune_dataset_stage_status()
        self.refresh_fine_tune_workflow()
        if not stage_ok:
            self.fine_tune_log.append(stage_message)
            QMessageBox.warning(self, "Fine-tune blocked", stage_message)
            return
        resume_path = Path(self.resume_checkpoint.text()) if self.resume_checkpoint.text().strip() else None
        dataset_dir = Path(self.train_data_dir.text())
        vocab_size = self._current_training_vocab_size(dataset_dir)
        if vocab_size <= 0:
            QMessageBox.warning(self, "Fine-tune blocked", "Could not determine tokenizer vocabulary size. Prepare the fine-tuning dataset first.")
            return
        model_config = self._current_model_config(vocab_size=vocab_size)
        training_config = self._current_training_config(resume_path, training_mode="fine_tune")
        if not self._run_training_preflight(model_config, training_config):
            return
        self.active_training_output_dir = training_config.output_dir
        self._prepare_fine_tune_run_folder(training_config)
        self._init_telemetry_store(training_config.output_dir)
        self.fine_tune_log.append("")
        self.fine_tune_progress.setValue(0)
        self.training_progress.setValue(0)
        self.fine_tune_eta_metric.setText("ETA: -")
        self.fine_tune_epoch_metric.setText("Epoch: -")
        self.fine_tune_step_metric.setText("Step: -")
        self.fine_tune_loss_metric.setText("Train loss: -")
        self.fine_tune_val_metric.setText("Val loss: -")
        self.fine_tune_lr_metric.setText("LR: -")
        self.fine_tune_speed_metric.setText("Speed: -")
        self.fine_tune_grad_metric.setText("Grad: -")
        self.training_epoch_metric.setText("Epoch: -")
        self.training_step_metric.setText("Step: -")
        self.training_loss_metric.setText("Train loss: -")
        self.training_val_metric.setText("Val loss: -")
        self.training_health_metric.setText("Health: -")
        self.training_health_points = []
        self.training_lr_metric.setText("LR: -")
        self.training_speed_metric.setText("Speed: -")
        self.training_grad_metric.setText("Grad: -")
        self.training_vram_metric.setText("VRAM: -")
        self.training_eta_metric.setText("ETA: -")
        self.loss_chart.clear()
        self.optimization_chart.clear()
        self.stability_chart.clear()
        self.throughput_chart.clear()
        self.memory_chart.clear()
        self.live_progress.setValue(0)
        self.live_sample_text.setText("Training text: -")
        self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), 0, None)
        self.fine_tune_log.append("Fine-tuning started...")
        self.project_state.setText("Fine-tuning")
        self.train_status.setText("Training: fine-tuning")
        self._run_task(
            run_fine_tuning_job,
            (dataset_dir, model_config, training_config, self._training_stage_value()),
            self._training_finished,
            self.fine_tune_log,
            self.fine_tune_progress,
            with_progress=True,
            button=self.fine_tune_button,
            stop_button=self.stop_fine_tune_button,
            busy_text="Fine-tuning",
            task_kind="fine_tune",
        )
