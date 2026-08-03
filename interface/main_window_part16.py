from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart16:
    def _run_training_preflight(self, model_config: ModelConfig, training_config: TrainingConfig) -> bool:
        """Run pre-training checklist and disk-space guard.

        Args:
            model_config: Selected model architecture.
            training_config: Selected training settings.

        Returns:
            True when training may continue.
        """

        log = self.active_training_log or self.training_log
        data_dir = Path(self.train_data_dir.text())
        output_dir = training_config.output_dir
        errors: list[str] = []
        warnings: list[str] = []
        info: list[str] = []
        resettable_errors: list[str] = []
        missing: list[str] = []
        if not (data_dir / "tokenizer.json").exists():
            missing.append("tokenizer.json")
        has_npy_tokens = (data_dir / "train_tokens.npy").exists() and (data_dir / "val_tokens.npy").exists()
        has_json_tokens = (data_dir / "train_tokens.json").exists() and (data_dir / "val_tokens.json").exists()
        if not has_npy_tokens and not has_json_tokens:
            missing.append("train_tokens.(npy/json), val_tokens.(npy/json)")
        if not data_dir.exists():
            errors.append(f"Dataset folder does not exist: {data_dir}")
        elif missing:
            errors.append(f"Dataset is not prepared. Missing: {', '.join(missing)}")
        else:
            info.append("Dataset artifacts found.")

        vocab_size = 0
        train_tokens = 0
        val_tokens = 0
        summary = {}
        try:
            summary_path = data_dir / "dataset_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
            if vocab_size <= 0:
                tokenizer_data = json.loads((data_dir / "tokenizer.json").read_text(encoding="utf-8"))
                vocab_size = len(tokenizer_data.get("model", {}).get("vocab", {}))
            train_tokens = int(summary.get("train_token_count", 0) or 0)
            val_tokens = int(summary.get("val_token_count", 0) or 0)
            if train_tokens <= 0 and (data_dir / "train_tokens.json").exists():
                train_tokens = len(json.loads((data_dir / "train_tokens.json").read_text(encoding="utf-8")))
            if val_tokens <= 0 and (data_dir / "val_tokens.json").exists():
                val_tokens = len(json.loads((data_dir / "val_tokens.json").read_text(encoding="utf-8")))
        except Exception as exc:
            warnings.append(f"Could not fully inspect dataset metadata: {exc}")

        if vocab_size > 0:
            model_config.vocab_size = vocab_size
            info.append(f"Tokenizer vocab: {vocab_size:,}.")
        elif not missing:
            errors.append("Could not determine tokenizer vocabulary size.")
        if train_tokens and train_tokens <= model_config.context_length:
            errors.append("Training token count must be larger than context length.")
        elif train_tokens:
            info.append(f"Training tokens: {train_tokens:,}; validation tokens: {val_tokens:,}.")
            if train_tokens < 50_000:
                warnings.append("Training token count is very small; expect smoke-test quality.")

        try:
            model_config.validate()
        except Exception as exc:
            errors.append(f"Model architecture is invalid: {exc}")
        try:
            training_config.validate()
        except Exception as exc:
            errors.append(f"Training options are invalid: {exc}")
        if model_config.attention_backend == "sdpa":
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                if training_config.device == "cuda" and torch.cuda.is_available():
                    flash_enabled = bool(getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)())
                    info.append("Attention backend: SDPA selected; Flash Attention may be used by PyTorch." if flash_enabled else "Attention backend: SDPA selected; CUDA flash kernel is not enabled.")
                else:
                    info.append("Attention backend: SDPA selected; CPU/backend fallback will be used if needed.")
            else:
                warnings.append("SDPA attention selected, but this PyTorch build does not expose scaled_dot_product_attention.")
        else:
            warnings.append("Manual attention backend selected. This is useful for debugging but can be slower.")
        if training_config.peft_method == "lora":
            info.append(
                "PEFT: LoRA adapters enabled. Intermediate checkpoints will save adapter weights; final_model.pt will be merged."
            )

        if training_config.device == "cuda" and not torch.cuda.is_available():
            errors.append("CUDA is selected, but PyTorch cannot use CUDA on this machine.")
        elif training_config.device == "cuda":
            info.append(f"CUDA ready: {torch.cuda.get_device_name(0)}.")
            if training_config.data_loader_workers > 0:
                info.append(f"CPU-assisted batch loading enabled with {training_config.data_loader_workers} worker(s).")
        else:
            warnings.append("CPU training is selected. This can be very slow.")
        if sys.platform.startswith("win") and training_config.data_loader_workers > 4:
            warnings.append("High CPU worker counts can duplicate dataset memory on Windows. Start with 2-4 workers and increase carefully.")

        active_resume_path: Optional[Path] = None
        resume_path = training_config.resume_from_checkpoint if training_config.resume else None
        if resume_path and not Path(resume_path).exists():
            errors.append(f"Selected resume checkpoint does not exist: {resume_path}")
        elif training_config.resume:
            if resume_path is None:
                resume_path = latest_checkpoint(output_dir / "checkpoints")
            if resume_path is None:
                info.append("Resume latest is enabled, but no checkpoint exists yet.")
            else:
                active_resume_path = Path(resume_path)
                try:
                    compatibility = check_resume_compatibility(active_resume_path, model_config, training_config)
                    info.extend(compatibility.info)
                    warnings.extend(compatibility.warnings)
                    errors.extend(compatibility.errors)
                    resettable_errors.extend(compatibility.errors)
                    if training_config.require_compatible_resume:
                        if not compatibility.can_load_optimizer_state:
                            message = "Safe resume requires matching optimizer state."
                            errors.append(message)
                            resettable_errors.append(message)
                        if not compatibility.can_load_scheduler_state:
                            message = "Safe resume requires matching scheduler state."
                            errors.append(message)
                            resettable_errors.append(message)
                        if not compatibility.can_load_scaler_state:
                            message = "Safe resume requires matching AMP scaler state."
                            errors.append(message)
                            resettable_errors.append(message)
                except Exception as exc:
                    errors.append(f"Could not inspect resume checkpoint: {exc}")
        if training_config.training_mode == "fine_tune" and active_resume_path is None:
            base_path = training_config.fine_tune_from_checkpoint
            if base_path is None:
                errors.append("Fine-tune mode requires a base checkpoint.")
            elif not Path(base_path).exists():
                errors.append(f"Fine-tune base checkpoint does not exist: {base_path}")
            else:
                try:
                    compatibility = check_resume_compatibility(Path(base_path), model_config, training_config)
                    info.append(f"Fine-tune base checkpoint: {Path(base_path).name}.")
                    warnings.extend(
                        warning for warning in compatibility.warnings
                        if not warning.startswith("Optimizer changed:") and not warning.startswith("LR scheduler changed:")
                    )
                    errors.extend(compatibility.errors)
                    checkpoint_vocab = self._checkpoint_vocab_size(Path(base_path))
                    if checkpoint_vocab and checkpoint_vocab != vocab_size:
                        errors.append(self._tokenizer_mismatch_help(checkpoint_vocab, vocab_size))
                    if not compatibility.errors:
                        info.append("Fine-tune base weights are compatible. Optimizer state will start fresh.")
                except Exception as exc:
                    errors.append(f"Could not inspect fine-tune base checkpoint: {exc}")
        elif training_config.training_mode == "pretrain" and active_resume_path is None:
            info.append("A fresh pretraining run will start from random weights.")
        elif training_config.training_mode == "fine_tune" and active_resume_path is not None:
            info.append("Existing run checkpoint found; training will resume that run instead of reloading the fine-tune base.")

        output_dir.mkdir(parents=True, exist_ok=True)
        estimate = estimate_training_resources(model_config, training_config, train_tokens)
        self.last_training_estimate = estimate
        self._update_model_estimate_chips(estimate, model_config, training_config, train_tokens)
        params = int(estimate["parameters"])
        checkpoint_bytes = float(estimate["checkpoint_bytes"])
        checkpoint_count = int(estimate["checkpoint_count"])
        estimated_storage = float(estimate["estimated_storage"])
        estimated_vram = float(estimate["vram_bytes"])
        free_bytes = shutil.disk_usage(output_dir).free
        info.append(f"Estimated parameters: {params:,}.")
        info.append(f"Estimated checkpoint size: {format_bytes(checkpoint_bytes)}.")
        info.append(f"Estimated training VRAM: {format_bytes(estimated_vram)}.")
        info.append(f"Estimated training storage need: {format_bytes(estimated_storage)}.")
        info.append(f"Free space on model drive: {format_bytes(free_bytes)}.")
        if training_config.device == "cuda" and torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info()
            info.append(f"GPU free/total VRAM: {format_bytes(free_vram)} / {format_bytes(total_vram)}.")
            if estimated_vram > free_vram * 0.9:
                warnings.append("Estimated VRAM is close to or above currently free GPU memory.")
        if free_bytes < estimated_storage * 1.25:
            errors.append("Not enough free disk space for estimated checkpoints and final model.")
        elif free_bytes < estimated_storage * 2:
            warnings.append("Free disk space is close to the estimated training storage need.")
        if checkpoint_count > 50:
            warnings.append("Save interval may create many checkpoints. Increase Save every or clean old checkpoints.")

        log.clear()
        log.append("Training checklist")
        for line in info:
            log.append(f"[OK] {line}")
        for line in warnings:
            log.append(f"[WARN] {line}")
        for line in errors:
            log.append(f"[ERROR] {line}")

        if errors:
            hard_errors = [error for error in errors if error not in resettable_errors]
            if active_resume_path is not None and resettable_errors and not hard_errors:
                message = (
                    "The existing checkpoint was created with different model settings, so it cannot be resumed.\n\n"
                    "This is expected if you intentionally changed architecture, block style, tokenizer, "
                    "context length, attention layout, or other checkpoint-shaped settings.\n\n"
                    "You can start a fresh training run with the current settings. This will delete old "
                    "checkpoints and training summaries in the selected model output folder.\n\n"
                    f"Model folder:\n{output_dir}\n\n"
                    "Continue and start from scratch?"
                )
                choice = QMessageBox.question(
                    self,
                    "Start From Scratch?",
                    message,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if choice == QMessageBox.Yes:
                    removed = self._clear_training_run_artifacts(output_dir)
                    training_config.resume = False
                    training_config.resume_from_checkpoint = None
                    self.resume_training.setChecked(False)
                    self.resume_checkpoint.clear()
                    log.append("")
                    log.append("Starting fresh run with current settings.")
                    for path in removed:
                        log.append(f"Removed old training artifact: {path}")
                    LOGGER.warning(
                        "User chose to discard incompatible resume checkpoint %s and start fresh in %s",
                        active_resume_path,
                        output_dir,
                    )
                    if training_config.training_mode == "fine_tune":
                        base_path = training_config.fine_tune_from_checkpoint
                        if base_path is None or not Path(base_path).exists():
                            log.append("[ERROR] Fine-tune mode requires an existing base checkpoint after reset.")
                            QMessageBox.warning(self, "Training blocked", "Fine-tune mode still needs a valid base checkpoint.")
                            return False
                        compatibility = check_resume_compatibility(Path(base_path), model_config, training_config)
                        if compatibility.errors:
                            for line in compatibility.errors:
                                log.append(f"[ERROR] {line}")
                            QMessageBox.warning(self, "Training blocked", "The base checkpoint is still incompatible with current settings.")
                            return False
                        log.append("[OK] Old run cleared; fine-tune base checkpoint is compatible.")
                    else:
                        log.append("[OK] Old run cleared; pretraining will start from random weights.")
                    self.project_state.setText("Training reset")
                    self.train_status.setText("Training: starting fresh")
                    return True
            LOGGER.error("Training blocked by preflight checklist.")
            for line in info:
                LOGGER.info("Training preflight OK: %s", line)
            for line in warnings:
                LOGGER.warning("Training preflight warning: %s", line)
            for line in errors:
                LOGGER.error("Training preflight error: %s", line)
            self.project_state.setText("Training blocked")
            self.train_status.setText("Training: blocked")
            QMessageBox.warning(self, "Training blocked", "Fix the checklist errors before starting training.")
            return False
        existing_artifacts = self._training_run_artifacts(output_dir)
        if (
            training_config.training_mode == "pretrain"
            and active_resume_path is None
            and existing_artifacts
        ):
            artifact_text = "\n".join(f"- {path.name}" for path in existing_artifacts)
            message = (
                "This model folder already contains training artifacts from a previous run.\n\n"
                "If you changed architecture or low-memory settings and want a clean start, "
                "the old checkpoints should be removed first.\n\n"
                f"Model folder:\n{output_dir}\n\n"
                f"Artifacts found:\n{artifact_text}\n\n"
                "Delete these artifacts and start from scratch?"
            )
            choice = QMessageBox.question(
                self,
                "Clean Previous Run?",
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                self.project_state.setText("Training cancelled")
                self.train_status.setText("Training: idle")
                log.append("Training cancelled. Previous run artifacts were kept.")
                return False
            removed = self._clear_training_run_artifacts(output_dir)
            training_config.resume = False
            training_config.resume_from_checkpoint = None
            self.resume_training.setChecked(False)
            self.resume_checkpoint.clear()
            log.append("")
            log.append("Previous run artifacts removed. Training will start from scratch with current settings.")
            for path in removed:
                log.append(f"Removed old training artifact: {path}")
            LOGGER.warning("User cleaned previous training artifacts in %s before starting from scratch.", output_dir)
        if warnings:
            message = "Training checklist has warnings. Continue anyway?\n\n" + "\n".join(f"- {warning}" for warning in warnings[:8])
            choice = QMessageBox.question(self, "Training warnings", message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if choice != QMessageBox.Yes:
                self.project_state.setText("Training cancelled")
                self.train_status.setText("Training: idle")
                return False
        return True

    def start_training(self) -> None:
        if not bool(QApplication.instance().property("license_valid")):
            self.train_context_length.setValue(min(self.train_context_length.value(), 1000))
        """Collect training options and start model training."""

        launch_target = self._training_launch_target_value()
        if launch_target == "runpod":
            self.launch_runpod_worker_for_current_training()
            return
        if launch_target == "remote":
            self.publish_remote_training_job()
            return
        self.active_training_log = self.training_log
        self.active_training_progress = self.training_progress
        self.active_training_final_button_text = "Start Training"
        resume_path = Path(self.resume_checkpoint.text()) if self.resume_checkpoint.text().strip() else None
        dataset_dir = Path(self.train_data_dir.text())
        vocab_size = self._current_training_vocab_size(dataset_dir)
        if vocab_size <= 0:
            QMessageBox.warning(self, "Training blocked", "Could not determine tokenizer vocabulary size. Prepare the dataset first.")
            return
        model_config = self._current_model_config(vocab_size=vocab_size)
        training_config = self._current_training_config(resume_path, training_mode="pretrain")
        if not self._run_training_preflight(model_config, training_config):
            return
        self.active_training_output_dir = training_config.output_dir
        self._init_telemetry_store(training_config.output_dir)
        self.training_log.append("")
        self.training_progress.setValue(0)
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
        self.live_prediction_chart.update_distribution(0, None)
        self.live_attention_chart.update_heatmap(0, None)
        self.live_activation_chart.update_histogram(0, None)
        self.live_gradient_chart.update_flow(self.n_layer.value(), None, 0)
        self.live_progress.setValue(0)
        self.live_epoch_metric.setText("Epoch: -")
        self.live_step_metric.setText("Step: -")
        self.live_tokens_metric.setText("Tokens/sec: -")
        self.live_loss_metric.setText("Loss: -")
        self.live_lr_metric.setText("LR: -")
        self.live_data_metric.setText("Data: -")
        self.live_sample_text.setText("Training text: -")
        self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), 0, None)
        self._set_meter(self.live_cpu_bar, "CPU", self._system_cpu_value())
        self._set_meter(self.live_gpu_bar, "GPU memory", None)
        self._set_meter(self.live_vram_bar, "VRAM reserved", None)
        self._set_meter(self.live_ram_bar, "System RAM", self._system_ram_value())
        self.live_worker_status.setText(f"CPU workers: {self.data_loader_workers.value()}")
        self.training_log.append("Training started...")
        self.project_state.setText("Training")
        self.train_status.setText("Training: running")
        self._run_task(
            run_training_job,
            (dataset_dir, model_config, training_config),
            self._training_finished,
            self.training_log,
            self.training_progress,
            with_progress=True,
            button=self.train_button,
            stop_button=self.stop_training_button,
            busy_text="Training",
            task_kind="training",
        )

