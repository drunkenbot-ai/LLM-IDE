from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart8:
    def _apply_project_state(self, data: dict[str, Any]) -> None:
        """Restore UI state from a saved project dictionary.

        Args:
            data: Project state loaded from JSON.
        """

        self.search_box.setText(str(data.get("project_name", "")))
        paths = data.get("paths", {})
        dataset = data.get("dataset", {})
        training = data.get("training", {})
        export = data.get("export", {})
        chat = data.get("chat", {})
        distributed = data.get("distributed", {})

        self.input_dir.setText(str(paths.get("source_vault", "")))
        self.dataset_dir.setText(str(paths.get("dataset_core", "")))
        self.train_data_dir.setText(str(paths.get("training_dataset", "")))
        self.model_dir.setText(str(paths.get("model_output", "")))
        self.export_model_dir.setText(str(paths.get("export_model_core", "")))
        self.export_dir.setText(str(paths.get("export_output", "")))
        self.llama_cpp_dir.setText(str(paths.get("llama_cpp_dir", "")))
        self.gguf_output_path.setText(str(paths.get("gguf_output_path", "")))
        self.gguf_path.setText(str(paths.get("gguf_model", "")))
        self.microgpt_chat_path.setText(str(paths.get("microgpt_chat_model", "")))
        self.tokenizer_path.setText(str(paths.get("tokenizer_import", "")))
        self.resume_checkpoint.setText(str(paths.get("resume_checkpoint", "")))
        self.fine_tune_checkpoint.setText(str(paths.get("fine_tune_checkpoint", "")))
        self.fine_tune_output_dir.setText(str(paths.get("fine_tune_output", "")))

        self._set_dataset_plan(
            dict(dataset.get("domain_plan", {})),
            str(dataset.get("domain_plan_preset", "Balanced Tiny LLM")),
        )
        saved_default_data_paths = dataset.get("default_data_paths")
        self._set_selected_default_data_paths(
            list(saved_default_data_paths) if saved_default_data_paths is not None else None
        )
        if hasattr(self, "external_dataset_dir") and dataset.get("external_dataset_dir"):
            self.external_dataset_dir.setText(str(dataset["external_dataset_dir"]))
        self._refresh_external_dataset_status()
        self.auto_vocab.setChecked(bool(dataset.get("auto_vocab", True)))
        self.manual_vocab_size.setValue(int(dataset.get("manual_vocab_size", self.manual_vocab_size.value())))
        include_conversation = bool(dataset.get("include_conversation_datasets", False))
        self._set_dataset_stage(str(dataset.get("dataset_stage", "base")))
        self.include_conversation_datasets.setChecked(include_conversation)
        self._set_selected_conversation_datasets(list(dataset.get("conversation_datasets", [])))
        self.conversation_sample_limit.setValue(int(dataset.get("conversation_sample_limit", self.conversation_sample_limit.value())))
        self._set_mixture_weights(dict(dataset.get("mixture_weights", {})))
        self.min_frequency.setValue(int(dataset.get("min_frequency", self.min_frequency.value())))
        self.context_length.setValue(int(dataset.get("context_length", self.context_length.value())))
        self.validation_split.setValue(float(dataset.get("validation_split", self.validation_split.value())))
        self.max_workers.setValue(int(dataset.get("max_workers", self.max_workers.value())))
        self._set_combo_by_data(self.prepare_mode, str(dataset.get("prepare_mode", "incremental")), {
            "incremental": "Incremental update",
            "full_rebuild": "Full rebuild",
            "force_reprocess": "Force reprocess",
        })
        self._set_combo_by_data(self.tokenizer_strategy, str(dataset.get("tokenizer_strategy", "auto")), {
            "auto": "Auto",
            "train_new": "Train new tokenizer",
            "reuse_dataset": "Reuse dataset tokenizer",
            "import_tokenizer": "Import tokenizer.json",
        })
        self.code_training_mode.setChecked(bool(dataset.get("code_training_mode", True)))
        self.include_prose.setChecked(bool(dataset.get("include_prose", True)))
        self.include_source_code.setChecked(bool(dataset.get("include_source_code", True)))
        self.extract_code_blocks.setChecked(bool(dataset.get("extract_code_blocks", True)))
        self.preserve_indentation.setChecked(bool(dataset.get("preserve_indentation", True)))
        self.instruction_samples.setChecked(bool(dataset.get("instruction_samples", True)))
        self._set_combo_by_data(self.reasoning_sample_mode, str(dataset.get("reasoning_sample_mode", "scaffold")), {
            "scaffold": "Reasoning scaffold",
            "detailed": "Detailed code reasoning",
            "none": "No reasoning wrapper",
        })

        self._set_combo_text(self.preset, str(training.get("preset", self.preset.currentText())))
        self._set_combo_text(self.architecture_style, str(training.get("architecture_style", self.architecture_style.currentText())))
        self._set_combo_by_data(self.training_launch_target, str(training.get("launch_target", "local")), {
            "local": "Local machine",
            "remote": "Remote workers",
            "runpod": "RunPod cloud",
        })
        if hasattr(self, "fine_tune_launch_target"):
            self._set_combo_by_data(self.fine_tune_launch_target, str(training.get("fine_tune_launch_target", "local")), {
                "local": "Local machine",
                "remote": "Remote workers",
                "runpod": "RunPod cloud",
            })
        self.n_embd.setValue(int(training.get("n_embd", self.n_embd.value())))
        self.n_head.setValue(int(training.get("n_head", self.n_head.value())))
        self._set_combo_by_data(self.attention_type, str(training.get("attention_type", "mha")), {
            "mha": "Multi-head",
            "gqa": "Grouped-query",
            "mqa": "Multi-query",
        })
        self.kv_head_count.setValue(int(training.get("kv_head_count", self.kv_head_count.value())))
        self._set_combo_by_data(self.attention_backend, str(training.get("attention_backend", "sdpa")), {
            "sdpa": "SDPA / Flash when available",
            "manual": "Manual",
        })
        self.attention_window.setValue(int(training.get("attention_window", self.attention_window.value())))
        self._set_combo_by_data(self.training_mode, str(training.get("training_mode", "pretrain")), {
            "pretrain": "Pretrain from scratch",
            "fine_tune": "Fine-tune checkpoint",
            "instruction_fine_tune": "Instruction fine-tune",
            "conversation_fine_tune": "Conversation fine-tune",
            "code_fine_tune": "Code fine-tune",
        })
        training_stage = str(training.get("training_stage", ""))
        if training_stage == "instruction":
            self._set_combo_text(self.training_mode, "Instruction fine-tune")
        elif training_stage == "conversation":
            self._set_combo_text(self.training_mode, "Conversation fine-tune")
        elif training_stage == "code":
            self._set_combo_text(self.training_mode, "Code fine-tune")
        self._set_combo_by_data(self.peft_method, str(training.get("peft_method", "none")), {
            "none": "Full fine-tune",
            "lora": "LoRA adapters",
        })
        self.lora_rank.setValue(int(training.get("lora_rank", self.lora_rank.value())))
        self.lora_alpha.setValue(float(training.get("lora_alpha", self.lora_alpha.value())))
        self.lora_dropout.setValue(float(training.get("lora_dropout", self.lora_dropout.value())))
        self._set_combo_by_data(self.lora_targets, str(training.get("lora_target_modules", "attention")), {
            "attention": "Attention projections",
            "mlp": "MLP projections",
            "attention,mlp": "Attention + MLP",
        })
        self.n_layer.setValue(int(training.get("n_layer", self.n_layer.value())))
        self.train_context_length.setValue(int(training.get("context_length", self.train_context_length.value())))
        self.dropout.setValue(float(training.get("dropout", self.dropout.value())))
        self._set_combo_text(self.training_profile, str(training.get("training_profile", self.training_profile.currentText())))
        self.epochs.setValue(int(training.get("epochs", self.epochs.value())))
        self.batch_size.setValue(int(training.get("batch_size", self.batch_size.value())))
        self.learning_rate.setValue(float(training.get("learning_rate", self.learning_rate.value())))
        self.weight_decay.setValue(float(training.get("weight_decay", self.weight_decay.value())))
        self._set_combo_by_data(self.optimizer_name, str(training.get("optimizer_name", "adamw")), {
            "adamw": "AdamW",
            "adam": "Adam",
            "lion": "Lion",
            "adafactor": "Adafactor",
        })
        self._set_combo_by_data(self.scheduler_name, str(training.get("scheduler_name", "warmup_linear")), {
            "warmup_linear": "Warmup linear",
            "cosine": "Cosine decay",
            "polynomial": "Polynomial decay",
            "one_cycle": "One-cycle",
            "constant": "Constant",
        })
        self.min_lr_ratio.setValue(float(training.get("scheduler_min_lr_ratio", self.min_lr_ratio.value())))
        self.polynomial_power.setValue(float(training.get("polynomial_power", self.polynomial_power.value())))
        self.gradient_accumulation.setValue(int(training.get("gradient_accumulation", self.gradient_accumulation.value())))
        self.sample_stride.setValue(int(training.get("sample_stride", self.sample_stride.value())))
        self.warmup_steps.setValue(int(training.get("warmup_steps", self.warmup_steps.value())))
        self.eval_interval.setValue(int(training.get("eval_interval", self.eval_interval.value())))
        self.max_eval_batches.setValue(int(training.get("max_eval_batches", self.max_eval_batches.value())))
        self.save_interval.setValue(int(training.get("save_interval", self.save_interval.value())))
        self.data_loader_workers.setValue(int(training.get("data_loader_workers", self.data_loader_workers.value())))
        self.max_grad_norm.setValue(float(training.get("max_grad_norm", self.max_grad_norm.value())))
        self.activation_checkpointing.setChecked(bool(training.get("activation_checkpointing", False)))
        self.seed.setValue(int(training.get("seed", self.seed.value())))
        self._set_combo_text(self.device, str(training.get("device", self.device.currentText())))
        self.use_amp.setChecked(bool(training.get("use_amp", self.use_amp.isChecked())))
        self._set_combo_by_data(self.precision, str(training.get("precision", "fp16")), {
            "fp16": "FP16",
            "bf16": "BF16",
            "fp32": "FP32",
        })
        self.resume_training.setChecked(bool(training.get("resume", self.resume_training.isChecked())))
        self.resume_safety.setChecked(bool(training.get("require_compatible_resume", True)))
        self.early_stopping.setChecked(bool(training.get("early_stopping", True)))
        self.benchmark_prompts.setPlainText(str(training.get("benchmark_prompts", self.benchmark_prompts.toPlainText())))
        self.benchmark_tokens.setValue(int(training.get("benchmark_tokens", self.benchmark_tokens.value())))
        self.benchmark_temperature.setValue(float(training.get("benchmark_temperature", self.benchmark_temperature.value())))
        self.benchmark_kv_cache.setChecked(bool(training.get("benchmark_kv_cache", True)))

        self._set_combo_text(self.quant_mode, str(export.get("quantization", self.quant_mode.currentText())))
        self._set_combo_text(self.gguf_outtype, str(export.get("gguf_outtype", self.gguf_outtype.currentText())))
        self.llama_context.setValue(int(chat.get("context", self.llama_context.value())))
        self._set_combo_by_data(self.chat_model_backend, str(chat.get("model_backend", "gguf")), {
            "gguf": "GGUF / llama.cpp",
            "microgpt": "MicroGPT checkpoint",
        })
        self.llama_threads.setValue(int(chat.get("cpu_threads", self.llama_threads.value())))
        self.llama_gpu_layers.setValue(int(chat.get("gpu_layers", self.llama_gpu_layers.value())))
        self.thinking_enabled.setChecked(bool(chat.get("thinking_enabled", True)))
        self._set_combo_text(self.reasoning_effort, str(chat.get("reasoning_effort", self.reasoning_effort.currentText())))
        self.reasoning_effort.setEnabled(self.thinking_enabled.isChecked())
        self.chat_max_tokens.setValue(int(chat.get("max_tokens", self.chat_max_tokens.value())))
        self.chat_temperature.setValue(float(chat.get("temperature", self.chat_temperature.value())))
        self.chat_top_p.setValue(float(chat.get("top_p", self.chat_top_p.value())))
        self.chat_repeat_penalty.setValue(float(chat.get("repeat_penalty", self.chat_repeat_penalty.value())))
        self.system_prompt.setPlainText(str(chat.get("system_prompt", "")))
        if hasattr(self, "coordinator_host"):
            self.coordinator_host.setText(str(distributed.get("host", self.coordinator_host.text())))
            self.coordinator_port.setValue(int(distributed.get("port", self.coordinator_port.value())))
            self.coordinator_artifact_root.setText(str(distributed.get("artifact_root", self.coordinator_artifact_root.text())))
            self.coordinator_public_url.setText(str(distributed.get("public_url", self.coordinator_public_url.text())))
        self._update_tokenizer_strategy_controls()
        self._update_training_mode_controls()
        self._restore_artifact_status(data.get("artifacts", {}))
        self.refresh_fine_tune_workflow()

    def _restore_artifact_status(self, artifacts: dict[str, Any]) -> None:
        """Refresh top-bar and button state from saved or existing artifacts.

        Args:
            artifacts: Saved artifact summary dictionary.
        """

        dataset_dir = Path(self.dataset_dir.text()) if self.dataset_dir.text().strip() else None
        if dataset_dir and self._dataset_artifacts_exist(dataset_dir):
            summary = self._read_json_if_exists(dataset_dir / "dataset_summary.json") or artifacts.get("dataset_summary") or {}
            document_count = int(summary.get("document_count", 0) or 0)
            token_count = int(summary.get("token_count", 0) or 0)
            code_count = int(summary.get("code_sample_count", 0) or 0)
            prose_count = int(summary.get("prose_sample_count", 0) or 0)
            conversation_count = int(summary.get("conversation_sample_count", 0) or 0)
            vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
            self._update_dataset_quality_report(summary)
            self.prepare_button.setText("DataSet Prepared")
            self.dataset_progress.setValue(100)
            if vocab_size:
                self.auto_vocab_label.setText(f"{vocab_size:,}")
            if code_count or prose_count or conversation_count:
                self.dataset_status.setText(
                    f"Dataset: {code_count:,} code, {prose_count:,} prose, {conversation_count:,} chat, {token_count:,} tokens"
                )
            elif document_count or token_count:
                self.dataset_status.setText(f"Dataset: {document_count:,} files, {token_count:,} tokens")
            else:
                self.dataset_status.setText("Dataset: prepared")
            version = summary.get("dataset_version", {})
            if isinstance(version, dict) and version.get("version_id"):
                self.dataset_log.append(f"Dataset version: {version['version_id']}")
            self.train_data_dir.setText(str(dataset_dir))
            self.dataset_log.append(f"Dataset already prepared: {dataset_dir}")
        else:
            self.prepare_button.setText("Prepare Dataset")
            self.dataset_progress.setValue(0)
            self.dataset_status.setText("Dataset: not prepared")
            self.auto_vocab_label.setText("Auto after reading files")
            self._reset_dataset_quality_report()

        model_dir = Path(self.model_dir.text()) if self.model_dir.text().strip() else None
        if model_dir and (model_dir / "final_model.pt").exists():
            summary = self._read_json_if_exists(model_dir / "training_summary.json") or artifacts.get("training_summary") or {}
            loss = summary.get("final_train_loss")
            self.train_status.setText(f"Training: loss {float(loss):.4f}" if loss is not None else "Training: model ready")
            self.export_model_dir.setText(str(model_dir))

        export_dir = Path(self.export_dir.text()) if self.export_dir.text().strip() else None
        if export_dir and export_dir.exists() and any(export_dir.iterdir()):
            self.export_status.setText("Export: artifacts found")

    @staticmethod
    def _dataset_artifacts_exist(dataset_dir: Path) -> bool:
        """Return whether a dataset folder has the required prepared files.

        Args:
            dataset_dir: Dataset folder.

        Returns:
            True if required dataset artifacts exist.
        """

        if not dataset_dir.exists():
            return False
        if not (dataset_dir / "tokenizer.json").exists():
            return False
        has_npy_tokens = (dataset_dir / "train_tokens.npy").exists() and (dataset_dir / "val_tokens.npy").exists()
        has_json_tokens = (dataset_dir / "train_tokens.json").exists() and (dataset_dir / "val_tokens.json").exists()
        return has_npy_tokens or has_json_tokens

    @staticmethod
    def _safe_project_name(project_name: str) -> str:
        """Return a filesystem-safe project folder name.

        Args:
            project_name: Raw user project name.

        Returns:
            Safe folder name.
        """

        return re.sub(r"[^A-Za-z0-9_.-]+", "_", project_name).strip("._") or "MicroLLMProject"

    @staticmethod
    def _read_json_if_exists(path: Path) -> Optional[Any]:
        """Read a JSON file when it exists.

        Args:
            path: JSON file path.

        Returns:
            Parsed JSON or ``None``.
        """

        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        """Set combo text when the value exists.

        Args:
            combo: Combo box to update.
            text: Display text to select.
        """

        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
        elif combo.isEditable():
            combo.setEditText(text)

    def _set_combo_by_data(self, combo: QComboBox, value: str, labels: dict[str, str]) -> None:
        """Set a combo by internal saved value.

        Args:
            combo: Combo box to update.
            value: Internal saved value.
            labels: Mapping from saved value to display label.
        """

        self._set_combo_text(combo, labels.get(value, value))

    def _run_task(
        self,
        fn,
        args,
        on_finished,
        log: QTextEdit,
        progress_bar: QProgressBar,
        with_progress: bool = False,
        button: Optional[QPushButton] = None,
        stop_button: Optional[QPushButton] = None,
        busy_text: str = "Working",
        task_kind: str = "",
        isolate_process: bool = False,
    ) -> None:
        """Run a long task on a background thread.

        Args:
            fn: Callable to execute.
            args: Positional arguments for the callable.
            on_finished: Slot called with the task result.
            log: Log widget receiving progress messages.
            progress_bar: Progress bar receiving percent updates.
            with_progress: Whether to pass a progress callback to the task.
            button: Optional button to disable while running.
            stop_button: Optional stop button to enable while running.
            busy_text: Button text shown while running.
            task_kind: Optional notification stage key.
            isolate_process: Run the task inside a child process.
        """

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please wait for the current task to finish.")
            return

        LOGGER.info("Starting background task: %s", getattr(fn, "__name__", str(fn)))
        self.active_task_kind = task_kind
        if button:
            self._set_button_busy(button, busy_text)
        if stop_button:
            stop_button.setEnabled(True)
            self.active_stop_button = stop_button

        self.stop_event = Event()
        self.progress_queue = Queue()
        self.active_log = log
        self.active_progress_bar = progress_bar
        self.thread = QThread(self)
        worker_class = ProcessTaskWorker if isolate_process else TaskWorker
        self.worker = worker_class(
            fn,
            *args,
            progress_queue=self.progress_queue,
            with_progress=with_progress,
            stop_event=self.stop_event,
        )
        self.result_bridge = WorkerSignalBridge(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.result_bridge.finished)
        self.result_bridge.finished.connect(on_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.result_bridge.failed)
        self.result_bridge.failed.connect(self._task_failed_from_worker)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.progress_timer.start(100)
        self.thread.start()

    @Slot(str)
    def _task_failed_from_worker(self, message: str) -> None:
        """Handle a worker failure on the UI thread.

        Args:
            message: Error message emitted by the worker.
        """

        if self.active_log is None or self.active_progress_bar is None:
            return
        LOGGER.error("Background task failed: %s", message)
        if self.active_task_kind == "chat":
            self.chat_status.setText(f"Chat: load failed - {message}")
        elif self.active_task_kind == "dataset_download":
            self.external_dataset_version.setText(f"Download failed: {message}")
            self.dataset_plan_progress.setVisible(False)
        self._task_failed(message, self.active_log, self.active_progress_bar)

