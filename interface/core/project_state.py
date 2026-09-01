from __future__ import annotations

# ProjectStateMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class ProjectStateMixin:
    def _default_project_state(self) -> dict[str, Any]:
        """Build the default state used for a newly created project.

        Returns:
            JSON-style project state with fresh paths and default settings.
        """

        runs_dir = Path.cwd() / "runs"
        dataset_dir = runs_dir / "dataset"
        model_dir = runs_dir / "model"
        fine_tune_dir = runs_dir / "fine_tune"
        export_dir = runs_dir / "export"
        return {
            "schema": "drunkenbot_ide_project",
            "version": 1,
            "theme": "dark",
            "theme_preference_version": 1,
            "project_name": "",
            "project_dir": "",
            "paths": {
                "source_vault": "",
                "dataset_core": str(dataset_dir),
                "training_dataset": str(dataset_dir),
                "model_output": str(model_dir),
                "export_model_core": str(model_dir),
                "export_output": str(export_dir),
                "llama_cpp_dir": "",
                "gguf_output_path": str(export_dir / "model.gguf"),
                "gguf_model": "",
                "microgpt_chat_model": "",
                "tokenizer_import": "",
                "resume_checkpoint": "",
                "fine_tune_checkpoint": "",
                "fine_tune_output": str(fine_tune_dir),
            },
            "dataset": {
                "domain_plan_preset": "Balanced Tiny LLM",
                "domain_plan": dataset_plan_defaults(),
                "default_data_paths": [str(path) for path, _category in iter_default_data_files()],
                "auto_vocab": True,
                "manual_vocab_size": 8000,
                "include_conversation_datasets": False,
                "dataset_stage": "base",
                "conversation_datasets": [],
                "conversation_sample_limit": 20000,
                "conversation_dataset_path": "",
                "instruction_dataset_path": "",
                "mixture_weights": {},
                "min_frequency": 2,
                "context_length": 128,
                "validation_split": 0.1,
                "lowercase": False,
                "max_workers": 4,
                "prepare_mode": "incremental",
                "tokenizer_strategy": "auto",
                "code_training_mode": True,
                "include_prose": True,
                "include_source_code": True,
                "extract_code_blocks": True,
                "preserve_indentation": True,
                "instruction_samples": True,
                "reasoning_sample_mode": "scaffold",
            },
            "training": {
                "preset": "Tiny",
                "architecture_style": "Classic GPT",
                "launch_target": "local",
                "training_mode": "pretrain",
                "training_stage": "base",
                "peft_method": "none",
                "lora_rank": 8,
                "lora_alpha": 16.0,
                "lora_dropout": 0.05,
                "lora_target_modules": "attention",
                "n_embd": 128,
                "n_head": 4,
                "n_layer": 4,
                "context_length": 128,
                "dropout": 0.1,
                "training_profile": "Stable LLM",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.0003,
                "weight_decay": 0.1,
                "gradient_accumulation": 1,
                "warmup_steps": 100,
                "eval_interval": 100,
                "max_eval_batches": 50,
                "save_interval": 500,
                "data_loader_workers": 0,
                "max_grad_norm": 1.0,
                "activation_checkpointing": False,
                "seed": 1337,
                "device": self.device.currentText(),
                "use_amp": self.use_amp_default,
                "resume": True,
                "require_compatible_resume": True,
                "benchmark_prompts": "\n\n".join(DEFAULT_BENCHMARK_PROMPTS),
                "benchmark_tokens": 128,
                "benchmark_temperature": 0.7,
                "benchmark_kv_cache": True,
            },
            "export": {
                "quantization": "FP16 checkpoint",
                "gguf_outtype": "f16",
            },
            "chat": {
                "model_backend": "gguf",
                "context": 2048,
                "cpu_threads": 4,
                "gpu_layers": -1,
                "thinking_enabled": True,
                "reasoning_effort": "Balanced",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "system_prompt": "",
            },
            "distributed": {
                "host": "0.0.0.0",
                "port": 8765,
                "artifact_root": str(Path.home() / ".drunkenbot_ide" / "artifacts"),
                "public_url": "http://127.0.0.1:8765",
            },
            "artifacts": {},
        }

    def _reset_project_runtime_state(self) -> None:
        """Clear logs, progress, charts, and status labels for a new project."""

        self.dataset_log.clear()
        self.training_log.clear()
        self.fine_tune_log.clear()
        self.benchmark_log.clear()
        self.export_log.setPlainText(
            "Export options:\n"
            "- Bundle copies final_model.pt, tokenizer.json, and training_summary.json.\n"
            "- HF package writes model_core/hf_model for portable MicroGPT loading.\n"
            "- FP16 checkpoint quantization works now.\n"
            "- GGUF conversion uses llama.cpp when model_core/hf_model exists.\n"
            "- Native MicroGPT checkpoints are not written as fake GGUF files.\n"
        )
        for progress in (
            self.dataset_progress,
            self.training_progress,
            self.fine_tune_progress,
            self.benchmark_progress,
            self.export_progress,
            self.chat_progress,
        ):
            progress.setRange(0, 100)
            progress.setValue(0)
        self.dataset_status.setText("Dataset: not prepared")
        self.train_status.setText("Training: idle")
        self.export_status.setText("Export: waiting")
        self.chat_status.setText("Chat: no model loaded")
        self.prepare_button.setText("Prepare Dataset")
        self.train_button.setText("Start Training")
        self.fine_tune_button.setText("Start Fine-Tune")
        self.stop_dataset_button.setEnabled(False)
        self.stop_training_button.setEnabled(False)
        self.stop_fine_tune_button.setEnabled(False)
        self.training_process_status.setText("Worker: detached | Run: - | PID: -")
        self.fine_tune_process_status.setText("Worker: detached | Run: - | PID: -")
        self.stop_benchmark_button.setEnabled(False)
        self.stop_chat_button.setEnabled(False)
        self.load_llm_button.setText("Load Model")
        self._update_chat_backend_controls()
        self._reset_dataset_quality_report()
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
        self.model_size_metric.setText("Model: -")
        self.vram_estimate_metric.setText("VRAM est: -")
        self.parameter_breakdown_metric.setText("Params: -")
        self.memory_breakdown_metric.setText("Memory: -")
        self.architecture_advisor_metric.setText("Advisor: -")
        self.history_metric.setText(f"Runs: {len(self._load_training_history())}")
        self.loss_chart.clear()
        self.optimization_chart.clear()
        self.stability_chart.clear()
        self.throughput_chart.clear()
        self.memory_chart.clear()
        self.live_prediction_chart.update_distribution(0, None)
        self.live_attention_chart.update_heatmap(0, None)
        self.live_activation_chart.update_histogram(0, None)
        self.live_gradient_chart.update_flow(self.n_layer.value(), None, 0)
        self.live_sample_text.setText("Training text: -")
        self.telemetry_db_path = None
        self.telemetry_run_id = ""
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_scrub_active = False
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_time_slider.blockSignals(False)
        self.live_timeline_label.setText("Timeline: no saved telemetry")
        self._set_meter(self.live_cpu_bar, "CPU", self._system_cpu_value())
        self._set_meter(self.live_training_cpu_bar, "Training process CPU", None)
        self._set_meter(self.live_ui_cpu_bar, "UI process CPU", None)
        self._set_meter(self.live_gpu_bar, "GPU memory", None)
        self._set_meter(self.live_vram_bar, "VRAM reserved", None)
        self._set_meter(self.live_ram_bar, "System RAM", self._system_ram_value())
        self.live_worker_status.setText(f"CPU workers: {self.data_loader_workers.value()}")
        self._clear_chat_messages()
        self.chat_markdown = ""
        self.chat_stream_prefix = ""
        self.chat_stream_reply = ""
        self.chat_stats.setText("Idle")
        self._add_chat_message("assistant", "Load a GGUF or MicroGPT model to start testing.")

    def _project_state_dict(self, project_name: str, project_dir: Path) -> dict[str, Any]:
        """Collect all UI state that defines a Micro LLM project.

        Args:
            project_name: User-facing project name.
            project_dir: Folder where the project file will live.

        Returns:
            JSON-serializable project state.
        """

        dataset_dir = Path(self.dataset_dir.text()) if self.dataset_dir.text().strip() else None
        model_dir = Path(self.model_dir.text()) if self.model_dir.text().strip() else None
        export_dir = Path(self.export_dir.text()) if self.export_dir.text().strip() else None
        now_iso = datetime.now().isoformat(timespec="seconds")
        created_at = now_iso
        existing_project_file = project_dir / "project.json"
        if existing_project_file.exists():
            try:
                existing_data = json.loads(existing_project_file.read_text(encoding="utf-8"))
            except Exception:
                existing_data = {}
            if isinstance(existing_data, dict):
                # Preserve the original creation timestamp across saves.
                # "saved_at" below is overwritten every save, so it cannot be
                # used as a creation date; fall back to it only for projects
                # saved before this field existed.
                created_at = str(existing_data.get("created_at") or existing_data.get("saved_at") or now_iso)
        return {
            "schema": "drunkenbot_ide_project",
            "version": 1,
            "theme": self.theme_name,
            "theme_preference_version": 1,
            "project_name": project_name,
            "project_dir": str(project_dir),
            "created_at": created_at,
            "saved_at": now_iso,
            "paths": {
                "source_vault": self.input_dir.text(),
                "dataset_core": self.dataset_dir.text(),
                "training_dataset": self.train_data_dir.text(),
                "model_output": self.model_dir.text(),
                "export_model_core": self.export_model_dir.text(),
                "export_output": self.export_dir.text(),
                "llama_cpp_dir": self.llama_cpp_dir.text(),
                "gguf_output_path": self.gguf_output_path.text(),
                "gguf_model": self.gguf_path.text(),
                "microgpt_chat_model": self.microgpt_chat_path.text(),
                "tokenizer_import": self.tokenizer_path.text(),
                "resume_checkpoint": self.resume_checkpoint.text(),
                "fine_tune_checkpoint": self.fine_tune_checkpoint.text(),
                "fine_tune_output": self.fine_tune_output_dir.text(),
            },
            "dataset": {
                "domain_plan_preset": self.dataset_plan_preset.currentText() if hasattr(self, "dataset_plan_preset") else "Balanced Tiny LLM",
                "domain_plan": self._dataset_plan_from_ui(),
                "default_data_paths": [str(path) for path in self._selected_default_data_paths()],
                "external_dataset_dir": self.external_dataset_dir.text() if hasattr(self, "external_dataset_dir") else "",
                "auto_vocab": self.auto_vocab.isChecked(),
                "manual_vocab_size": self.manual_vocab_size.value(),
                "include_conversation_datasets": self.include_conversation_datasets.isChecked(),
                "dataset_stage": self._dataset_stage_value(),
                "conversation_datasets": self._selected_conversation_datasets(),
                "conversation_sample_limit": self.conversation_sample_limit.value(),
                "mixture_weights": self._mixture_weights_from_ui(),
                "min_frequency": self.min_frequency.value(),
                "context_length": self.context_length.value(),
                "validation_split": self.validation_split.value(),
                "lowercase": False,
                "max_workers": self.max_workers.value(),
                "prepare_mode": self._prepare_mode_value(),
                "tokenizer_strategy": self._tokenizer_strategy_value(),
                "code_training_mode": self.code_training_mode.isChecked(),
                "include_prose": self.include_prose.isChecked(),
                "include_source_code": self.include_source_code.isChecked(),
                "extract_code_blocks": self.extract_code_blocks.isChecked(),
                "preserve_indentation": self.preserve_indentation.isChecked(),
                "instruction_samples": self.instruction_samples.isChecked(),
                "reasoning_sample_mode": self._reasoning_sample_mode_value(),
            },
            "training": {
                "preset": self.preset.currentText(),
                "architecture_style": self.architecture_style.currentText(),
                "launch_target": self._training_launch_target_value(),
                "fine_tune_launch_target": self._fine_tune_launch_target_value(),
                "training_stage": self._training_stage_value(),
                "n_embd": self.n_embd.value(),
                "n_head": self.n_head.value(),
                "attention_type": self._attention_type_value(),
                "kv_head_count": self.kv_head_count.value(),
                "attention_backend": self._attention_backend_value(),
                "attention_window": self.attention_window.value(),
                "training_mode": self._training_mode_value(),
                "peft_method": self._peft_method_value(),
                "lora_rank": self.lora_rank.value(),
                "lora_alpha": self.lora_alpha.value(),
                "lora_dropout": self.lora_dropout.value(),
                "lora_target_modules": self._lora_target_value(),
                "n_layer": self.n_layer.value(),
                "context_length": self.train_context_length.value(),
                "dropout": self.dropout.value(),
                "training_profile": self.training_profile.currentText(),
                "epochs": self.epochs.value(),
                "batch_size": self.batch_size.value(),
                "learning_rate": self.learning_rate.value(),
                "weight_decay": self.weight_decay.value(),
                "optimizer_name": self._optimizer_value(),
                "scheduler_name": self._scheduler_value(),
                "scheduler_min_lr_ratio": self.min_lr_ratio.value(),
                "polynomial_power": self.polynomial_power.value(),
                "gradient_accumulation": self.gradient_accumulation.value(),
                "sample_stride": self.sample_stride.value(),
                "warmup_steps": self.warmup_steps.value(),
                "eval_interval": self.eval_interval.value(),
                "max_eval_batches": self.max_eval_batches.value(),
                "save_interval": self.save_interval.value(),
                "data_loader_workers": self.data_loader_workers.value(),
                "max_grad_norm": self.max_grad_norm.value(),
                "activation_checkpointing": self.activation_checkpointing.isChecked(),
                "seed": self.seed.value(),
                "device": self.device.currentText(),
                "use_amp": self.use_amp.isChecked(),
                "precision": self._precision_value(),
                "resume": self.resume_training.isChecked(),
                "require_compatible_resume": self.resume_safety.isChecked(),
                "early_stopping": self.early_stopping.isChecked(),
                "benchmark_prompts": self.benchmark_prompts.toPlainText(),
                "benchmark_tokens": self.benchmark_tokens.value(),
                "benchmark_temperature": self.benchmark_temperature.value(),
                "benchmark_kv_cache": self.benchmark_kv_cache.isChecked(),
            },
            "export": {
                "quantization": self.quant_mode.currentText(),
                "gguf_outtype": self.gguf_outtype.currentText(),
            },
            "chat": {
                "model_backend": self._chat_backend_value(),
                "context": self.llama_context.value(),
                "cpu_threads": self.llama_threads.value(),
                "gpu_layers": self.llama_gpu_layers.value(),
                "thinking_enabled": self.thinking_enabled.isChecked(),
                "reasoning_effort": self.reasoning_effort.currentText(),
                "max_tokens": self.chat_max_tokens.value(),
                "temperature": self.chat_temperature.value(),
                "top_p": self.chat_top_p.value(),
                "repeat_penalty": self.chat_repeat_penalty.value(),
                "system_prompt": self.system_prompt.toPlainText(),
            },
            "distributed": {
                "host": self.coordinator_host.text(),
                "port": self.coordinator_port.value(),
                "artifact_root": self.coordinator_artifact_root.text(),
                "public_url": self.coordinator_public_url.text(),
            },
            "artifacts": {
                "dataset_summary": self._read_json_if_exists(dataset_dir / "dataset_summary.json") if dataset_dir else None,
                "training_summary": self._read_json_if_exists(model_dir / "training_summary.json") if model_dir else None,
                "export_summary": self._read_json_if_exists(export_dir / "export_summary.json") if export_dir else None,
            },
        }

    @staticmethod
    def _safe_project_name(project_name: str) -> str:
        """Return a filesystem-safe project folder name.

        Args:
            project_name: Raw user project name.

        Returns:
            Safe folder name.
        """

        return re.sub(r"[^A-Za-z0-9_.-]+", "_", project_name).strip("._") or "DrunkenBotProject"

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
