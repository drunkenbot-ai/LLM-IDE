from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart17:
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

    @Slot(object)
    def _training_finished(self, result: Any) -> None:
        """Update UI after training finishes.

        Args:
            result: Training result.
        """

        log = self.active_training_log or self.training_log
        progress = self.active_training_progress or self.training_progress
        progress.setValue(100)
        if progress is not self.training_progress:
            self.training_progress.setValue(100)
        if hasattr(self, "live_progress"):
            self.live_progress.setValue(100)
        log.append(f"Saved model: {result.checkpoint_path}")
        log.append(f"Final train loss: {result.final_train_loss:.4f}")
        if result.final_val_loss is not None:
            log.append(f"Final validation loss: {result.final_val_loss:.4f}")
        training_summary: dict[str, Any] = {}
        try:
            training_summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        except Exception:
            training_summary = {}
        best_checkpoint = str(training_summary.get("recommended_checkpoint_path") or "")
        best_val_loss = training_summary.get("best_val_loss")
        if best_checkpoint:
            if best_val_loss is not None:
                log.append(f"Recommended checkpoint: {best_checkpoint} (best validation loss {float(best_val_loss):.4f})")
            else:
                log.append(f"Recommended checkpoint: {best_checkpoint}")
        output_dir = self.active_training_output_dir or Path(result.checkpoint_path).parent
        stage_key = self.active_task_kind if self.active_task_kind in {"training", "fine_tune"} else "training"
        self.export_model_dir.setText(str(output_dir))
        try:
            if stage_key != "fine_tune" and Path(output_dir).resolve() == Path(self.model_dir.text()).resolve():
                self.fine_tune_checkpoint.setText(str(result.checkpoint_path))
        except OSError:
            pass
        if getattr(result, "stopped", False):
            self.project_state.setText("Training stopped")
            self.train_status.setText("Training: stopped, checkpoint saved")
            log.append("Training stopped safely. Resume from this checkpoint or the latest checkpoint.")
        else:
            self.project_state.setText("Training complete")
            self.train_status.setText(f"Training: loss {result.final_train_loss:.4f}")
        title = "Fine-tuning complete" if stage_key == "fine_tune" else "Model training complete"
        if getattr(result, "stopped", False):
            title = "Fine-tuning stopped" if stage_key == "fine_tune" else "Model training stopped"
        completion_lines = [
            f"Checkpoint: {result.checkpoint_path}",
            f"Summary: {result.summary_path}",
            f"Final train loss: {result.final_train_loss:.4f}",
        ]
        if result.final_val_loss is not None:
            completion_lines.append(f"Final validation loss: {result.final_val_loss:.4f}")
        if best_checkpoint:
            completion_lines.append(f"Recommended checkpoint: {best_checkpoint}")
        if best_val_loss is not None:
            completion_lines.append(f"Best validation loss: {float(best_val_loss):.4f}")
        completion_lines.append(f"Output: {output_dir}")
        self._notify_complete(stage_key, title, completion_lines)
        self._append_training_history(result)
        self._clear_button_busy(self.active_training_final_button_text)
        self.active_training_log = None
        self.active_training_progress = None
        self.active_training_output_dir = None

    def run_benchmark(self) -> None:
        """Run benchmark prompts against the current trained model."""

        prompts = normalize_prompts(self.benchmark_prompts.toPlainText())
        self.benchmark_log.append(f"Running benchmark with {len(prompts)} prompt(s)...")
        self.benchmark_progress.setValue(0)
        self.project_state.setText("Benchmarking")
        self._run_task(
            evaluate_checkpoint,
            (
                Path(self.model_dir.text()),
                prompts,
                None,
                self.benchmark_tokens.value(),
                self.benchmark_temperature.value(),
                50,
                self.device.currentText(),
                self.benchmark_kv_cache.isChecked(),
            ),
            self._benchmark_finished,
            self.benchmark_log,
            self.benchmark_progress,
            with_progress=True,
            button=self.run_benchmark_button,
            stop_button=self.stop_benchmark_button,
            busy_text="Benchmarking",
        )

    @Slot(object)
    def _benchmark_finished(self, result: Any) -> None:
        """Update UI after benchmark prompts finish.

        Args:
            result: Benchmark result object.
        """

        self.benchmark_progress.setRange(0, 100)
        self.benchmark_progress.setValue(100)
        self.benchmark_log.append(
            f"Benchmark complete: {result.prompt_count} prompt(s), {result.total_seconds:.2f}s, "
            f"{result.total_generated_tokens} generated token(s), {result.tokens_per_second:.2f} tok/s."
        )
        self.benchmark_log.append(f"Benchmark saved: {result.output_path}")
        self.project_state.setText("Benchmark complete")
        self._clear_button_busy("Run Benchmark")

    def toggle_llm_model(self) -> None:
        """Load or unload the selected chat model depending on current state."""

        if self.chat_session is not None:
            self.unload_llm_model()
            return
        self.load_llm_model()

    def load_llm_model(self) -> None:
        """Load a selected model backend for chat testing."""

        backend = self._chat_backend_value()
        path_text = self.microgpt_chat_path.text().strip() if backend == "microgpt" else self.gguf_path.text().strip()
        if not path_text:
            required = "MicroGPT model folder or checkpoint" if backend == "microgpt" else "GGUF model file"
            QMessageBox.information(self, "Model required", f"Choose a {required} first.")
            return
        model_path = Path(path_text)
        self.chat_progress.setValue(0)
        self._render_chat_markdown("**Loading model...**")
        self.chat_stats.setText("Loading model...")
        self.project_state.setText("Loading chat model")
        self.chat_status.setText("Chat: loading model")
        loader = load_microgpt_chat_session if backend == "microgpt" else load_llama_chat_session
        args = (
            (model_path, self.device.currentText())
            if backend == "microgpt"
            else (model_path, self.llama_context.value(), self.llama_threads.value(), self.llama_gpu_layers.value())
        )
        self._run_task(
            loader,
            args,
            self._llm_loaded,
            self.chat_event_log,
            self.chat_progress,
            button=self.load_llm_button,
            busy_text="Loading Model",
            task_kind="chat",
        )

    @Slot(object)
    def _llm_loaded(self, session: Any) -> None:
        """Store a loaded GGUF chat session.

        Args:
            session: Loaded ``LlamaChatSession``.
        """

        self.chat_session = session
        self._clear_chat_messages()
        self.chat_markdown = ""
        self._add_chat_message(
            "assistant",
            f"Loaded model: `{session.model_path.name}`\n\n{session.runtime_summary}\n\nSend a message to begin.",
        )
        self.chat_progress.setValue(100)
        self.chat_stats.setText(session.runtime_summary)
        self.project_state.setText("Chat model loaded")
        self.chat_status.setText(f"Chat: {session.runtime_summary}")
        self._clear_button_busy("Unload")
        self._tip(self.load_llm_button, "Unload the currently loaded model from memory.")

    def unload_llm_model(self) -> None:
        """Unload the active chat model and clear chat state."""

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please wait for the current task to finish.")
            return
        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        self._clear_chat_messages()
        self.chat_markdown = ""
        self._add_chat_message("assistant", "Model unloaded.\n\nLoad a model to start testing.")
        self.chat_progress.setRange(0, 100)
        self.chat_progress.setValue(0)
        self.chat_stats.setText("Idle")
        self.project_state.setText("Ready")
        self.chat_status.setText("Chat: no model loaded")
        self.load_llm_button.setText("Load Model")
        self._update_chat_backend_controls()

    def send_chat_message(self) -> None:
        """Send a prompt to the loaded chat model."""

        if self.chat_session is None:
            QMessageBox.information(self, "Load model", "Load a model before sending a message.")
            return
        prompt = self.chat_input.toPlainText().strip()
        if not prompt:
            return
        self.pending_user_message = prompt
        self.chat_input.clear()
        self._add_chat_message("user", prompt, resend_prompt=prompt)
        self.chat_stream_reply = ""
        self._add_chat_message("assistant", "_Thinking..._", resend_prompt=prompt)
        self.chat_progress.setRange(0, 0)
        self.chat_stats.setText("Thinking...")
        self.project_state.setText("Generating")
        self.chat_status.setText("Chat: generating reply")
        streamer = stream_microgpt_chat_reply if self._chat_backend_value() == "microgpt" else stream_chat_reply
        self._run_task(
            streamer,
            (
                self.chat_session,
                prompt,
                self.system_prompt.toPlainText(),
                self.chat_max_tokens.value(),
                self.chat_temperature.value(),
                self.chat_top_p.value(),
                self.chat_repeat_penalty.value(),
                self.reasoning_effort.currentText(),
                self.thinking_enabled.isChecked(),
            ),
            self._chat_reply_finished,
            self.chat_event_log,
            self.chat_progress,
            with_progress=True,
            button=self.send_chat_button,
            stop_button=self.stop_chat_button,
            busy_text="Thinking",
        )

    @Slot(object)
    def _chat_reply_finished(self, reply: Any) -> None:
        """Render the model reply.

        Args:
            reply: Assistant reply text and metrics.
        """

        result = reply if isinstance(reply, dict) else {"reply": str(reply)}
        text = str(result.get("reply", "")).strip()
        if text:
            self.chat_stream_reply = text
        else:
            self.chat_stream_reply = self.chat_stream_reply or "_No reply returned._"
        self._render_chat_markdown(self.chat_stream_reply)
        self.chat_progress.setRange(0, 100)
        self.chat_progress.setValue(100)
        self._set_chat_stats(
            float(result.get("elapsed_seconds", 0.0)),
            int(result.get("token_count", 0)),
            float(result.get("tokens_per_second", 0.0)),
        )
        self.project_state.setText("Ready")
        self.chat_status.setText("Chat: ready")
        self._clear_button_busy("Send")

    def reset_chat(self) -> None:
        """Clear the chat transcript and model conversation memory."""

        if self.chat_session is not None:
            self.chat_session.reset()
        self._clear_chat_messages()
        self.chat_markdown = ""
        self.chat_stream_prefix = ""
        self.chat_stream_reply = ""
        self._add_chat_message("assistant", "Chat reset.")
        self.chat_stats.setText("Idle")
        self.chat_status.setText("Chat: ready")

    def _append_chat_markdown(self, role: str, content: str) -> None:
        """Append one rendered chat message.

        Args:
            role: Display role heading.
            content: Markdown content.
        """

        block = f"### {role}\n{content.strip()}\n"
        self.chat_markdown = f"{self.chat_markdown.rstrip()}\n\n{block}" if self.chat_markdown else block
        self._add_chat_message("user" if role.lower() in {"you", "user"} else "assistant", content)

    def create_bundle(self) -> None:
        if not bool(QApplication.instance().property("license_valid")):
            self.export_log.append("Export Bay is available only in the licensed version.")
            return
        """Create a portable model export bundle."""

        self.export_log.append("Creating model bundle...")
        self.export_progress.setValue(15)
        try:
            output = export_project_bundle(Path(self.export_model_dir.text()), Path(self.export_dir.text()))
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"Bundle created: {output}")
        self.export_status.setText("Export: bundle created")

    def quantize_model(self) -> None:
        if not bool(QApplication.instance().property("license_valid")):
            return
        """Create a quantized FP16 checkpoint when selected."""

        mode = self.quant_mode.currentText()
        if not mode.startswith("FP16"):
            self.export_log.append("This GGUF quantization target is planned. FP16 checkpoint quantization is available now.")
            return
        checkpoint = Path(self.export_model_dir.text()) / "final_model.pt"
        output = Path(self.export_dir.text()) / "final_model_fp16.pt"
        self.export_log.append("Creating FP16 checkpoint...")
        self.export_progress.setValue(20)
        try:
            result = quantize_checkpoint(checkpoint, output, mode="fp16")
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"Quantized checkpoint created: {result}")
        self.export_status.setText("Export: FP16 checkpoint ready")

    def export_hf_package(self) -> None:
        if not bool(QApplication.instance().property("license_valid")):
            return
        """Create an HF-style MicroGPT package."""

        self.export_log.append("Creating HF-style MicroGPT package...")
        self.export_progress.setValue(20)
        try:
            result = export_hf_microgpt_package(Path(self.export_model_dir.text()))
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"HF package created: {result}")
        self.export_log.append("Note: this package is MicroGPT model_type, not a llama.cpp-supported Llama model.")
        self.export_status.setText("Export: HF package ready")

