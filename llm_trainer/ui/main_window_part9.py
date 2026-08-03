from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart9:
    def stop_active_task(self) -> None:
        """Request a graceful stop for the active background task."""

        if self.stop_event is None:
            return
        LOGGER.info("Stop requested for active background task")
        self.stop_event.set()
        self._notify_failure("Stop requested", "The task is stopping at the next safe point.")
        if self.active_log is not None:
            self.active_log.append("Stop requested. Finishing the current safe point...")
        if self.active_stop_button is not None:
            self.active_stop_button.setEnabled(False)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                LOGGER.exception(
                    "Failed to empty CUDA cache in _thread_finished")

    @Slot()
    def request_shutdown_from_signal(self) -> None:
        """Handle Ctrl+C from a terminal without leaving Qt threads wedged."""

        self.interrupt_count += 1
        if self.interrupt_count > 1:
            os._exit(130)
        if self.stop_event is not None:
            self.stop_event.set()
        if self.active_log is not None:
            self.active_log.append("Interrupt received. Requesting stop...")
        self.project_state.setText("Stopping")
        if self.thread is None:
            QApplication.quit()
            return
        QTimer.singleShot(3000, lambda: os._exit(130) if self.thread is not None else QApplication.quit())

    def closeEvent(self, event: Any) -> None:
        """Clean up background services before the window closes.

        Args:
            event: Qt close event.
        """

        if self.thread is not None:
            if self.stop_event is not None:
                self.stop_event.set()
            if self.active_log is not None:
                self.active_log.append("Close requested. Stopping active task first...")
            self.project_state.setText("Stopping")
            LOGGER.info("Close requested while background task is running; waiting for task shutdown")
            event.ignore()
            QTimer.singleShot(500, self.close)
            return
        if self.coordinator_server is not None:
            self.stop_coordinator_server()
        super().closeEvent(event)

    def _handle_progress(self, event: object, log: QTextEdit, progress_bar: QProgressBar) -> None:
        """Apply one progress event to UI widgets.

        Args:
            event: Progress dictionary or message.
            log: Log widget to append messages to.
            progress_bar: Progress bar to update.
        """

        if isinstance(event, dict):
            if event.get("type") == "chat_delta":
                self._apply_chat_delta(event)
                return
            message = event.get("message")
            percent = event.get("percent")
            if log in (self.training_log, getattr(self, "fine_tune_log", None)):
                self._update_training_metrics(event, update_fine_tune=log is getattr(self, "fine_tune_log", None))
            if message:
                log.append(str(message))
                if log in (self.training_log, getattr(self, "fine_tune_log", None)) and hasattr(self, "live_log"):
                    self.live_log.append(str(message))
            if percent is not None:
                progress_bar.setValue(max(0, min(100, int(percent))))
                if log in (self.training_log, getattr(self, "fine_tune_log", None)) and hasattr(self, "live_progress"):
                    self.live_progress.setValue(max(0, min(100, int(percent))))
        else:
            log.append(str(event))

    def _notify_progress(self, event: dict[str, Any]) -> None:
        """Send throttled external progress notifications for long tasks.

        Args:
            event: Progress event emitted by a worker.
        """

        if not self.active_task_kind or self.notification_manager is None:
            return
        if self.active_task_kind not in {"dataset", "training", "fine_tune"}:
            return
        title = {
            "dataset": "Dataset preparation",
            "training": "Model training",
            "fine_tune": "Fine-tuning",
        }[self.active_task_kind]
        percent = event.get("percent")
        self.notification_manager.notify_progress(
            self.active_task_kind,
            title,
            self._notification_lines_from_event(event),
            int(percent) if percent is not None else None,
        )

    def _notify_complete(self, stage_key: str, title: str, lines: list[str]) -> None:
        """Send an external completion notification when configured.

        Args:
            stage_key: Notification stage key.
            title: User-facing title.
            lines: Plain-text summary lines.
        """

        if self.notification_manager is not None:
            self.notification_manager.notify_complete(stage_key, title, lines)

    def _notify_failure(self, title: str, message: str) -> None:
        """Send an external failure or stop notification for the active task.

        Args:
            title: User-facing title.
            message: Failure details.
        """

        if self.active_task_kind and self.notification_manager is not None:
            self.notification_manager.notify_failure(self.active_task_kind, title, message)

    def _notification_lines_from_event(self, event: dict[str, Any]) -> list[str]:
        """Build compact notification text from a worker progress event.

        Args:
            event: Progress event emitted by a worker.

        Returns:
            Body lines for the notification message.
        """

        lines: list[str] = []
        if event.get("message"):
            lines.append(str(event["message"]))
        if "epoch" in event and "total_epochs" in event:
            lines.append(f"Epoch: {event['epoch']}/{event['total_epochs']}")
        if "step" in event and "total_steps" in event:
            lines.append(f"Step: {event['step']}/{event['total_steps']}")
        train_loss = self._finite_metric(event.get("train_loss"))
        if train_loss is not None:
            lines.append(f"Train loss: {float(train_loss):.4f}")
        val_loss = self._finite_metric(event.get("val_loss"))
        if val_loss is not None:
            lines.append(f"Validation loss: {float(val_loss):.4f}")
        learning_rate = self._finite_metric(event.get("learning_rate"))
        if learning_rate is not None:
            lines.append(f"Learning rate: {float(learning_rate):.2e}")
        tokens_per_second = self._finite_metric(event.get("tokens_per_second"))
        if tokens_per_second is not None:
            lines.append(f"Speed: {float(tokens_per_second):.0f} tokens/sec")
        eta_seconds = self._finite_metric(event.get("eta_seconds"))
        if eta_seconds is not None:
            lines.append(f"ETA: {self._format_duration(float(eta_seconds))}")
        vram_allocated = self._finite_metric(event.get("vram_allocated_gb"))
        vram_reserved = self._finite_metric(event.get("vram_reserved_gb"))
        if vram_allocated is not None or vram_reserved is not None:
            allocated = "-" if vram_allocated is None else f"{float(vram_allocated):.2f} GB"
            reserved = "-" if vram_reserved is None else f"{float(vram_reserved):.2f} GB"
            lines.append(f"VRAM: {allocated} allocated, {reserved} reserved")
        return lines[:10]

    def _update_training_metrics(self, event: dict[str, Any], update_fine_tune: bool = False) -> None:
        """Update training metric chips from a progress event.

        Args:
            event: Progress event emitted by the training backend.
            update_fine_tune: Whether to mirror metrics into the Fine-Tuning tab chips.
        """

        if "epoch" in event and "total_epochs" in event:
            self.training_epoch_metric.setText(f"Epoch: {event['epoch']}/{event['total_epochs']}")
            if update_fine_tune and hasattr(self, "fine_tune_epoch_metric"):
                self.fine_tune_epoch_metric.setText(f"Epoch: {event['epoch']}/{event['total_epochs']}")
        if "step" in event and "total_steps" in event:
            self.training_step_metric.setText(f"Step: {event['step']}/{event['total_steps']}")
            if update_fine_tune and hasattr(self, "fine_tune_step_metric"):
                self.fine_tune_step_metric.setText(f"Step: {event['step']}/{event['total_steps']}")
        train_loss = self._finite_metric(event.get("train_loss"))
        if train_loss is not None:
            self.training_loss_metric.setText(f"Train loss: {float(train_loss):.4f}")
            if update_fine_tune and hasattr(self, "fine_tune_loss_metric"):
                self.fine_tune_loss_metric.setText(f"Train loss: {float(train_loss):.4f}")
        val_loss = self._finite_metric(event.get("val_loss"))
        if val_loss is not None:
            self.training_val_metric.setText(f"Val loss: {float(val_loss):.4f}")
            if update_fine_tune and hasattr(self, "fine_tune_val_metric"):
                self.fine_tune_val_metric.setText(f"Val loss: {float(val_loss):.4f}")
        step = event.get("step")
        if step is not None and (train_loss is not None or val_loss is not None):
            step_int_for_loss = int(step)
            self.loss_chart.add_metrics(step_int_for_loss, train_loss, val_loss)
            self._update_training_health(step_int_for_loss, train_loss, val_loss)
        if step is None:
            return
        step_int = int(step)
        self._record_live_metric(event)
        learning_rate = self._finite_metric(event.get("learning_rate"))
        grad_norm = self._finite_metric(event.get("grad_norm"))
        weight_norm = self._finite_metric(event.get("weight_norm"))
        update_ratio = self._finite_metric(event.get("update_ratio"))
        tokens_per_second = self._finite_metric(event.get("tokens_per_second"))
        samples_per_second = self._finite_metric(event.get("samples_per_second"))
        vram_allocated = self._finite_metric(event.get("vram_allocated_gb"))
        vram_reserved = self._finite_metric(event.get("vram_reserved_gb"))
        gpu_memory = self._finite_metric(event.get("gpu_memory_percent"))
        system_cpu = self._finite_metric(event.get("system_cpu_percent"))
        system_ram = self._finite_metric(event.get("system_ram_percent"))
        data_workers = event.get("data_loader_workers")
        eta_seconds = self._finite_metric(event.get("eta_seconds"))
        if learning_rate is not None:
            self.training_lr_metric.setText(f"LR: {float(learning_rate):.2e}")
            if update_fine_tune and hasattr(self, "fine_tune_lr_metric"):
                self.fine_tune_lr_metric.setText(f"LR: {float(learning_rate):.2e}")
        if grad_norm is not None:
            self.training_grad_metric.setText(f"Grad: {float(grad_norm):.3f}")
            if update_fine_tune and hasattr(self, "fine_tune_grad_metric"):
                self.fine_tune_grad_metric.setText(f"Grad: {float(grad_norm):.3f}")
        if tokens_per_second is not None:
            self.training_speed_metric.setText(f"Speed: {float(tokens_per_second):.0f} tok/s")
            if update_fine_tune and hasattr(self, "fine_tune_speed_metric"):
                self.fine_tune_speed_metric.setText(f"Speed: {float(tokens_per_second):.0f} tok/s")
        if vram_allocated is not None:
            self.training_vram_metric.setText(f"VRAM: {float(vram_allocated):.2f} GB")
        if eta_seconds is not None:
            self.training_eta_metric.setText(f"ETA: {self._format_duration(float(eta_seconds))}")
            if update_fine_tune and hasattr(self, "fine_tune_eta_metric"):
                self.fine_tune_eta_metric.setText(f"ETA: {self._format_duration(float(eta_seconds))}")
        if learning_rate is not None or grad_norm is not None:
            self.optimization_chart.add_values(step_int, learning_rate, grad_norm)
        if weight_norm is not None or update_ratio is not None:
            self.stability_chart.add_values(step_int, weight_norm, update_ratio)
        if tokens_per_second is not None or samples_per_second is not None:
            self.throughput_chart.add_values(step_int, tokens_per_second, samples_per_second)
        if vram_allocated is not None or vram_reserved is not None:
            self.memory_chart.add_values(step_int, vram_allocated, vram_reserved)
        if hasattr(self, "live_epoch_metric"):
            self._update_live_training_metrics(
                step_int,
                event,
                train_loss,
                learning_rate,
                grad_norm,
                update_ratio,
                tokens_per_second,
                samples_per_second,
                vram_allocated,
                vram_reserved,
                gpu_memory,
                system_cpu,
                system_ram,
                data_workers,
            )

    def _update_training_health(
        self,
        step: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
    ) -> None:
        """Update the training health advisor from recent loss values.

        Args:
            step: Current optimizer step.
            train_loss: Latest training loss.
            val_loss: Latest validation loss.
        """

        self.training_health_points.append((step, train_loss, val_loss))
        self.training_health_points = self.training_health_points[-12:]
        latest_train = next((item[1] for item in reversed(self.training_health_points) if item[1] is not None), None)
        latest_val = next((item[2] for item in reversed(self.training_health_points) if item[2] is not None), None)
        val_points = [(item[0], item[2]) for item in self.training_health_points if item[2] is not None]
        if latest_train is None and latest_val is None:
            label = "Health: collecting"
            tip = "Waiting for train and validation loss."
        elif latest_train is not None and latest_val is not None and latest_train < 0.2 and latest_val > max(2.0, latest_train * 8.0):
            label = "Health: validation gap"
            tip = "Training loss is very low while validation loss is high. Check overfitting, validation split, tokenizer match, or eval settings."
        elif len(val_points) >= 3 and val_points[-1][1] > val_points[-2][1] > val_points[-3][1]:
            label = "Health: overfitting?"
            tip = "Validation loss has increased for three checks. Consider stopping, reducing epochs, or improving validation data."
        elif latest_train is not None and (latest_train > 20.0 or not math.isfinite(latest_train)):
            label = "Health: diverging"
            tip = "Training loss is unstable or extremely high. Lower learning rate and check gradients/data."
        elif latest_val is not None and latest_val > 10.0:
            label = "Health: high val loss"
            tip = "Validation loss is high. This may be early training, a difficult validation split, or a dataset/tokenizer mismatch."
        elif latest_train is not None and latest_val is not None and latest_val <= latest_train * 1.8:
            label = "Health: stable"
            tip = "Training and validation loss are reasonably close."
        else:
            label = "Health: watching"
            tip = "Collecting more loss points before making a stronger diagnosis."
        self.training_health_metric.setText(label)
        self._tip(self.training_health_metric, tip)

    @staticmethod
    def _finite_metric(value: Any) -> Optional[float]:
        """Return a finite metric value or ``None``.

        Args:
            value: Raw metric value.

        Returns:
            Finite float, or ``None`` when invalid.
        """

        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def _update_live_training_metrics(
        self,
        step: int,
        event: dict[str, Any],
        train_loss: Optional[float],
        learning_rate: Optional[float],
        grad_norm: Optional[float],
        update_ratio: Optional[float],
        tokens_per_second: Optional[float],
        samples_per_second: Optional[float],
        vram_allocated: Optional[float],
        vram_reserved: Optional[float],
        gpu_memory: Optional[float],
        system_cpu: Optional[float],
        system_ram: Optional[float],
        data_workers: Optional[int],
    ) -> None:
        """Update live tracker widgets from one training progress event.

        Args:
            step: Current optimizer step.
            event: Progress event emitted by training.
            train_loss: Latest training loss.
            learning_rate: Current learning rate.
            grad_norm: Current gradient norm.
            update_ratio: Current parameter update ratio.
            tokens_per_second: Current token throughput.
            samples_per_second: Current sample throughput.
            vram_allocated: Current CUDA allocated memory in GB.
            vram_reserved: Current CUDA reserved memory in GB.
            gpu_memory: Current GPU memory pressure percentage.
            system_cpu: Current system CPU utilization percentage.
            system_ram: Current system RAM utilization percentage.
            data_workers: CPU data-loader worker count.
        """

        total_steps = event.get("total_steps")
        if "epoch" in event and "total_epochs" in event:
            self.live_epoch_metric.setText(f"Epoch: {event['epoch']}/{event['total_epochs']}")
        if total_steps:
            self.live_step_metric.setText(f"Step: {step:,}/{int(total_steps):,}")
            data_percent = min(100.0, max(0.0, (step / max(1, int(total_steps))) * 100.0))
            self.live_data_metric.setText(f"Data: {data_percent:.1f}%")
            self.live_progress.setValue(int(data_percent))
        else:
            self.live_step_metric.setText(f"Step: {step:,}")
        if tokens_per_second is not None:
            self.live_tokens_metric.setText(f"Tokens/sec: {float(tokens_per_second):,.0f}")
        if train_loss is not None:
            self.live_loss_metric.setText(f"Loss: {float(train_loss):.4f}")
        if learning_rate is not None:
            self.live_lr_metric.setText(f"LR: {float(learning_rate):.2e}")
        sample_text = str(event.get("sample_text") or "").strip()
        if sample_text:
            self.live_sample_text.setText(f"Training text: {self._compact_preview_text(sample_text, 220)}")
        self.live_layer_status.setText(f"Layers: {self.n_layer.value()}")
        self.live_head_status.setText(f"Heads: {self.n_head.value()}")
        self.live_hidden_status.setText(f"Hidden size: {self.n_embd.value()}")
        self.live_batch_status.setText(f"Batch size: {self.batch_size.value()}")
        self.live_context_status.setText(f"Context: {self.train_context_length.value()}")
        self.live_device_status.setText(f"Device: {self.device.currentText()}")
        self.live_worker_status.setText(f"CPU workers: {data_workers if data_workers is not None else self.data_loader_workers.value()}")
        self._set_meter(self.live_cpu_bar, "CPU", system_cpu if system_cpu is not None else self._system_cpu_value())
        self._set_meter(self.live_gpu_bar, "GPU memory", gpu_memory)
        if vram_allocated is not None or vram_reserved is not None:
            allocated = float(vram_allocated or 0.0)
            reserved = float(vram_reserved or 0.0)
            reserved_percent = None
            if self.device.currentText().startswith("cuda") and torch.cuda.is_available():
                try:
                    _, total_vram = torch.cuda.mem_get_info()
                    reserved_percent = min(100.0, 100.0 * reserved * (1024 ** 3) / max(total_vram, 1))
                except Exception:
                    reserved_percent = None
            self._set_meter(self.live_vram_bar, "VRAM reserved", reserved_percent)
            self.live_vram_label.setText(f"VRAM reserved: {reserved:.2f} GB ({allocated:.2f} GB active)")
        self._set_meter(self.live_ram_bar, "System RAM", system_ram if system_ram is not None else self._system_ram_value())
        latest_loss = float(train_loss) if train_loss is not None else None
        self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), step, latest_loss)
        self.live_prediction_chart.update_distribution(step, latest_loss)
        self.live_attention_chart.update_heatmap(step, grad_norm)
        self.live_activation_chart.update_histogram(step, tokens_per_second)
        self.live_gradient_chart.update_flow(self.n_layer.value(), grad_norm, step)

    def _system_ram_value(self) -> Optional[float]:
        """Read system RAM utilization for live telemetry.

        Returns:
            System RAM percentage, or None when unavailable.
        """

        if psutil is None:
            return None
        return float(psutil.virtual_memory().percent)
