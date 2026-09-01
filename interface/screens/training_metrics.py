from __future__ import annotations

# TrainingMetricsMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class TrainingMetricsMixin:
    def _update_training_metrics(
        self,
        event: dict[str, Any],
        update_fine_tune: bool = False,
        render_live: bool = True,
    ) -> None:
        """Update training metric chips from a progress event.

        Args:
            event: Progress event emitted by the training backend.
            update_fine_tune: Whether to mirror metrics into the Fine-Tuning tab chips.
        """
        if "epoch" in event and "total_epochs" in event:
            self._set_text_if_changed(
                self.training_epoch_metric,
                f"Epoch: {event['epoch']}/{event['total_epochs']}",
            )
            if update_fine_tune and hasattr(self, "fine_tune_epoch_metric"):
                self._set_text_if_changed(
                    self.fine_tune_epoch_metric,
                    f"Epoch: {event['epoch']}/{event['total_epochs']}",
                )
        if "step" in event and "total_steps" in event:
            self._set_text_if_changed(
                self.training_step_metric,
                f"Step: {event['step']}/{event['total_steps']}",
            )
            if update_fine_tune and hasattr(self, "fine_tune_step_metric"):
                self._set_text_if_changed(
                    self.fine_tune_step_metric,
                    f"Step: {event['step']}/{event['total_steps']}",
                )
        train_loss = self._finite_metric(event.get("train_loss"))
        if train_loss is not None:
            self._set_text_if_changed(
                self.training_loss_metric,
                f"Train loss: {float(train_loss):.4f}",
            )
            if update_fine_tune and hasattr(self, "fine_tune_loss_metric"):
                self._set_text_if_changed(
                    self.fine_tune_loss_metric,
                    f"Train loss: {float(train_loss):.4f}",
                )
        val_loss = self._finite_metric(event.get("val_loss"))
        if val_loss is not None:
            self._set_text_if_changed(
                self.training_val_metric,
                f"Val loss: {float(val_loss):.4f}",
            )
            if update_fine_tune and hasattr(self, "fine_tune_val_metric"):
                self._set_text_if_changed(
                    self.fine_tune_val_metric,
                    f"Val loss: {float(val_loss):.4f}",
                )
        step = event.get("step")
        live_visible = render_live and self.pages.currentIndex() == self.live_page_index
        if step is not None and (train_loss is not None or val_loss is not None):
            step_int_for_loss = int(step)
            self._update_training_health(step_int_for_loss, train_loss, val_loss)
            if live_visible:
                self.loss_chart.add_metrics(step_int_for_loss, train_loss, val_loss)
        if step is None:
            return
        step_int = int(step)
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
            self._set_text_if_changed(
                self.training_lr_metric,
                f"LR: {float(learning_rate):.2e}",
            )
            if update_fine_tune and hasattr(self, "fine_tune_lr_metric"):
                self._set_text_if_changed(
                    self.fine_tune_lr_metric,
                    f"LR: {float(learning_rate):.2e}",
                )
        if grad_norm is not None:
            self._set_text_if_changed(
                self.training_grad_metric,
                f"Grad: {float(grad_norm):.3f}",
            )
            if update_fine_tune and hasattr(self, "fine_tune_grad_metric"):
                self._set_text_if_changed(
                    self.fine_tune_grad_metric,
                    f"Grad: {float(grad_norm):.3f}",
                )
        if tokens_per_second is not None:
            self._set_text_if_changed(
                self.training_speed_metric,
                f"Speed: {float(tokens_per_second):.0f} tok/s",
            )
            if update_fine_tune and hasattr(self, "fine_tune_speed_metric"):
                self._set_text_if_changed(
                    self.fine_tune_speed_metric,
                    f"Speed: {float(tokens_per_second):.0f} tok/s",
                )
        if vram_allocated is not None:
            self._set_text_if_changed(
                self.training_vram_metric,
                f"VRAM: {float(vram_allocated):.2f} GB",
            )
        if eta_seconds is not None:
            self._set_text_if_changed(
                self.training_eta_metric,
                f"ETA: {self._format_duration(float(eta_seconds))}",
            )
            if update_fine_tune and hasattr(self, "fine_tune_eta_metric"):
                self._set_text_if_changed(
                    self.fine_tune_eta_metric,
                    f"ETA: {self._format_duration(float(eta_seconds))}"
                )
        elapsed_seconds = self._finite_metric(event.get("elapsed_seconds"))
        if elapsed_seconds is not None:
            self._set_text_if_changed(
                self.training_elapsed_metric,
                f"Total time: {self._format_duration(float(elapsed_seconds))}",
            )
        if not live_visible:
            return
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
        self._set_text_if_changed(self.training_health_metric, label)
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
