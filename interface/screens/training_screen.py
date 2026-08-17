from __future__ import annotations

# TrainingScreenMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class TrainingScreenMixin:
    def _refresh_training_layout(self) -> None:
        """Apply responsive card columns on the training page."""

        if not self.training_cards or self.training_controls_grid is None:
            return
        width = self.pages.width() if hasattr(self, "pages") else self.width()
        if width >= 900:
            columns = 2
        else:
            columns = 1
        if columns == self.training_controls_columns:
            return
        self._set_training_card_columns(columns)

    def _set_training_card_columns(self, columns: int) -> None:
        """Reflow the training cards into the requested column count.

        Args:
            columns: Number of columns to use.
        """

        if self.training_controls_grid is None:
            return
        while self.training_controls_grid.count():
            self.training_controls_grid.takeAt(0)
        for index, card in enumerate(self.training_cards):
            row = index // columns
            column = index % columns
            self.training_controls_grid.addWidget(card, row, column)
        for column in range(2):
            self.training_controls_grid.setColumnStretch(column, 1 if column < columns else 0)
        self.training_controls_columns = columns

    def _configure_device_options(self) -> None:
        """Populate training device choices without duplicate CPU entries."""

        self.device.clear()
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            self.device.addItem("cuda")
            self.device.addItem("cpu")
            self.device_info.setText(f"CUDA ready: {device_name}")
            self.use_amp_default = True
        else:
            self.device.addItem("cpu")
            cuda_build = getattr(torch.backends, "cuda", None)
            built_with_cuda = bool(cuda_build and torch.backends.cuda.is_built())
            if built_with_cuda:
                detail = "CUDA build found, but no usable NVIDIA GPU/driver was detected."
            else:
                detail = "CUDA is not available in this PyTorch install."
            self.device_info.setText(detail)
            self.use_amp_default = False

    def _training_launch_target_value(self) -> str:
        """Return whether training should launch locally or remotely.

        Returns:
            ``local`` or ``remote``.
        """

        if self.training_launch_target.currentText() == "RunPod cloud":
            return "runpod"
        return "remote" if self.training_launch_target.currentText() == "Remote workers" else "local"

    def _architecture_style_config(self) -> dict[str, Any]:
        """Return ModelConfig keyword arguments for the selected block style.

        Returns:
            Architecture style settings.
        """

        if self.architecture_style.currentText() == "Llama-like":
            return {
                "norm_type": "rmsnorm",
                "position_encoding": "rope",
                "mlp_type": "swiglu",
                "rope_theta": self.rope_theta.value(),
            }
        return {
            "norm_type": "layernorm",
            "position_encoding": "learned",
            "mlp_type": "gelu",
            "rope_theta": self.rope_theta.value(),
        }

    def _optimizer_value(self) -> str:
        """Return the selected optimizer identifier.

        Returns:
            Stable optimizer name used by the trainer.
        """

        return {
            "AdamW": "adamw",
            "Adam": "adam",
            "Lion": "lion",
            "Adafactor": "adafactor",
        }.get(self.optimizer_name.currentText(), "adamw")

    def _scheduler_value(self) -> str:
        """Return the selected scheduler identifier.

        Returns:
            Stable scheduler name used by the trainer.
        """

        return {
            "Warmup linear": "warmup_linear",
            "Cosine decay": "cosine",
            "Polynomial decay": "polynomial",
            "One-cycle": "one_cycle",
            "Constant": "constant",
        }.get(self.scheduler_name.currentText(), "warmup_linear")

    def _precision_value(self) -> str:
        """Return the selected numeric precision identifier.

        Returns:
            Stable precision name used by the trainer.
        """

        return {
            "FP16": "fp16",
            "BF16": "bf16",
            "FP32": "fp32",
        }.get(self.precision.currentText(), "fp16")

    def _training_output_dir_for_mode(self, training_mode: Optional[str]) -> Path:
        """Return the output folder for a training mode.

        Args:
            training_mode: Training mode override.

        Returns:
            Base model or fine-tune output folder.
        """

        return self._fine_tune_output_path() if training_mode == "fine_tune" else Path(self.model_dir.text())

    def _attention_type_value(self) -> str:
        """Return the selected attention layout identifier.

        Returns:
            Stable attention type used by the model.
        """

        return {
            "Multi-head": "mha",
            "Grouped-query": "gqa",
            "Multi-query": "mqa",
        }.get(self.attention_type.currentText(), "mha")

    def _attention_backend_value(self) -> str:
        """Return the selected attention backend identifier.

        Returns:
            Stable attention backend used by the model.
        """

        return {
            "SDPA / Flash when available": "sdpa",
            "Manual": "manual",
        }.get(self.attention_backend.currentText(), "sdpa")

    def _profile_architecture_scale(self) -> float:
        context = max(8, self.context_length.value())
        try:
            # embedding = max(1, self.embedding_size.value())
            embedding = max(1, self.n_embd.value())
        except Exception as err:
            LOGGER.error(err)
            embedding = 1/256
            pass
        layers = max(1, self.n_layer.value())
        return max(0.5, min(8.0, (context / 512) * (embedding / 256) * (layers / 6)))

    def _apply_profile_runtime_defaults(self, profile: str) -> None:
        scale = self._profile_architecture_scale()
        cpu_count = max(1, os.cpu_count() or 1)
        workers = min(8, max(1, cpu_count // 2))
        profile_defaults = {
            "Low-memory": (128, 100, 16, 1000, 2),
            "Code fine-tune": (256, 50, 32, 500, 2),
            "Experimental Lion": (256, 100, 50, 500, 2),
            "Stable LLM": (128, 100, 50, 500, 2),
        }
        stride, eval_interval, eval_batches, save_interval, worker_divisor = profile_defaults.get(profile, profile_defaults["Stable LLM"])
        if scale >= 2.0:
            stride *= 2
            eval_interval *= 2
            save_interval *= 2
            eval_batches = max(8, eval_batches // 2)
        elif scale <= 0.75:
            eval_interval = max(25, eval_interval // 2)
            save_interval = max(100, save_interval // 2)
        target_effective_batch = {
            "Low-memory": 16,
            "Code fine-tune": 16,
            "Experimental Lion": 32,
            "Stable LLM": 32,
        }.get(profile, 32)
        if scale >= 4.0:
            batch_size = 2
        elif scale >= 2.0:
            batch_size = 4
        elif scale >= 1.25:
            batch_size = 8
        elif scale <= 0.75:
            batch_size = 32
        else:
            batch_size = 16
        batch_size = min(self.batch_size.maximum(), batch_size)
        self.batch_size.setValue(batch_size)
        accumulation = max(1, (target_effective_batch + batch_size - 1) // batch_size)
        self.gradient_accumulation.setValue(min(self.gradient_accumulation.maximum(), accumulation))
        self.sample_stride.setValue(min(self.sample_stride.maximum(), max(1, stride)))
        self.eval_interval.setValue(min(self.eval_interval.maximum(), max(0, eval_interval)))
        self.max_eval_batches.setValue(min(self.max_eval_batches.maximum(), max(0, eval_batches)))
        self.save_interval.setValue(min(self.save_interval.maximum(), max(1, save_interval)))
        self.data_loader_workers.setValue(min(self.data_loader_workers.maximum(), workers // worker_divisor))
        patience = {"Low-memory": 4, "Code fine-tune": 2, "Experimental Lion": 3, "Stable LLM": 3}.get(profile, 3)
        if eval_interval >= 200:
            patience += 1
        if eval_batches <= 16:
            patience += 1
        self.early_stopping_patience.setValue(min(self.early_stopping_patience.maximum(), patience))
        epochs = {"Low-memory": 5, "Code fine-tune": 3, "Experimental Lion": 4, "Stable LLM": 5}.get(profile, 5)
        self.epochs.setValue(min(self.epochs.maximum(), epochs))

    def apply_training_profile(self) -> None:
        """Apply the selected optimizer/scheduler/regularization profile.

        Each branch below explicitly sets every field it conceptually owns
        (optimizer, scheduler, LR/regularization, precision/memory knobs,
        batch shape, and early-stopping patience), even fields that happen
        to match the previous profile's value. This is deliberate: profiles
        must be idempotent when switched between, or a field set by a
        previously applied profile (e.g. activation_checkpointing=True from
        Low-memory) can silently survive into a later profile that never
        mentions it, producing a configuration no single profile actually
        intended.

        Two categories of fields are deliberately NOT touched here:
          - attention_type / kv_head_count: an architecture choice, not a
            training-strategy choice. Low-memory sets these to
            Grouped-query because that specific profile is about reducing
            memory end-to-end; the other profiles leave whatever the user
            has selected alone rather than silently reverting it.
          - training_mode / peft_method / lora_* (Code fine-tune only):
            these belong to the fine-tuning tab's widgets, not this tab's.
        """

        profile = self.training_profile.currentText()
        if profile == "Low-memory":
            self._set_combo_text(self.optimizer_name, "Adafactor")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.0002)
            self.weight_decay.setValue(0.05)
            self.min_lr_ratio.setValue(0.05)
            self.polynomial_power.setValue(1.0)
            self.max_grad_norm.setValue(1.0)
            self._set_combo_text(self.precision, "BF16" if torch.cuda.is_available() else "FP32")
            self.use_amp.setChecked(True)
            self._set_combo_text(self.attention_type, "Grouped-query")
            self.kv_head_count.setValue(max(1, self.n_head.value() // 2))
            self.activation_checkpointing.setChecked(True)
            # The two knobs that most directly control peak memory: shrink
            # the batch and make up the lost effective batch size with
            # gradient accumulation, and avoid extra data-loader worker
            # processes competing for memory.
            self.batch_size.setValue(4)
            self.gradient_accumulation.setValue(4)
            self.data_loader_workers.setValue(0)
            self.warmup_steps.setValue(100)
            self.dropout.setValue(0.1)
            self.early_stopping_patience.setValue(3)
        elif profile == "Code fine-tune":
            self._set_combo_text(self.optimizer_name, "AdamW")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.00005)
            self.weight_decay.setValue(0.05)
            self.min_lr_ratio.setValue(0.1)
            self.polynomial_power.setValue(1.0)
            self.max_grad_norm.setValue(0.5)
            self._set_combo_text(self.precision, "FP16")
            self.use_amp.setChecked(True)
            self.activation_checkpointing.setChecked(False)
            self.batch_size.setValue(16)
            self.gradient_accumulation.setValue(1)
            self.data_loader_workers.setValue(0)
            self.warmup_steps.setValue(50)
            self.dropout.setValue(0.05)
            # Fine-tuning generally needs less patience than a full
            # pretraining run before validation loss plateaus meaningfully.
            self.early_stopping_patience.setValue(2)
            self._set_combo_text(self.training_mode, "Fine-tune checkpoint")
            self._set_combo_text(self.peft_method, "LoRA adapters")
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.lora_dropout.setValue(0.05)
            self._set_combo_text(self.lora_targets, "Attention projections")
        elif profile == "Experimental Lion":
            self._set_combo_text(self.optimizer_name, "Lion")
            self._set_combo_text(self.scheduler_name, "One-cycle")
            self.learning_rate.setValue(0.0001)
            self.weight_decay.setValue(0.1)
            self.min_lr_ratio.setValue(0.01)
            self.polynomial_power.setValue(1.0)
            self.max_grad_norm.setValue(1.0)
            # Lion is reported to be more sensitive to fp16 under/overflow
            # than AdamW; prefer bf16 where available, fp32 otherwise.
            self._set_combo_text(self.precision, "BF16" if torch.cuda.is_available() else "FP32")
            self.use_amp.setChecked(True)
            self.activation_checkpointing.setChecked(False)
            self.batch_size.setValue(16)
            self.gradient_accumulation.setValue(1)
            self.data_loader_workers.setValue(0)
            self.warmup_steps.setValue(100)
            self.dropout.setValue(0.1)
            self.early_stopping_patience.setValue(3)
        else:
            self._set_combo_text(self.optimizer_name, "AdamW")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.0003)
            self.weight_decay.setValue(0.1)
            self.min_lr_ratio.setValue(0.1)
            self.polynomial_power.setValue(1.0)
            self.max_grad_norm.setValue(1.0)
            self._set_combo_text(self.precision, "FP16")
            self.use_amp.setChecked(True)
            self.activation_checkpointing.setChecked(False)
            self.batch_size.setValue(16)
            self.gradient_accumulation.setValue(1)
            self.data_loader_workers.setValue(0)
            self.warmup_steps.setValue(100)
            self.dropout.setValue(0.1)
            self.early_stopping_patience.setValue(3)
        self._apply_profile_runtime_defaults(profile)
        self._update_training_mode_controls()
        self.refresh_model_estimate()
        self.training_log.append(f"Applied training profile: {profile}")

    def _apply_preset(self, preset: str) -> None:
        """Apply architecture values for a preset.

        Args:
            preset: Selected preset name.
        """

        if preset == "Tiny":
            self.n_embd.setValue(128)
            self.n_head.setValue(4)
            self.n_layer.setValue(4)
        elif preset == "Small":
            self.n_embd.setValue(512)
            self.n_head.setValue(8)
            self.n_layer.setValue(8)
