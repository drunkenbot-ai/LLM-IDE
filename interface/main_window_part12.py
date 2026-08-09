from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart12:
    def _set_selected_default_data_paths(self, paths: Optional[list[Any]]) -> None:
        """Restore bundled default data checkbox selections.

        Args:
            paths: Saved bundled data file paths. ``None`` means no
                preference was ever saved (a brand-new project), and every
                file is selected by default. An explicit empty list means
                the user deliberately deselected everything, and that
                choice is restored as-is rather than falling back to
                "select everything" -- previously the two cases were
                indistinguishable, so saving a project with nothing
                selected silently reset to everything selected on reload.
        """

        if not hasattr(self, "default_data_actions"):
            return
        if paths is None:
            selected = set(self.default_data_actions)
        else:
            selected = {str(Path(path)) for path in paths}
        self.default_data_tree_updating = True
        try:
            for path, item in self.default_data_actions.items():
                item.setCheckState(0, Qt.Checked if path in selected else Qt.Unchecked)
            self._refresh_default_data_category_states()
        finally:
            self.default_data_tree_updating = False

    def _set_dataset_blueprint_refresh_busy(self, busy: bool) -> None:
        """Toggle refresh busy state indicators for the Dataset Sources page."""

        if hasattr(self, "dataset_plan_refresh_button"):
            self.dataset_plan_refresh_button.setEnabled(not busy)
            self.dataset_plan_refresh_button.setText("Refreshing..." if busy else "Refresh")
        if hasattr(self, "dataset_plan_progress"):
            if busy:
                self.dataset_plan_progress.setRange(0, 0)
                self.dataset_plan_progress.setVisible(True)
            else:
                self.dataset_plan_progress.setRange(0, 100)
                self.dataset_plan_progress.setValue(0)
                self.dataset_plan_progress.setVisible(False)

    def refresh_dataset_blueprint_files(self) -> None:
        """Reload the Dataset Blueprint file tree from disk."""

        root = getattr(self, "blueprint_data_root", default_data_root())
        self._refresh_external_dataset_status()
        selected_paths = [str(path) for path in self._selected_default_data_paths()]
        self._set_dataset_blueprint_refresh_busy(True)
        QApplication.processEvents()
        try:
            self._refresh_dataset_blueprint_source(
                Path(root),
                saved_paths=selected_paths,
                saved_plan=self._dataset_plan_from_ui(),
                preset="Custom",
            )
            self.project_state.setText("Blueprint refreshed")
            LOGGER.info("Dataset blueprint tree refreshed from %s", root)
        finally:
            self._set_dataset_blueprint_refresh_busy(False)
            self._apply_dataset_license_gating()

    def _handle_default_data_tree_changed(self, item: Any, column: int) -> None:
        """Handle category and file toggles in the bundled data tree.

        Args:
            item: Changed tree item.
            column: Changed column index.
        """

        if column != 0 or getattr(self, "default_data_tree_updating", False):
            return
        data = item.data(0, Qt.UserRole) or {}
        if data.get("kind") != "category":
            self.default_data_tree_updating = True
            try:
                self._refresh_default_data_category_states()
            finally:
                self.default_data_tree_updating = False
            if hasattr(self, "_mixture_weights_state"):
                delattr(self, "_mixture_weights_state")
            return
        state = item.checkState(0)
        if state == Qt.PartiallyChecked:
            return
        self.default_data_tree_updating = True
        try:
            for index in range(item.childCount()):
                item.child(index).setCheckState(0, state)
        finally:
            self.default_data_tree_updating = False
        if hasattr(self, "_mixture_weights_state"):
            delattr(self, "_mixture_weights_state")

    def _refresh_default_data_category_states(self) -> None:
        """Refresh category checkbox states from child file selections."""

        if not hasattr(self, "default_data_category_items"):
            return
        for category_item in self.default_data_category_items.values():
            checked = 0
            partial = False
            for index in range(category_item.childCount()):
                state = category_item.child(index).checkState(0)
                if state == Qt.Checked:
                    checked += 1
                elif state == Qt.PartiallyChecked:
                    partial = True
            if partial or 0 < checked < category_item.childCount():
                category_item.setCheckState(0, Qt.PartiallyChecked)
            elif checked == category_item.childCount() and category_item.childCount() > 0:
                category_item.setCheckState(0, Qt.Checked)
            else:
                category_item.setCheckState(0, Qt.Unchecked)

    def _set_dataset_plan(self, plan: dict[str, Any], preset: str = "Custom") -> None:
        """Restore high-level dataset blueprint controls.

        Args:
            plan: Saved dataset domain percentages.
            preset: Saved preset label.
        """

        if not hasattr(self, "dataset_plan_spins"):
            return
        self._restoring_dataset_plan = True
        try:
            values = {**dataset_plan_defaults(), **(plan or {})}
            for key, widget in self.dataset_plan_spins.items():
                widget.blockSignals(True)
                try:
                    widget.setValue(float(values.get(key, 0.0)))
                except (TypeError, ValueError):
                    widget.setValue(0.0)
                widget.blockSignals(False)
            self.dataset_plan_preset.blockSignals(True)
            if preset == "Custom":
                self.dataset_plan_preset.setCurrentText(preset)
            else:
                self.dataset_plan_preset.setCurrentText("Custom")
            self.dataset_plan_preset.blockSignals(False)
        finally:
            self._restoring_dataset_plan = False
        self._update_dataset_plan_total()

    def _dataset_plan_mark_custom(self, *_args: Any) -> None:
        """Mark the dataset blueprint as custom after manual edits."""

        if getattr(self, "_restoring_dataset_plan", False):
            return
        if hasattr(self, "_mixture_weights_state"):
            delattr(self, "_mixture_weights_state")
        if hasattr(self, "dataset_plan_preset") and self.dataset_plan_preset.currentText() != "Custom":
            self.dataset_plan_preset.blockSignals(True)
            self.dataset_plan_preset.setCurrentText("Custom")
            self.dataset_plan_preset.blockSignals(False)

    def _update_dataset_plan_total(self) -> None:
        """No-op retained for compatibility after blueprint percentage removal."""

        return

    def normalize_dataset_plan(self) -> None:
        """No-op retained for compatibility after blueprint percentage removal."""

        return

    def apply_dataset_plan_preset(self, preset: str) -> None:
        """No-op retained for compatibility after blueprint percentage removal.

        Args:
            preset: Preset label from the Dataset Blueprint combo box.
        """

        return

    def apply_dataset_plan_to_ingestion(self) -> None:
        """Clear ingestion mixture overrides (category percentages are disabled)."""

        self._set_mixture_weights({})
        if hasattr(self, "dataset_log"):
            self.dataset_log.append("Dataset blueprint applied: category percentages are disabled.")
        self.project_state.setText("Blueprint applied")
        LOGGER.info("Dataset blueprint applied with category percentages disabled")

    def _mixture_weights_from_ui(self) -> dict[str, float]:
        """Return dataset mixture weights from the Ingest tab.

        Returns:
            Empty mapping because category percentages are disabled.
        """

        if not hasattr(self, "_mixture_weights_state"):
            self._mixture_weights_state = {}
        return {}

    def _set_mixture_weights(self, weights: dict[str, Any]) -> None:
        """Restore dataset mixture weights.

        Args:
            weights: Saved mixture weights by source family.
        """

        self._mixture_weights_state = {}

    def _update_mixture_total(self) -> None:
        """No-op retained for compatibility after mixture percentage removal."""

        return

    def _normalize_mixture_weights(self) -> None:
        """No-op retained for compatibility after mixture percentage removal."""

        return

    def _training_launch_target_value(self) -> str:
        """Return whether training should launch locally or remotely.

        Returns:
            ``local`` or ``remote``.
        """

        if self.training_launch_target.currentText() == "RunPod cloud":
            return "runpod"
        return "remote" if self.training_launch_target.currentText() == "Remote workers" else "local"

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

    def _training_output_dir_for_mode(self, training_mode: Optional[str]) -> Path:
        """Return the output folder for a training mode.

        Args:
            training_mode: Training mode override.

        Returns:
            Base model or fine-tune output folder.
        """

        return self._fine_tune_output_path() if training_mode == "fine_tune" else Path(self.model_dir.text())

