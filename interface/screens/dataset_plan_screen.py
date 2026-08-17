from __future__ import annotations

# DatasetPlanScreenMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class DatasetPlanScreenMixin:
    def _reset_dataset_blueprint_source(self, data_root: Path) -> None:
        """Reset the current Dataset Sources tree without replacing its widget."""
        self.blueprint_data_root = Path(data_root)
        if hasattr(self, "external_dataset_dir"):
            self.external_dataset_dir.setText(str(data_root))
        if hasattr(self, "dataset_plan_source_label"):
            self.dataset_plan_source_label.setText(f"Source: {data_root}")
        if hasattr(self, "default_data_tree"):
            self.default_data_tree.clear()
            self.default_data_actions.clear()
            self.default_data_category_items.clear()
            self.default_data_tree.addTopLevelItem(
                QTreeWidgetItem(["No project data files were found.", "", ""])
            )

    def _refresh_dataset_blueprint_source(
        self,
        data_root: Path,
        saved_paths: Optional[list[Any]] = None,
        saved_plan: Optional[dict[str, Any]] = None,
        preset: str = "Balanced Tiny LLM",
    ) -> None:
        """Rebuild the Dataset Blueprint tab from a source data folder.

        Args:
            data_root: Project-local training data folder.
            saved_paths: Optional selected file paths to restore.
            saved_plan: Optional saved domain recipe.
            preset: Saved recipe preset.
        """

        if not hasattr(self, "pages"):
            self.blueprint_data_root = Path(data_root)
            return
        self.blueprint_data_root = Path(data_root)
        if hasattr(self, "default_data_tree"):
            selected = saved_paths if saved_paths is not None else self._selected_default_data_paths()
            populate_default_data_tree(self, self.blueprint_data_root)
            self._set_selected_default_data_paths(selected)
            if saved_plan is not None:
                self._set_dataset_plan(saved_plan, preset)
            self.dataset_plan_source_label.setText(f"Source: {self.blueprint_data_root}")
            return
        current_index = self.pages.currentIndex()
        old_page = self.pages.widget(0)
        old_page.hide()
        QApplication.processEvents()
        new_page = self._build_dataset_plan_tab()
        self.pages.removeWidget(old_page)
        old_page.setParent(None)
        old_page.deleteLater()
        self.pages.insertWidget(0, new_page)
        if saved_plan is not None:
            self._set_dataset_plan(saved_plan, preset)
        if saved_paths is not None:
            self._set_selected_default_data_paths(saved_paths)
        elif self.current_project_file is not None:
            self._set_selected_default_data_paths(None)
        self.pages.setCurrentIndex(current_index)

    def download_latest_external_dataset(self) -> None:
        """Download the latest managed dataset into the selected install folder."""
        destination = Path(self.external_dataset_dir.text()).expanduser()
        try:
            manifest = load_manifest()
        except Exception as exc:
            self.external_dataset_version.setText(f"Could not load dataset options: {exc}")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Select dataset components")
        dialog.setMinimumWidth(480)
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.addWidget(QLabel(f"Dataset version {manifest.version}"))
        version_selector = QComboBox()
        version_selector.addItem(manifest.version, DEFAULT_MANIFEST_URL)
        installed_version_file = destination / "version.txt"
        if installed_version_file.is_file():
            installed_version = installed_version_file.read_text(encoding="utf-8").strip()
            if installed_version and installed_version != manifest.version:
                version_selector.addItem(
                    installed_version,
                    f"https://github.com/drunkenbot-ai/dataset/releases/download/dataset-v{installed_version}/manifest.json",
                )
        dialog_layout.addWidget(QLabel("Dataset version"))
        dialog_layout.addWidget(version_selector)
        component_tree = QTreeWidget()
        component_tree.setHeaderLabels(["Component", "Files"])
        for category in manifest.categories:
            if category.file_count <= 0:
                continue
            item = QTreeWidgetItem([category.name, str(category.file_count)])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            existing = destination / category.name
            has_existing = existing.exists() and any(existing.rglob("*"))
            item.setCheckState(0, Qt.Checked if has_existing else Qt.Unchecked)
            component_tree.addTopLevelItem(item)
        dialog_layout.addWidget(component_tree)
        buttons = QHBoxLayout()
        download_button = QPushButton("Download selected")
        cancel_button = QPushButton("Cancel")
        buttons.addStretch(1)
        buttons.addWidget(cancel_button)
        buttons.addWidget(download_button)
        dialog_layout.addLayout(buttons)
        cancel_button.clicked.connect(dialog.reject)
        download_button.clicked.connect(dialog.accept)
        if dialog.exec() != QDialog.Accepted:
            return
        categories = [
            component_tree.topLevelItem(index).text(0)
            for index in range(component_tree.topLevelItemCount())
            if component_tree.topLevelItem(index).checkState(0) == Qt.Checked
        ]
        if not categories:
            self.external_dataset_version.setText("Select at least one dataset component.")
            return

        self.dataset_log.append(f"Downloading latest external dataset to {destination}...")
        self.external_dataset_version.setText("Downloading latest dataset...")
        self.dataset_plan_progress.setVisible(True)
        self._run_task(
            partial(download_latest_dataset, manifest_url=version_selector.currentData()),
            (destination, categories),
            self._external_dataset_download_finished,
            self.dataset_log,
            self.dataset_plan_progress,
            with_progress=True,
            button=self.external_dataset_download_button,
            busy_text="Downloading dataset",
            task_kind="dataset_download",
        )

    def _refresh_external_dataset_status(self) -> None:
        """Restore the installed dataset version from the selected folder."""
        if not hasattr(self, "external_dataset_dir"):
            return
        version_file = Path(self.external_dataset_dir.text()).expanduser() / "version.txt"
        if not version_file.is_file():
            self.external_dataset_version.setText("Installed version: not installed")
            self.external_dataset_download_button.setEnabled(True)
            return
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except OSError:
            version = ""
        if version:
            has_dataset_files = any(
                path.is_file() and path.name not in {"version.txt", "manifest.json"}
                for path in Path(self.external_dataset_dir.text()).expanduser().rglob("*")
            )
            if not has_dataset_files:
                self.external_dataset_version.setText("Installed version: not installed")
                self.external_dataset_download_button.setEnabled(True)
                return
            self.external_dataset_version.setText(f"Installed version: {version}")
            try:
                latest = load_manifest()
            except Exception as exc:
                LOGGER.warning("Could not check for a newer dataset release: %s", exc)
                self.external_dataset_download_button.setEnabled(True)
                return
            installed_root = Path(self.external_dataset_dir.text()).expanduser()
            missing_components = [
                category.name
                for category in latest.categories
                if category.file_count > 0
                and not (
                    (installed_root / category.name).exists()
                    and any((installed_root / category.name).rglob("*"))
                )
            ]
            if is_newer_version(latest.version, version):
                self.external_dataset_version.setText(
                    f"Installed version: {version} (update available: {latest.version})"
                )
                self.external_dataset_download_button.setEnabled(True)
            elif missing_components:
                self.external_dataset_version.setText(
                    f"Installed version: {version} ({len(missing_components)} components missing)"
                )
                self.external_dataset_download_button.setEnabled(True)
            else:
                self.external_dataset_download_button.setEnabled(False)
        else:
            self.external_dataset_version.setText("Installed version: not installed")
            self.external_dataset_download_button.setEnabled(True)

    @Slot(object)
    def _external_dataset_download_finished(self, manifest: Any) -> None:
        """Apply the downloaded dataset as the active source vault."""
        self.dataset_log.append(f"Installed external dataset version {manifest.version}.")
        self.external_dataset_version.setText(f"Installed version: {manifest.version}")
        self.external_dataset_download_button.setEnabled(False)
        self.dataset_plan_progress.setVisible(False)
        self._refresh_dataset_blueprint_source(
            Path(self.external_dataset_dir.text()),
            saved_plan=self._dataset_plan_from_ui(),
            preset=(
                self.dataset_plan_preset.currentText()
                if hasattr(self, "dataset_plan_preset")
                else "Balanced Tiny LLM"
            ),
        )
        self.input_dir.setText(self.external_dataset_dir.text())

    def _dataset_plan_from_ui(self) -> dict[str, float]:
        """Return dataset blueprint state.

        Returns:
            Empty mapping because category percentages are disabled.
        """

        return {}

    def _selected_default_data_paths(self) -> list[Path]:
        """Return bundled default data files selected in the Dataset Blueprint.

        Returns:
            Selected bundled data paths.
        """

        if not hasattr(self, "default_data_actions"):
            return [path for path, _category in iter_default_data_files()]
        return [
            Path(path)
            for path, item in self.default_data_actions.items()
            if item.checkState(0) == Qt.Checked
        ]

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
