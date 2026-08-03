from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart6:
    def _create_project_at(self, project_name: str, base_dir: Path) -> Path:
        """Create and activate a new project at the selected folder.

        Args:
            project_name: User-facing project name.
            base_dir: Parent folder for the new project.

        Returns:
            Path to the created project.json file.
        """

        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        project_dir = base_dir / self._safe_project_name(project_name)
        project_file = project_dir / "project.json"
        project_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_project_workspace(project_dir)
        copied_count = self._ensure_project_training_data(project_dir)
        self.current_project_file = project_file
        self._apply_project_state(self._default_project_state())
        self.search_box.setText(project_name)
        self._apply_project_workspace_paths(project_dir)
        self._reset_dataset_blueprint_source(project_dir / "training_data")
        self._apply_project_runtime_environment(project_dir)
        self._refresh_notification_manager(project_dir)
        if hasattr(self, "runpod_api_key"):
            self.load_runpod_settings()
        self._reset_project_runtime_state()
        project_file.write_text(json.dumps(self._project_state_dict(project_name, project_dir), indent=2), encoding="utf-8")
        _register_recent_project(project_file)
        self.project_state.setText("New project")
        LOGGER.info("New project created: %s", project_file)
        self.dataset_log.append(f"Started a new project: {project_file}")
        self.dataset_log.append(f"Project workspace: {project_dir}")
        self.dataset_log.append(
            "Bundled training data is no longer included; use Dataset Sources to download or select a dataset."
        )
        self.dataset_log.append(f"Notifier config: {project_dir / 'notifier_config.json'}")
        return project_file

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

    def _open_project_file(self, project_file: Path) -> None:
        """Open and activate a project file.

        Args:
            project_file: Path to ``project.json``.
        """

        data = json.loads(project_file.read_text(encoding="utf-8"))
        self.current_project_file = project_file
        _register_recent_project(project_file)
        self._ensure_project_workspace(self.current_project_file.parent)
        dataset_state = data.get("dataset", {}) if isinstance(data, dict) else {}
        saved_default_data_paths = dataset_state.get("default_data_paths")
        self._refresh_dataset_blueprint_source(
            self.current_project_file.parent / "training_data",
            saved_paths=(list(saved_default_data_paths) if saved_default_data_paths is not None else None),
            saved_plan=dict(dataset_state.get("domain_plan", {})),
            preset=str(dataset_state.get("domain_plan_preset", "Balanced Tiny LLM")),
        )
        self._apply_project_state(data)
        self._apply_project_runtime_environment(self.current_project_file.parent)
        self._refresh_notification_manager(self.current_project_file.parent)
        if hasattr(self, "runpod_api_key"):
            self.load_runpod_settings()
        if self.model_dir.text().strip():
            self._load_existing_telemetry(Path(self.model_dir.text()))
        self.project_state.setText("Project opened")
        LOGGER.info("Project opened: %s", project_file)
        self.dataset_log.append(f"Opened project: {project_file}")
        self.dataset_log.append(f"Notifier config: {self.current_project_file.parent / 'notifier_config.json'}")
        self.refresh_model_estimate()

    def _project_dialog_start_dir(self) -> str:
        """Return the best initial folder for project dialogs.

        Returns:
            Active project folder, its parent, or the current folder.
        """

        if self.current_project_file is not None:
            return str(self.current_project_file.parent)
        text = self.dataset_dir.text().strip() if hasattr(self, "dataset_dir") else ""
        if text:
            path = Path(text)
            for candidate in (path, path.parent):
                if candidate.exists():
                    return str(candidate)
        return str(Path.cwd())

    def _ensure_project_workspace(self, project_dir: Path) -> None:
        """Create standard folders inside a project.

        Args:
            project_dir: Project root folder.
        """

        for name in ("datasets", "models", "fine_tunes", "exports", "training_data", "cache", "temp"):
            (project_dir / name).mkdir(parents=True, exist_ok=True)
        ensure_notifier_config(project_dir / "notifier_config.json")
        ensure_runpod_config(project_dir / "runpod_config.json")

    def _ensure_project_training_data(self, project_dir: Path) -> int:
        """Create the project training-data folder without bundling corpus files."""
        target_root = project_dir / "training_data"
        target_root.mkdir(parents=True, exist_ok=True)
        return 0

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

    def _refresh_notification_manager(self, project_dir: Optional[Path] = None) -> None:
        """Load notification settings for the current project.

        Args:
            project_dir: Optional project root folder.
        """

        if project_dir is None and self.current_project_file is not None:
            project_dir = self.current_project_file.parent
        config_path = default_notifier_config_path(project_dir)
        self.notification_manager = NotificationManager(config_path)
        LOGGER.info("Notifier config active: %s", config_path)

    def _apply_project_workspace_paths(self, project_dir: Path) -> None:
        """Point project output fields at the standard project folders.

        Args:
            project_dir: Project root folder.
        """

        dataset_dir = project_dir / "datasets"
        model_dir = project_dir / "models"
        fine_tune_dir = project_dir / "fine_tunes"
        export_dir = project_dir / "exports"
        training_data_dir = project_dir / "training_data"
        self.dataset_dir.setText(str(dataset_dir))
        self.train_data_dir.setText(str(dataset_dir))
        self.model_dir.setText(str(model_dir))
        self.fine_tune_checkpoint.setText(str(model_dir / "final_model.pt"))
        self.fine_tune_output_dir.setText(str(fine_tune_dir / "latest"))
        self.export_model_dir.setText(str(model_dir))
        self.export_dir.setText(str(export_dir))
        self.gguf_output_path.setText(str(export_dir / "model.gguf"))
        if not self.input_dir.text().strip():
            self.input_dir.setText(str(training_data_dir))

    def _apply_project_runtime_environment(self, project_dir: Path) -> None:
        """Prefer project-local cache/temp folders for runtime work.

        Args:
            project_dir: Project root folder.
        """

        cache_dir = project_dir / "cache"
        temp_dir = project_dir / "temp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for key in ("TMPDIR", "TEMP", "TMP"):
            os.environ[key] = str(temp_dir)
        for key in ("TORCH_HOME", "HF_HOME", "TRANSFORMERS_CACHE", "PYTORCH_KERNEL_CACHE"):
            os.environ[key] = str(cache_dir / key.lower())
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


