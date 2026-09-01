from __future__ import annotations

# ProjectManagerMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class ProjectManagerMixin:
    def save_project(self) -> None:
        """Save the current project settings into a named project folder."""

        project_name = self.search_box.text().strip() or "DrunkenBotProject"
        safe_name = self._safe_project_name(project_name)
        if self.current_project_file is None:
            base_dir = QFileDialog.getExistingDirectory(self, "Choose parent folder for project", self._project_dialog_start_dir())
            if not base_dir:
                return
            project_dir = Path(base_dir) / safe_name
            project_file = project_dir / "project.json"
        else:
            project_file = self.current_project_file
            project_dir = project_file.parent
        project_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_project_workspace(project_dir)
        if self.current_project_file is None:
            self._apply_project_workspace_paths(project_dir)
        project_file.write_text(json.dumps(self._project_state_dict(project_name, project_dir), indent=2), encoding="utf-8")
        self.current_project_file = project_file
        _register_recent_project(project_file)
        self._apply_project_runtime_environment(project_dir)
        self._refresh_notification_manager(project_dir)
        if hasattr(self, "runpod_api_key"):
            self.load_runpod_settings()
        self.project_state.setText("Project saved")
        LOGGER.info("Project saved: %s", project_file)
        if self.current_project_file == project_file:
            self.dataset_log.append(f"Project saved: {project_file}")
            self.dataset_log.append(f"Project workspace: {project_dir}")
            self.dataset_log.append(f"Notifier config: {project_dir / 'notifier_config.json'}")

    def new_project(self) -> None:
        """Start a fresh project and clear the active project file binding."""

        if self.thread is not None or self.training_controller.active:
            QMessageBox.information(self, "Task running", "Please stop or wait for the current task before creating a new project.")
            return
        if self.current_project_file is not None or self.search_box.text().strip():
            choice = QMessageBox.question(
                self,
                "New project",
                "Start a new project? Unsaved changes in the current project will not be saved automatically.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

        base_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose folder where the new project will be created",
            self._project_dialog_start_dir(),
        )
        if not base_dir:
            self.project_state.setText("Project creation cancelled")
            return
        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        project_name = self.search_box.text().strip() or "DrunkenBotProject"
        try:
            self._create_project_at(project_name, Path(base_dir))
        except Exception as exc:
            QMessageBox.warning(self, "New project failed", f"Could not create project:\n{exc}")

    def open_project(self) -> None:
        """Open a saved project file and restore UI settings."""

        project_file, _ = QFileDialog.getOpenFileName(
            self,
            "Open Micro LLM project",
            self._project_dialog_start_dir(),
            "Micro LLM project (project.json *.json);;All files (*)",
        )
        if not project_file:
            return
        try:
            self._open_project_file(Path(project_file))
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", f"Could not open project:\n{exc}")
            return

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
        self.discover_training_run()
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

    def _open_project_file(self, project_file: Path) -> None:
        """Open and activate a project file.

        Args:
            project_file: Path to ``project.json``.
        """

        self.training_controller.detach()
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
        self.discover_training_run()
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
