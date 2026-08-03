from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart2:
    def _build_training_tab(self) -> QWidget:
        """Build the training configuration page.

        Returns:
            Training page widget.
        """

        return build_training_tab(self)

    def _build_fine_tuning_tab(self) -> QWidget:
        """Build the fine-tuning page.

        Returns:
            Fine-tuning page widget.
        """

        return build_fine_tuning_tab(self)

    def _build_live_training_tab(self) -> QWidget:
        """Build the live training tracker page.

        Returns:
            Live training tracker page widget.
        """

        return build_live_training_tab(self)

    def _build_job_manager_tab(self) -> QWidget:
        """Build the distributed job manager page.

        Returns:
            Job manager page widget.
        """

        return build_job_manager_tab(self)

    def refresh_job_manager_tab(self) -> None:
        """Refresh the job manager dashboard tables."""

        if not hasattr(self, "job_worker_table"):
            return
        workers = self.job_manager.list_workers()
        jobs = self.job_manager.list_jobs()
        heartbeats = self.job_manager.state_store.latest_heartbeats()
        worker_rows = []
        for worker in workers:
            heartbeat = heartbeats.get(worker.worker_id, {})
            metrics = heartbeat.get("metrics") or {}
            active_job = heartbeat.get("active_job_id") or self._active_job_for_worker(worker.worker_id)
            capabilities = worker.capabilities or {}
            cpu_ram_gpu = (
                f"CPU {capabilities.get('cpu_count', '-')}, "
                f"RAM {capabilities.get('system_ram_gb', '-')} GB, "
                f"VRAM {capabilities.get('total_vram_gb', '-')} GB"
            )
            if metrics:
                cpu_ram_gpu = f"{cpu_ram_gpu}, util {metrics.get('gpu_util', metrics.get('gpu_memory_percent', '-'))}"
            worker_rows.append(
                [
                    worker.worker_id,
                    worker.status.value,
                    worker.backend.value,
                    worker.device,
                    worker.last_heartbeat_at or "-",
                    active_job or "-",
                    cpu_ram_gpu,
                    ", ".join(capabilities.get("labels") or []) or "-",
                ]
            )
        set_table_rows(self.job_worker_table, worker_rows)

        job_rows = []
        for managed in jobs:
            job = managed.spec
            metrics = managed.latest_metrics
            stage_label = str(job.metadata.get("training_stage") or job.metadata.get("training_mode") or job.training.training_mode)
            job_rows.append(
                [
                    job.job_id,
                    stage_label,
                    job.status.value,
                    managed.assigned_worker_id or "-",
                    job.runtime.backend.value,
                    self._metric_pair(metrics.epoch if metrics else None, metrics.total_epochs if metrics else None),
                    self._metric_pair(metrics.step if metrics else None, metrics.total_steps if metrics else None),
                    str(job.training.batch_size),
                    str(job.model.config.layer_count),
                    self._metric_float(metrics.train_loss if metrics else None),
                    self._metric_float(metrics.tokens_per_second if metrics else None, suffix=" tok/s"),
                    managed.updated_at,
                ]
            )
        set_table_rows(self.job_table, job_rows)
        active_count = sum(1 for item in jobs if item.spec.status.value in {"assigned", "running", "paused", "stopping"})
        queued_count = sum(1 for item in jobs if item.spec.status.value == "queued")
        self.job_worker_count_label.setText(f"Workers: {len(workers)}")
        self.job_active_count_label.setText(f"Active jobs: {active_count}")
        self.job_queue_count_label.setText(f"Queued jobs: {queued_count}")
        self.job_db_label.setText(f"State DB: {self.job_manager.state_store.db_path}")
        self.job_manager_progress.setValue(100)

    def pause_all_managed_jobs(self) -> None:
        """Pause all managed jobs."""

        count = self.job_manager.pause_all_jobs()
        self.job_manager_log.append(f"Pause requested for {count} job(s).")
        self.refresh_job_manager_tab()

    def resume_all_managed_jobs(self) -> None:
        """Resume all paused managed jobs."""

        count = self.job_manager.resume_all_jobs()
        self.job_manager_log.append(f"Resumed {count} job(s).")
        self.refresh_job_manager_tab()

    def stop_all_managed_jobs(self) -> None:
        """Stop all managed jobs."""

        count = self.job_manager.stop_all_jobs()
        self.job_manager_log.append(f"Stop requested for {count} job(s).")
        self.refresh_job_manager_tab()

    def mark_stale_workers_offline(self) -> None:
        """Mark stale remote workers offline."""

        workers = self.job_manager.mark_stale_workers_offline()
        if workers:
            self.job_manager_log.append(f"Marked offline: {', '.join(workers)}")
        else:
            self.job_manager_log.append("No stale remote workers found.")
        self.refresh_job_manager_tab()

    def start_coordinator_server(self) -> None:
        """Start the coordinator API used by remote workers."""

        if self.coordinator_server is not None:
            self.job_manager_log.append("Coordinator API is already running.")
            return
        host = self.coordinator_host.text().strip() or "0.0.0.0"
        port = self.coordinator_port.value()
        artifact_root = Path(self.coordinator_artifact_root.text().strip()).expanduser()
        artifact_root.mkdir(parents=True, exist_ok=True)
        try:
            self.coordinator_server = CoordinatorApiServer(
                manager=self.job_manager,
                host=host,
                port=port,
                artifact_root=artifact_root,
            )
            self.coordinator_thread = Thread(target=self.coordinator_server.serve_forever, daemon=True)
            self.coordinator_thread.start()
        except Exception as exc:
            self.coordinator_server = None
            self.coordinator_thread = None
            QMessageBox.warning(self, "Coordinator failed", f"Could not start coordinator API:\n{exc}")
            return
        public_url = self.coordinator_public_url.text().strip() or f"http://127.0.0.1:{port}"
        self.coordinator_public_url.setText(public_url.rstrip("/"))
        self.coordinator_status_label.setText(f"Coordinator: running at {public_url.rstrip('/')}")
        self.coordinator_start_button.setEnabled(False)
        self.coordinator_stop_button.setEnabled(True)
        self.project_state.setText("Coordinator running")
        self.job_manager_log.append(f"Coordinator API started on {host}:{port}.")
        self.job_manager_log.append(f"Artifact sync root: {artifact_root}")

    def stop_coordinator_server(self) -> None:
        """Stop the coordinator API."""

        if self.coordinator_server is None:
            return
        self.coordinator_server.shutdown()
        if self.coordinator_thread is not None:
            self.coordinator_thread.join(timeout=3)
        self.coordinator_server = None
        self.coordinator_thread = None
        self.coordinator_status_label.setText("Coordinator: stopped")
        self.coordinator_start_button.setEnabled(True)
        self.coordinator_stop_button.setEnabled(False)
        self.project_state.setText("Coordinator stopped")
        self.job_manager_log.append("Coordinator API stopped.")

    def _runpod_config_path(self) -> Path:
        """Return the active RunPod config path.

        Returns:
            Project-local RunPod config path when a project is open.
        """

        project_dir = self.current_project_file.parent if self.current_project_file is not None else None
        return default_runpod_config_path(project_dir)

    def load_runpod_settings(self) -> None:
        """Load RunPod settings into the Job Manager UI."""

        if not hasattr(self, "runpod_api_key"):
            return
        config_path = self._runpod_config_path()
        try:
            config = load_runpod_config(config_path)
        except Exception as exc:
            LOGGER.error("Could not load RunPod config: %s", exc)
            self.runpod_status_label.setText(f"RunPod config error: {exc}")
            return
        self.runpod_api_key.setText(config.api_key)
        self._set_combo_text(self.runpod_gpu_type, config.gpu_type_id)
        self._set_combo_text(self.runpod_cloud_type, config.cloud_type)
        self.runpod_image.setText(config.image_name)
        self.runpod_container_disk.setValue(config.container_disk_gb)
        self.runpod_volume_gb.setValue(config.volume_gb)
        self.runpod_min_ram.setValue(config.min_ram_per_gpu)
        self.runpod_min_vcpu.setValue(config.min_vcpu_per_gpu)
        self.runpod_spot.setChecked(config.interruptible)
        self.runpod_auto_terminate.setChecked(config.auto_terminate)
        status = "configured" if config.api_key.strip() else "API key needed"
        self.runpod_status_label.setText(f"RunPod: {status} ({config_path})")

    def save_runpod_settings(self) -> None:
        """Save RunPod settings from the Job Manager UI."""

        config = self._runpod_config_from_ui()
        config_path = self._runpod_config_path()
        save_runpod_config(config_path, config)
        self.runpod_status_label.setText(f"RunPod settings saved: {config_path}")
        self.job_manager_log.append(f"RunPod settings saved: {config_path}")
        LOGGER.info("RunPod settings saved: %s", config_path)

    def _runpod_config_from_ui(self) -> RunPodConfig:
        """Collect RunPod settings from the UI.

        Returns:
            RunPod configuration.
        """

        return RunPodConfig(
            api_key=self.runpod_api_key.text().strip(),
            image_name=self.runpod_image.text().strip(),
            gpu_type_id=self.runpod_gpu_type.currentText().strip(),
            gpu_count=1,
            cloud_type=self.runpod_cloud_type.currentText().strip(),
            interruptible=self.runpod_spot.isChecked(),
            container_disk_gb=self.runpod_container_disk.value(),
            volume_gb=self.runpod_volume_gb.value(),
            min_vcpu_per_gpu=self.runpod_min_vcpu.value(),
            min_ram_per_gpu=self.runpod_min_ram.value(),
            auto_terminate=self.runpod_auto_terminate.isChecked(),
            worker_labels="runpod,gpu",
        )

    def launch_runpod_worker_for_current_training(self, training_mode: str = "pretrain", stage: str = "base") -> None:
        """Publish the current training job and launch a RunPod worker Pod.

        Args:
            training_mode: Training mode for the queued job.
            stage: Dataset/training stage label.
        """

        if isinstance(training_mode, bool):
            training_mode = "pretrain"
            stage = "base"
        try:
            config = self._runpod_config_from_ui()
            save_runpod_config(self._runpod_config_path(), config)
            coordinator_url = self.coordinator_public_url.text().strip().rstrip("/")
            if not public_url_is_cloud_reachable(coordinator_url):
                raise ValueError(
                    "RunPod needs a public Worker URL. Start a tunnel or set Worker URL to a public address, "
                    "not localhost/127.0.0.1."
                )
            if self.coordinator_server is None:
                self.start_coordinator_server()
                if self.coordinator_server is None:
                    return
            job, bundle_path = self._publish_remote_training_job_spec(
                training_mode=training_mode,
                stage=stage,
                backend_label="runpod",
            )
            artifact_root = Path(self.coordinator_artifact_root.text().strip()).expanduser()
            bootstrap_path = create_runpod_worker_bundle(Path(__file__).resolve().parents[2], artifact_root)
            bootstrap_url = f"{coordinator_url}/artifacts/{bootstrap_path.name}"
            worker_id = f"runpod-{job.job_id}"
            pod_name = f"micro-llm-{self._safe_project_name(self.search_box.text().strip() or 'project')}-{job.job_id[-8:]}"
            result = RunPodClient(config.api_key).create_worker_pod(
                config=config,
                pod_name=pod_name,
                worker_id=worker_id,
                coordinator_url=coordinator_url,
                bootstrap_url=bootstrap_url,
            )
            managed = self.job_manager.get_job(job.job_id)
            managed.spec.metadata["runpod_pod_id"] = result.pod_id
            managed.spec.metadata["runpod_worker_id"] = result.worker_id
            managed.spec.metadata["runpod_cost_per_hour"] = result.cost_per_hour
            self.job_manager._persist_job(job.job_id)
        except Exception as exc:
            LOGGER.exception("RunPod launch failed")
            QMessageBox.warning(self, "RunPod launch failed", str(exc))
            if hasattr(self, "runpod_status_label"):
                self.runpod_status_label.setText(f"RunPod launch failed: {exc}")
            return
        self.runpod_status_label.setText(
            f"RunPod pod {result.pod_id} launched for {job.job_id} ({result.gpu_name}, {result.cost_per_hour}/hr)"
        )
        self.job_manager_log.append(f"RunPod pod launched: {result.pod_id}")
        self.job_manager_log.append(f"RunPod worker: {result.worker_id}")
        self.job_manager_log.append(f"RunPod GPU: {result.gpu_name}, cost/hr: {result.cost_per_hour}")
        self.job_manager_log.append(f"Worker bootstrap: {result.bootstrap_url}")
        self.project_state.setText("RunPod worker launched")
        self.refresh_job_manager_tab()

    def publish_remote_training_job(self, training_mode: str = "pretrain", stage: str = "base") -> None:
        """Bundle the current training setup and queue it for remote workers.

        Args:
            training_mode: Trainer mode to publish, either ``pretrain`` or ``fine_tune``.
            stage: Higher-level stage label for job manager display.
        """

        if isinstance(training_mode, bool):
            training_mode = "pretrain"
            stage = "base"
        if self.coordinator_server is None:
            self.start_coordinator_server()
            if self.coordinator_server is None:
                return
        try:
            job, bundle_path = self._publish_remote_training_job_spec(training_mode=training_mode, stage=stage)
        except Exception as exc:
            QMessageBox.warning(self, "Publish failed", f"Could not publish remote job:\n{exc}")
            return
        self.job_manager_log.append(f"Published remote job: {job.job_id}")
        self.job_manager_log.append(f"Input bundle: {bundle_path}")
        self.job_manager_log.append(f"Worker download URL: {job.metadata.get('artifact_bundle_url')}")
        self.project_state.setText("Remote job queued")
        self.refresh_job_manager_tab()

    def _publish_remote_training_job_spec(
        self,
        training_mode: str = "pretrain",
        stage: str = "base",
        backend_label: str = "remote",
    ) -> tuple[TrainingJobSpec, Path]:
        """Bundle and queue the current remote training job.

        Args:
            training_mode: Trainer mode to publish.
            stage: Higher-level stage label.
            backend_label: Human-readable backend label stored in metadata.

        Returns:
            Queued job and bundle path.
        """

        job = self._current_remote_training_job(training_mode=training_mode, stage=stage)
        job.metadata["launch_backend"] = backend_label
        artifact_root = Path(self.coordinator_artifact_root.text().strip()).expanduser()
        base_url = f"{self.coordinator_public_url.text().strip().rstrip('/')}/artifacts"
        bundle_path = create_job_artifact_bundle(job, artifact_root=artifact_root, base_url=base_url)
        self.job_manager.submit(job)
        return job, bundle_path

    def _current_remote_training_job(self, training_mode: str = "pretrain", stage: str = "base") -> TrainingJobSpec:
        """Build a remote-worker job from current training controls.

        Args:
            training_mode: Trainer mode to publish.
            stage: Higher-level stage label for job manager display.

        Returns:
            Complete training job spec ready to bundle and queue.

        Raises:
            FileNotFoundError: If the prepared dataset is missing.
            ValueError: If model or training options are invalid.
        """

        dataset_dir = Path(self.train_data_dir.text().strip())
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Prepared dataset folder does not exist: {dataset_dir}")
        if not self._dataset_artifacts_exist(dataset_dir):
            raise FileNotFoundError(
                "Prepared dataset is missing tokenizer or token files. "
                "Expected tokenizer.json plus train/val tokens in .npy or .json."
            )
        vocab_size = self._current_training_vocab_size(dataset_dir)
        if vocab_size <= 0:
            raise ValueError("Could not determine tokenizer vocabulary size from the prepared dataset.")
        resume_path = Path(self.resume_checkpoint.text()) if self.resume_checkpoint.text().strip() else None
        if resume_path is None and self.resume_training.isChecked():
            resume_path = latest_checkpoint(self._training_output_dir_for_mode(training_mode) / "checkpoints")
        model_config = self._current_model_config(vocab_size=vocab_size)
        training_config = self._current_training_config(resume_path, training_mode=training_mode)
        model_config.validate()
        training_config.validate()
        job = TrainingJobSpec.local(
            dataset_dir,
            model_config,
            training_config,
            metadata={
                "project_name": self.search_box.text().strip(),
                "submitted_from": "desktop_ui",
                "coordinator_url": self.coordinator_public_url.text().strip().rstrip("/"),
                "training_mode": training_mode,
                "training_stage": stage,
            },
        )
        job.runtime = RuntimeSpec(
            backend=BackendKind.REMOTE_CLIENT,
            device=training_config.device,
            tags=[training_config.device, "remote"],
        )
        return job

    def _active_job_for_worker(self, worker_id: str) -> str:
        """Return the active job ID for a worker.

        Args:
            worker_id: Worker identifier.

        Returns:
            Active job ID or empty string.
        """

        for managed in self.job_manager.list_jobs():
            if managed.assigned_worker_id == worker_id and managed.spec.status.value in {"assigned", "running", "paused", "stopping"}:
                return managed.spec.job_id
        return ""


