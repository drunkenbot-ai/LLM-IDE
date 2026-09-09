from __future__ import annotations

# JobManagerScreenMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class JobManagerScreenMixin:
    def refresh_job_manager_tab(self) -> None:
        """Refresh the job manager dashboard tables."""

        if not hasattr(self, "job_worker_table"):
            return
        if hasattr(self, "pages") and self.pages.currentIndex() != 5:
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
        if hasattr(self, "refresh_cluster_status"):
            self.refresh_cluster_status()

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
        """Return the active RunPod config path (deprecated)."""
        project_dir = self.current_project_file.parent if self.current_project_file is not None else None
        return default_runpod_config_path(project_dir)

    def load_runpod_settings(self) -> None:
        """RunPod support has been dropped."""
        pass

    def save_runpod_settings(self) -> None:
        """RunPod support has been dropped."""
        pass

    def _runpod_config_from_ui(self) -> Any:
        """RunPod support has been dropped."""
        return None

    def launch_runpod_worker_for_current_training(self, training_mode: str = "pretrain", stage: str = "base") -> None:
        """Inform user that RunPod integration has been discontinued."""
        QMessageBox.information(
            self,
            "RunPod Discontinued",
            "RunPod cloud integration has been discontinued. Please use the Port-Blocked Cluster "
            "subsystem (Local SGD over shared network storage) or local/remote workers.",
        )

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
