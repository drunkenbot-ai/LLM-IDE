"""Screen mixin for managing port-blocked distributed cluster training (Local SGD)."""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from cluster.bus import ClusterStorageBus
from cluster.worker import ClusterWorker
from interface.tabs.cluster_tab import set_cluster_table_rows


class ClusterScreenMixin:
    """Mixin for MainWindow handling distributed Local SGD cluster operations."""

    _local_cluster_worker: Optional[ClusterWorker] = None
    _local_cluster_thread: Optional[threading.Thread] = None
    _cached_cluster_bus: Optional[ClusterStorageBus] = None
    _cached_cluster_path: Optional[str] = None
    _cluster_poll_in_progress: bool = False

    def _get_cluster_bus(self) -> Optional[ClusterStorageBus]:
        """Get or initialize storage bus from the configured shared directory, using caching."""
        if not hasattr(self, "cluster_shared_dir"):
            return None
        path_str = self.cluster_shared_dir.text().strip()
        if not path_str:
            return None
        if self._cached_cluster_path == path_str and self._cached_cluster_bus is not None:
            return self._cached_cluster_bus
        try:
            p = Path(path_str)
            bus = ClusterStorageBus(p)
            self._cached_cluster_bus = bus
            self._cached_cluster_path = path_str
            return bus
        except Exception as exc:
            self._log_cluster_event(f"Error accessing shared storage: {exc}")
            return None

    def _log_cluster_event(self, message: str) -> None:
        """Append a timestamped message to the cluster event log."""
        if hasattr(self, "cluster_log"):
            now_str = time.strftime("%H:%M:%S")
            self.cluster_log.append(f"[{now_str}] {message}")

    def browse_cluster_shared_dir(self) -> None:
        """Browse filesystem for central shared network drive folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Central Shared Network Directory (SMB/NFS/NAS)",
            self.cluster_shared_dir.text() if hasattr(self, "cluster_shared_dir") else "",
        )
        if folder and hasattr(self, "cluster_shared_dir"):
            self.cluster_shared_dir.setText(folder)
            self._cached_cluster_bus = None
            self._cached_cluster_path = None
            self.refresh_cluster_status()

    def refresh_cluster_status(self) -> None:
        """Poll the SQLite bus in a background thread to prevent UI thread freezing."""
        if not hasattr(self, "cluster_shared_dir"):
            return
        path_str = self.cluster_shared_dir.text().strip()
        if not path_str:
            if hasattr(self, "cluster_status_label"):
                self.cluster_status_label.setText("Status: Configure shared storage path")
            return

        if self._cluster_poll_in_progress:
            return
        self._cluster_poll_in_progress = True

        def _bg_poll() -> None:
            try:
                p = Path(path_str)
                if not p.exists():
                    QTimer.singleShot(0, lambda: self._apply_cluster_telemetry(None, None, error="Folder not found"))
                    return

                if self._cached_cluster_path != path_str or self._cached_cluster_bus is None:
                    self._cached_cluster_bus = ClusterStorageBus(p)
                    self._cached_cluster_path = path_str

                bus = self._cached_cluster_bus
                workers = bus.list_workers(active_within_seconds=60.0)
                active_job = bus.get_active_job()
                QTimer.singleShot(0, lambda w=workers, j=active_job: self._apply_cluster_telemetry(w, j))
            except Exception as exc:
                err_msg = str(exc)
                QTimer.singleShot(0, lambda m=err_msg: self._apply_cluster_telemetry(None, None, error=m))
            finally:
                self._cluster_poll_in_progress = False

        threading.Thread(target=_bg_poll, daemon=True).start()

    def _apply_cluster_telemetry(
        self,
        workers: Optional[list[dict[str, Any]]],
        active_job: Optional[dict[str, Any]],
        error: Optional[str] = None,
    ) -> None:
        """Apply polled cluster telemetry to UI widgets on the main thread."""
        if error:
            if hasattr(self, "cluster_status_label"):
                self.cluster_status_label.setText(f"Status: {error}")
            return

        if workers is not None:
            online_count = sum(1 for w in workers if w.get("is_online"))
            if hasattr(self, "cluster_workers_label"):
                self.cluster_workers_label.setText(f"Active Workers: {online_count} / {len(workers)}")

            if hasattr(self, "cluster_worker_table"):
                rows = []
                for w in workers:
                    last_hb = w.get("last_heartbeat")
                    last_hb_str = (
                        time.strftime("%H:%M:%S", time.localtime(last_hb))
                        if last_hb
                        else "-"
                    )
                    rows.append([
                        str(w.get("worker_id", "-")),
                        str(w.get("hostname", "-")),
                        str(w.get("gpu_name", "-")),
                        str(w.get("vram_gb", 0)),
                        str(w.get("status", "OFFLINE")),
                        last_hb_str,
                    ])
                set_cluster_table_rows(self.cluster_worker_table, rows)

        if active_job:
            jid = active_job["job_id"]
            st = active_job["status"]
            cur_round = active_job.get("current_round", 0)
            max_rounds = active_job.get("max_rounds", 10)
            if hasattr(self, "cluster_status_label"):
                self.cluster_status_label.setText(f"Status: Job {jid} ({st})")
            if hasattr(self, "cluster_round_label"):
                self.cluster_round_label.setText(f"Round: {cur_round} / {max_rounds}")
        else:
            if hasattr(self, "cluster_status_label"):
                self.cluster_status_label.setText("Status: Fleet Idle")
            if hasattr(self, "cluster_round_label"):
                self.cluster_round_label.setText("Round: -")

    def launch_cluster_training_job(self) -> None:
        """Submit a distributed Local SGD training job to the shared drive."""
        bus = self._get_cluster_bus()
        if not bus:
            QMessageBox.warning(self, "Cluster Error", "Please configure a valid shared storage directory first.")
            return

        # Find dataset path (train_tokens.npy or configured dataset)
        dataset_path = None
        if hasattr(self, "_dataset_manifest_path"):
            manifest_dir = Path(self._dataset_manifest_path()).parent
            candidate = manifest_dir / "train_tokens.npy"
            if candidate.exists():
                dataset_path = str(candidate)

        if not dataset_path and hasattr(self, "output_dir_edit"):
            out_dir = Path(self.output_dir_edit.text())
            candidate = out_dir / "train_tokens.npy"
            if candidate.exists():
                dataset_path = str(candidate)

        if not dataset_path:
            # Check shared directory itself
            candidate = bus.shared_dir / "train_tokens.npy"
            if candidate.exists():
                dataset_path = str(candidate)

        if not dataset_path:
            # Prompt user for dataset path
            dataset_path, _ = QFileDialog.getOpenFileName(
                self, "Select Token Dataset (.npy)", str(bus.shared_dir), "NumPy Token Arrays (*.npy)"
            )

        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Dataset Missing", "A valid train_tokens.npy dataset file is required to launch cluster training.")
            return

        try:
            # Build configs
            model_cfg = dataclasses.asdict(self._current_model_config()) if hasattr(self, "_current_model_config") else {
                "vocab_size": 1000,
                "context_length": 512,
                "embedding_size": 256,
                "head_count": 4,
                "layer_count": 4,
            }
            training_cfg = dataclasses.asdict(self._current_training_config()) if hasattr(self, "_current_training_config") else {
                "learning_rate": 3e-4,
                "batch_size": 4,
            }

            job_id = f"cluster_job_{int(time.time())}"
            sync_steps = self.cluster_sync_steps.value() if hasattr(self, "cluster_sync_steps") else 250
            max_rounds = self.cluster_max_rounds.value() if hasattr(self, "cluster_max_rounds") else 10
            sync_timeout = float(self.cluster_sync_timeout.value() if hasattr(self, "cluster_sync_timeout") else 180)
            min_workers = self.cluster_min_workers.value() if hasattr(self, "cluster_min_workers") else 1

            bus.create_job(
                job_id=job_id,
                model_config=model_cfg,
                training_config=training_cfg,
                dataset_path=dataset_path,
                max_rounds=max_rounds,
                sync_interval_steps=sync_steps,
                min_workers=min_workers,
                sync_timeout_seconds=sync_timeout,
            )
            bus.set_job_status(job_id, "RUNNING")
            self._log_cluster_event(f"Successfully queued job {job_id} across cluster.")
            self.refresh_cluster_status()

        except Exception as exc:
            QMessageBox.critical(self, "Launch Error", f"Failed to submit cluster job:\n{exc}")
            self._log_cluster_event(f"Launch failed: {exc}")

    def pause_cluster_job(self) -> None:
        """Cooperatively pause active cluster job."""
        bus = self._get_cluster_bus()
        if not bus:
            return
        active = bus.get_active_job()
        if active:
            bus.set_job_status(active["job_id"], "PAUSED")
            self._log_cluster_event(f"Signal PAUSE set for job {active['job_id']}.")
            self.refresh_cluster_status()

    def resume_cluster_job(self) -> None:
        """Resume paused cluster job."""
        bus = self._get_cluster_bus()
        if not bus:
            return
        active = bus.get_active_job()
        if active:
            bus.set_job_status(active["job_id"], "RUNNING")
            self._log_cluster_event(f"Signal RESUME set for job {active['job_id']}.")
            self.refresh_cluster_status()

    def stop_cluster_job(self) -> None:
        """Stop active cluster job."""
        bus = self._get_cluster_bus()
        if not bus:
            return
        active = bus.get_active_job()
        if active:
            bus.set_job_status(active["job_id"], "STOPPED")
            self._log_cluster_event(f"Signal STOP set for job {active['job_id']}.")
            self.refresh_cluster_status()

    def toggle_local_cluster_worker(self) -> None:
        """Start or stop a local background cluster worker on this machine."""
        if self._local_cluster_worker is not None:
            # Stop existing worker
            self._local_cluster_worker.stop()
            self._local_cluster_worker = None
            self._local_cluster_thread = None
            if hasattr(self, "cluster_local_worker_btn"):
                self.cluster_local_worker_btn.setText("Start Local Worker")
            self._log_cluster_event("Stopped local worker daemon.")
            self.refresh_cluster_status()
        else:
            bus = self._get_cluster_bus()
            if not bus:
                QMessageBox.warning(self, "Error", "Shared network directory not configured.")
                return
            worker = ClusterWorker(bus=bus)
            self._local_cluster_worker = worker

            def _run():
                worker.run_daemon(poll_interval=2.0)

            t = threading.Thread(target=_run, daemon=True)
            self._local_cluster_thread = t
            t.start()

            if hasattr(self, "cluster_local_worker_btn"):
                self.cluster_local_worker_btn.setText("Stop Local Worker")
            self._log_cluster_event(f"Started local worker {worker.worker_id} on {worker.device_str}.")
            self.refresh_cluster_status()
