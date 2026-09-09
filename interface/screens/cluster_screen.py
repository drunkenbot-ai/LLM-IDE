"""Screen mixin for managing port-blocked distributed cluster training (Local SGD)."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox

from cluster.bus import ClusterStorageBus
from interface.tabs.cluster_tab import set_cluster_table_rows


class ClusterTelemetryBridge(QObject):
    """Qt signal bridge for thread-safe worker telemetry updates."""

    telemetry_ready = Signal(object, object, object)  # workers, active_job, error


class ClusterScreenMixin:
    """Mixin for MainWindow handling distributed Local SGD cluster operations."""

    _local_worker_procs: dict[str, subprocess.Popen] = {}
    _cluster_bridge: Optional[ClusterTelemetryBridge] = None
    _cached_cluster_bus: Optional[ClusterStorageBus] = None
    _cached_cluster_path: Optional[str] = None
    _cluster_poll_in_progress: bool = False

    def _ensure_cluster_bridge(self) -> ClusterTelemetryBridge:
        """Lazily initialize the cross-thread telemetry signal bridge."""
        if self._cluster_bridge is None:
            self._cluster_bridge = ClusterTelemetryBridge()
            self._cluster_bridge.telemetry_ready.connect(self._apply_cluster_telemetry)
        return self._cluster_bridge

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
        # Check local worker process liveness
        if hasattr(self, "_local_worker_procs") and self._local_worker_procs:
            alive = {}
            for dev, proc in list(self._local_worker_procs.items()):
                if proc.poll() is None:
                    alive[dev] = proc
                else:
                    self._log_cluster_event(f"Local worker for '{dev}' exited (code {proc.poll()}).")
            self._local_worker_procs = alive
            if hasattr(self, "cluster_local_worker_btn"):
                if alive:
                    self.cluster_local_worker_btn.setText(f"Stop Local Worker(s) ({len(alive)} active)")
                else:
                    self.cluster_local_worker_btn.setText("Start Local Worker(s)")

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

        bridge = self._ensure_cluster_bridge()

        def _bg_poll() -> None:
            try:
                p = Path(path_str)
                if not p.exists():
                    bridge.telemetry_ready.emit(None, None, "Folder not found")
                    return

                if self._cached_cluster_path != path_str or self._cached_cluster_bus is None:
                    self._cached_cluster_bus = ClusterStorageBus(p)
                    self._cached_cluster_path = path_str

                bus = self._cached_cluster_bus
                workers = bus.list_workers(active_within_seconds=60.0)
                active_job = bus.get_active_job()
                bridge.telemetry_ready.emit(workers, active_job, None)
            except Exception as exc:
                bridge.telemetry_ready.emit(None, None, str(exc))
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
                    vram_val = w.get("vram_gb", 0)
                    try:
                        vram_str = f"{float(vram_val):.1f}"
                    except (ValueError, TypeError):
                        vram_str = str(vram_val)
                    rows.append([
                        str(w.get("worker_id", "-")),
                        str(w.get("hostname", "-")),
                        str(w.get("gpu_name", "-")),
                        vram_str,
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
        """Start or stop independent local cluster worker background processes for all detected GPUs."""
        if not hasattr(self, "_local_worker_procs"):
            self._local_worker_procs = {}

        # If any local worker is currently running, stop them all
        active_procs = {d: p for d, p in self._local_worker_procs.items() if p.poll() is None}
        if active_procs:
            self._log_cluster_event(f"Stopping {len(active_procs)} local worker process(es)...")
            for dev, proc in active_procs.items():
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)
                except Exception as exc:
                    self._log_cluster_event(f"Error terminating worker {dev}: {exc}")
            self._local_worker_procs.clear()
            if hasattr(self, "cluster_local_worker_btn"):
                self.cluster_local_worker_btn.setText("Start Local Worker(s)")
            self._log_cluster_event("Stopped all local worker process(es).")
            self.refresh_cluster_status()
            return

        if not hasattr(self, "cluster_shared_dir"):
            return
        path_str = self.cluster_shared_dir.text().strip()
        if not path_str:
            QMessageBox.warning(self, "Error", "Shared network directory not configured.")
            return

        try:
            shared_p = Path(path_str)
            shared_p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to access shared directory:\n{exc}")
            return

        from cluster.cluster_worker import detect_all_gpus, get_device_tag, get_running_worker_pid, get_worker_executable

        all_gpus = detect_all_gpus()
        # Launch worker for each detected GPU (or cpu if no discrete GPU)
        devices = [g["device"] for g in all_gpus] if all_gpus else ["cpu"]

        root_dir = Path(__file__).resolve().parent.parent.parent
        script_path = root_dir / "cluster_worker.py"
        if not script_path.exists():
            script_path = root_dir / "cluster" / "cluster_worker.py"

        worker_exe = get_worker_executable()
        exe_name = Path(worker_exe).name
        log_path = Path(tempfile.gettempdir()) / "cluster_worker_local.log"

        env = os.environ.copy()
        env["LLM_SHARED_DIR"] = path_str
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        started_count = 0
        for dev in devices:
            tag = get_device_tag(dev)
            existing_pid = get_running_worker_pid(tag)
            if existing_pid:
                self._log_cluster_event(f"Worker for device '{dev}' already running on host (PID: {existing_pid}).")
                continue

            try:
                log_file = open(log_path, "a", encoding="utf-8")
                proc = subprocess.Popen(
                    [worker_exe, str(script_path), "--shared-dir", path_str, "--device", dev],
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=flags,
                )
                self._local_worker_procs[dev] = proc
                started_count += 1
                gpu_info = next((g for g in all_gpus if g["device"] == dev), {})
                gpu_display = gpu_info.get("name", dev)
                self._log_cluster_event(
                    f"Started local worker on '{dev}' [{gpu_display}] (PID: {proc.pid}).\n"
                    f"    Task Manager name: '{exe_name}' (PID: {proc.pid}). Logs: {log_path}"
                )
            except Exception as exc:
                self._log_cluster_event(f"Failed to launch worker for {dev}: {exc}")

        if hasattr(self, "cluster_local_worker_btn"):
            active_now = len([p for p in self._local_worker_procs.values() if p.poll() is None])
            if active_now > 0:
                self.cluster_local_worker_btn.setText(f"Stop Local Worker(s) ({active_now} active)")
            else:
                self.cluster_local_worker_btn.setText("Start Local Worker(s)")

        QTimer.singleShot(1000, self.refresh_cluster_status)

    def restart_local_cluster_workers(self) -> None:
        """Cleanly terminate and re-launch local worker processes."""
        if not hasattr(self, "_local_worker_procs"):
            self._local_worker_procs = {}

        # 1. Terminate all tracked local workers
        for dev, proc in list(self._local_worker_procs.items()):
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.5)
                except Exception:
                    proc.kill()
        self._local_worker_procs.clear()

        # 2. Stop any remaining background worker processes for this host
        from cluster.cluster_worker import stop_running_worker
        stop_running_worker()

        self._log_cluster_event("Restarting local cluster worker processes...")
        time.sleep(0.5)

        # 3. Start fresh workers
        self.toggle_local_cluster_worker()

    def clean_offline_cluster_workers(self) -> None:
        """Purge dead and offline workers from the database and fleet table."""
        bus = self._get_cluster_bus()
        if not bus:
            return
        try:
            removed = bus.delete_offline_workers(stale_threshold_seconds=60.0)
            self._log_cluster_event(f"Cleaned {removed} offline / stale worker(s) from database.")
            self.refresh_cluster_status()
        except Exception as exc:
            self._log_cluster_event(f"Error cleaning offline workers: {exc}")

    def show_cluster_worker_context_menu(self, pos) -> None:
        """Display right-click context menu on discovered fleet table rows."""
        if not hasattr(self, "cluster_worker_table"):
            return
        row = self.cluster_worker_table.rowAt(pos.y())
        if row < 0:
            return
        worker_id_item = self.cluster_worker_table.item(row, 0)
        if not worker_id_item:
            return
        worker_id = worker_id_item.text().strip()

        menu = QMenu(self)
        stop_act = menu.addAction(f"Stop Worker '{worker_id}'")
        restart_act = menu.addAction(f"Restart Worker '{worker_id}'")
        menu.addSeparator()
        delete_act = menu.addAction(f"Remove '{worker_id}' from Fleet")

        viewport = self.cluster_worker_table.viewport()
        global_pos = viewport.mapToGlobal(pos) if viewport else pos
        selected_act = menu.exec(global_pos)

        if selected_act == stop_act:
            self.stop_cluster_worker(worker_id)
        elif selected_act == restart_act:
            self.restart_cluster_worker(worker_id)
        elif selected_act == delete_act:
            self.delete_cluster_worker(worker_id)

    def stop_cluster_worker(self, worker_id: str) -> None:
        """Send STOP command to a specific worker."""
        # Terminate if local
        if hasattr(self, "_local_worker_procs"):
            for dev, proc in list(self._local_worker_procs.items()):
                tag = dev.replace(":", "_")
                if tag in worker_id and proc.poll() is None:
                    proc.terminate()

        bus = self._get_cluster_bus()
        if bus:
            bus.set_worker_command(worker_id, "STOP")
            self._log_cluster_event(f"Sent STOP command to worker '{worker_id}'.")
            self.refresh_cluster_status()

    def restart_cluster_worker(self, worker_id: str) -> None:
        """Send RESTART command to a specific worker."""
        bus = self._get_cluster_bus()
        if bus:
            bus.set_worker_command(worker_id, "RESTART")
            self._log_cluster_event(f"Sent RESTART command to worker '{worker_id}'.")
            self.refresh_cluster_status()

    def delete_cluster_worker(self, worker_id: str) -> None:
        """Delete a worker row from the database."""
        bus = self._get_cluster_bus()
        if bus:
            bus.delete_worker(worker_id)
            self._log_cluster_event(f"Removed worker '{worker_id}' from cluster database.")
            self.refresh_cluster_status()
