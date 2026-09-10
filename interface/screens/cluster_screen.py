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

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox

from cluster.bus import ClusterStorageBus
from interface.tabs.cluster_tab import set_cluster_table_rows


class ClusterTelemetryBridge(QObject):
    """Qt signal bridge for thread-safe worker telemetry updates."""

    telemetry_ready = Signal(object, object, object)  # workers, active_job, error


class ClusterCoordinatorThread(QThread):
    """Background worker thread executing the ClusterCoordinator round-averaging loop."""

    round_telemetry_ready = Signal(dict)
    job_finished = Signal(str, bool)

    def __init__(self, bus: ClusterStorageBus, job_id: str, poll_interval: float = 1.0, parent=None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.job_id = job_id
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        from cluster.coordinator import ClusterCoordinator
        coordinator = ClusterCoordinator(self.bus, self.job_id)

        # Signal coordinator liveness immediately on start
        if hasattr(self.bus, "touch_job"):
            self.bus.touch_job(self.job_id)

        def _on_round_progress(metrics: dict[str, Any]) -> None:
            if hasattr(self.bus, "touch_job"):
                self.bus.touch_job(self.job_id)
            self.round_telemetry_ready.emit(metrics)

        success = coordinator.run_job(
            poll_interval_seconds=self.poll_interval,
            telemetry_callback=_on_round_progress,
            stop_event=self._stop_event,
        )
        self.job_finished.emit(self.job_id, success)

    def stop(self) -> None:
        self._stop_event.set()


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert Paths and non-primitive types to JSON-serializable structures."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


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
        # Check local worker process liveness (including detached daemons across restarts)
        from cluster.cluster_worker import get_all_running_worker_pids
        running_pids = get_all_running_worker_pids()
        if hasattr(self, "_local_worker_procs") and self._local_worker_procs:
            alive = {}
            for dev, proc in list(self._local_worker_procs.items()):
                if proc.poll() is None:
                    alive[dev] = proc
            self._local_worker_procs = alive

        total_active_local = max(len(getattr(self, "_local_worker_procs", {})), len(running_pids))
        if hasattr(self, "cluster_local_worker_btn"):
            if total_active_local > 0:
                self.cluster_local_worker_btn.setText(f"Stop Local Worker(s) ({total_active_local} active)")
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
                    # VRAM Usage (e.g. "80% (3.2 / 4.0 GB)")
                    vram_val = float(w.get("vram_gb") or 0.0)
                    vram_used = float(w.get("vram_used_gb") or 0.0)
                    if vram_val > 0:
                        vram_pct = int(round((vram_used / vram_val) * 100))
                        vram_str = f"{vram_pct}% ({vram_used:.1f} / {vram_val:.1f} GB)"
                    elif vram_used > 0:
                        vram_str = f"{vram_used:.1f} GB"
                    else:
                        vram_str = f"{vram_val:.1f} GB"

                    # RAM Usage (e.g. "45% (7.2 / 16.0 GB)")
                    ram_total = float(w.get("ram_total_gb") or 0.0)
                    ram_used = float(w.get("ram_used_gb") or 0.0)
                    if ram_total > 0:
                        ram_pct = int(round((ram_used / ram_total) * 100))
                        ram_str = f"{ram_pct}% ({ram_used:.1f} / {ram_total:.1f} GB)"
                    else:
                        ram_str = "-"

                    # CPU Usage (e.g. "18%")
                    cpu_pct = float(w.get("cpu_percent") or 0.0)
                    cpu_str = f"{cpu_pct:.0f}%" if cpu_pct > 0 else "-"

                    rows.append([
                        str(w.get("worker_id", "-")),
                        str(w.get("hostname", "-")),
                        str(w.get("gpu_name", "-")),
                        vram_str,
                        ram_str,
                        cpu_str,
                        str(w.get("status", "OFFLINE")),
                        last_hb_str,
                    ])
                set_cluster_table_rows(self.cluster_worker_table, rows)

        if active_job:
            jid = active_job["job_id"]
            st = str(active_job.get("status", "")).upper()
            cur_round = active_job.get("current_round", 0)
            max_rounds = active_job.get("max_rounds", 10)
            if hasattr(self, "cluster_status_label"):
                self.cluster_status_label.setText(f"Status: Job {jid} ({st})")
            if hasattr(self, "cluster_round_label"):
                self.cluster_round_label.setText(f"Round: {cur_round} / {max_rounds}")

            if st == "RUNNING":
                if hasattr(self, "train_status"):
                    self.train_status.setText(f"Training: Cluster (Local SGD) - Round {cur_round}/{max_rounds}")
                if hasattr(self, "project_state"):
                    self.project_state.setText("Training")
                if hasattr(self, "stop_training_button"):
                    self.stop_training_button.setEnabled(True)
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
            # Ensure dataset is accessible to all cluster workers on shared storage
            try:
                is_under_shared = Path(dataset_path).resolve().is_relative_to(bus.shared_dir.resolve())
            except Exception:
                is_under_shared = False

            if not is_under_shared:
                shared_candidate = bus.shared_dir / Path(dataset_path).name
                if not shared_candidate.exists() or shared_candidate.stat().st_size != Path(dataset_path).stat().st_size:
                    self._log_cluster_event(f"Copying dataset to shared storage for cluster workers: {shared_candidate.name}...")
                    import shutil
                    shutil.copyfile(dataset_path, shared_candidate)
                # Also copy tokenizer/summary metadata if available in dataset folder
                orig_dir = Path(dataset_path).parent
                for meta_file in ["dataset_summary.json", "tokenizer.json"]:
                    src_meta = orig_dir / meta_file
                    dst_meta = bus.shared_dir / meta_file
                    if src_meta.exists() and (not dst_meta.exists() or dst_meta.stat().st_size != src_meta.stat().st_size):
                        try:
                            import shutil
                            shutil.copyfile(src_meta, dst_meta)
                        except Exception:
                            pass
                dataset_path = str(shared_candidate)

            # Determine vocabulary size accurately from dataset / tokenizer metadata
            vocab_size = 0
            candidate_dirs = [
                Path(dataset_path).parent,
                bus.shared_dir,
                Path(self.train_data_dir.text().strip()) if hasattr(self, "train_data_dir") and self.train_data_dir.text().strip() else None,
                Path(self.output_dir_edit.text().strip()) if hasattr(self, "output_dir_edit") and self.output_dir_edit.text().strip() else None,
            ]
            for c_dir in candidate_dirs:
                if c_dir and c_dir.exists() and hasattr(self, "_current_training_vocab_size"):
                    v = self._current_training_vocab_size(c_dir)
                    if v > 0:
                        vocab_size = v
                        break

            # Fallback: inspect token array upper bound
            if os.path.exists(dataset_path):
                try:
                    import numpy as np
                    tok_arr = np.load(dataset_path, mmap_mode="r")
                    sample_slice = tok_arr[:min(len(tok_arr), 100000)]
                    if len(sample_slice) > 0:
                        max_in_arr = int(np.max(sample_slice))
                        vocab_size = max(vocab_size, max_in_arr + 1)
                except Exception:
                    pass

            if vocab_size <= 0:
                vocab_size = 1000

            # Build configs and sanitize non-primitive types (WindowsPath, etc.)
            if hasattr(self, "_current_model_config"):
                model_cfg = dataclasses.asdict(self._current_model_config(vocab_size=vocab_size))
            else:
                model_cfg = {
                    "vocab_size": vocab_size,
                    "context_length": 512,
                    "embedding_size": 256,
                    "head_count": 4,
                    "layer_count": 4,
                }
            model_cfg["vocab_size"] = max(int(model_cfg.get("vocab_size", 0) or 0), vocab_size)

            training_cfg = dataclasses.asdict(self._current_training_config()) if hasattr(self, "_current_training_config") else {
                "learning_rate": 3e-4,
                "batch_size": 4,
            }

            model_cfg = _sanitize_for_json(model_cfg)
            training_cfg = _sanitize_for_json(training_cfg)

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
            self._start_cluster_coordinator(bus, job_id)

            # Auto-launch local worker(s) if none are running on this host
            from cluster.cluster_worker import get_all_running_worker_pids
            if not get_all_running_worker_pids() and hasattr(self, "start_local_cluster_workers"):
                self.start_local_cluster_workers()

            if hasattr(self, "train_button"):
                self.train_button.setEnabled(False)
                self.train_button.setText("Training...")
            if hasattr(self, "fine_tune_button"):
                self.fine_tune_button.setEnabled(False)
                self.fine_tune_button.setText("Fine-Tuning...")
            if hasattr(self, "stop_training_button"):
                self.stop_training_button.setEnabled(True)
                self.stop_training_button.setText("Stop")
            if hasattr(self, "project_state"):
                self.project_state.setText("Training")
            if hasattr(self, "train_status"):
                self.train_status.setText(f"Training: Cluster (Local SGD) - Round 0/{max_rounds}")

            self.refresh_cluster_status()

        except Exception as exc:
            QMessageBox.critical(self, "Launch Error", f"Failed to submit cluster job:\n{exc}")
            self._log_cluster_event(f"Launch failed: {exc}")

    def _start_cluster_coordinator(self, bus: ClusterStorageBus, job_id: str) -> None:
        """Launch detached background coordinator daemon to handle round synchronization and telemetry."""
        from cluster.cluster_worker import get_worker_executable

        worker_exe = get_worker_executable()
        shared_path_str = str(bus.shared_dir)
        coord_log_path = bus.jobs_dir / job_id / "coordinator.log"
        coord_log_path.parent.mkdir(parents=True, exist_ok=True)

        flags = 0
        if sys.platform == "win32":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            CREATE_NO_WINDOW = 0x08000000
            flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW

        try:
            log_file = open(coord_log_path, "a", encoding="utf-8")
            root_dir = Path(__file__).resolve().parent.parent.parent
            coord_script = root_dir / "cluster_coordinator.py"
            if coord_script.exists():
                cmd = [worker_exe, str(coord_script), "--shared-dir", shared_path_str, "--job-id", job_id]
            else:
                cmd = [worker_exe, "-m", "cluster.coordinator", "--shared-dir", shared_path_str, "--job-id", job_id]
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                close_fds=True,
            )
            self._coordinator_proc = proc
            self._log_cluster_event(f"Spawned detached coordinator daemon for job {job_id} (PID: {proc.pid}).")
        except Exception as exc:
            self._log_cluster_event(f"Failed to spawn detached coordinator daemon: {exc}. Falling back to in-process thread.")
            if hasattr(self, "_coordinator_thread") and self._coordinator_thread and self._coordinator_thread.isRunning():
                self._coordinator_thread.stop()
                self._coordinator_thread.wait(2000)

            self._coordinator_thread = ClusterCoordinatorThread(bus, job_id, poll_interval=1.0)
            self._coordinator_thread.round_telemetry_ready.connect(self._on_cluster_round_telemetry)
            self._coordinator_thread.job_finished.connect(self._on_cluster_job_finished)
            self._coordinator_thread.start()

    def _on_cluster_round_telemetry(self, telemetry: dict[str, Any]) -> None:
        """Receive round telemetry from the coordinator thread and update charts and UI chips."""
        if telemetry.get("type") == "round_waiting":
            round_num = int(telemetry.get("round", 0))
            ready_w = int(telemetry.get("ready_workers", 0))
            total_p = int(telemetry.get("total_participants", 1))
            elapsed_s = float(telemetry.get("elapsed_seconds", 0.0))
            if hasattr(self, "cluster_status_label"):
                self.cluster_status_label.setText(
                    f"Status: Waiting for round {round_num + 1} weights ({ready_w}/{total_p} ready, {elapsed_s:.0f}s)"
                )
            return

        round_num = int(telemetry.get("round", 0))
        cur_round = round_num + 1
        max_rounds = int(telemetry.get("max_rounds", 10))
        effective_step = int(telemetry.get("effective_step", 0))
        global_loss = float(telemetry.get("global_loss", 0.0))
        speed = float(telemetry.get("aggregate_tokens_per_sec", 0.0))
        workers_count = int(telemetry.get("ready_workers_count", 0))
        worker_losses = telemetry.get("worker_losses", {})

        # 1. Update Training Tab Charts & Metrics
        if hasattr(self, "loss_chart") and self.loss_chart:
            self.loss_chart.add_metrics(effective_step, global_loss, None)

        if hasattr(self, "throughput_chart") and self.throughput_chart:
            self.throughput_chart.add_values(effective_step, speed)

        if hasattr(self, "training_loss_metric"):
            self.training_loss_metric.setText(f"Train loss: {global_loss:.4f}")
        if hasattr(self, "training_speed_metric"):
            self.training_speed_metric.setText(f"Speed: {speed:,.0f} tok/s")
        if hasattr(self, "training_step_metric"):
            self.training_step_metric.setText(f"Step: {effective_step} (Round {cur_round}/{max_rounds})")
        if hasattr(self, "training_progress"):
            self.training_progress.setValue(int((cur_round / max(max_rounds, 1)) * 100))

        # 2. Update Live Tab Metrics
        if hasattr(self, "live_loss_metric"):
            self.live_loss_metric.setText(f"Loss: {global_loss:.4f}")
        if hasattr(self, "live_tokens_metric"):
            self.live_tokens_metric.setText(f"Tokens/sec: {speed:,.0f}")
        if hasattr(self, "live_step_metric"):
            self.live_step_metric.setText(f"Step: {effective_step}")
        if hasattr(self, "live_progress"):
            self.live_progress.setValue(int((cur_round / max(max_rounds, 1)) * 100))

        # 3. Update Cluster Tab
        if hasattr(self, "cluster_status_label"):
            self.cluster_status_label.setText(f"Status: Round {cur_round}/{max_rounds} (Loss: {global_loss:.4f})")
        if hasattr(self, "cluster_round_label"):
            self.cluster_round_label.setText(f"Round: {cur_round} / {max_rounds}")

        worker_breakdown = ", ".join(f"{w}: {l:.4f}" for w, l in worker_losses.items())
        msg = f"Round {cur_round}/{max_rounds} complete | Loss: {global_loss:.4f} | Aggregate Speed: {speed:,.0f} tok/s | Workers: {workers_count}"
        if worker_breakdown:
            msg += f" ({worker_breakdown})"
        self._log_cluster_event(msg)

        if hasattr(self, "training_log"):
            self.training_log.append(f"[Local SGD] {msg}")

        self.refresh_cluster_status()

    def _on_cluster_job_finished(self, job_id: str, success: bool) -> None:
        """Handle coordinator completion signal."""
        if success:
            self._log_cluster_event(f"Cluster job {job_id} successfully completed all rounds.")
            if hasattr(self, "cluster_status_label"):
                self.cluster_status_label.setText("Status: Completed")
            if hasattr(self, "project_state"):
                self.project_state.setText("Completed")
            if hasattr(self, "train_status"):
                self.train_status.setText("Training: completed")
        else:
            self._log_cluster_event(f"Cluster job {job_id} stopped or cancelled.")
            if hasattr(self, "project_state"):
                self.project_state.setText("Idle")
            if hasattr(self, "train_status"):
                self.train_status.setText("Training: idle")

        if hasattr(self, "stop_training_button"):
            self.stop_training_button.setEnabled(False)
        if hasattr(self, "train_button"):
            self.train_button.setEnabled(True)
            self.train_button.setText("Start Training")
        if hasattr(self, "fine_tune_button"):
            self.fine_tune_button.setEnabled(True)
            self.fine_tune_button.setText("Start Fine-Tune")

        self.refresh_cluster_status()

    def pause_cluster_job(self) -> None:
        """Cooperatively pause active cluster job."""
        bus = self._get_cluster_bus()
        if not bus:
            return
        active = bus.get_active_job()
        if active:
            bus.set_job_status(active["job_id"], "PAUSED")
            self._log_cluster_event(f"Signal PAUSE set for job {active['job_id']}.")
            if hasattr(self, "train_button"):
                self.train_button.setEnabled(True)
                self.train_button.setText("Resume Training")
            self.refresh_cluster_status()

    def _is_coordinator_running(self, bus: ClusterStorageBus, job_id: str) -> bool:
        """Check if coordinator process is currently running for this job."""
        if hasattr(self, "_coordinator_proc") and self._coordinator_proc:
            if self._coordinator_proc.poll() is None:
                return True
        pid_file = bus.jobs_dir / job_id / "coordinator.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
                import psutil
                if psutil.pid_exists(pid):
                    p = psutil.Process(pid)
                    if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                        return True
            except Exception:
                pass
        return False

    def resume_cluster_job(self, job_id: Optional[str] = None) -> None:
        """Resume execution of a cluster job (selected, active, or specified by job_id)."""
        bus = self._get_cluster_bus()
        if not bus:
            return

        target_job = None
        # 1. Check explicitly passed job_id
        if job_id:
            target_job = bus.get_job(job_id)

        # 2. Check user-selected job from table
        if not target_job and getattr(self, "_selected_cluster_job_id", None):
            target_job = bus.get_job(self._selected_cluster_job_id)

        # 3. Check active job in database
        if not target_job:
            target_job = bus.get_active_job()

        # 4. Check latest incomplete job in database
        if not target_job:
            for j in bus.list_all_jobs(limit=10):
                if int(j.get("current_round", 0)) < int(j.get("max_rounds", 10)) and j.get("status") not in {"COMPLETED"}:
                    target_job = j
                    break

        if not target_job:
            QMessageBox.information(
                self,
                "Resume Job",
                "No incomplete cluster job found to resume.\n\n"
                "Please select a job in the table or click 'Launch New Job' to start a new job.",
            )
            return

        jid = target_job["job_id"]
        cur_round = int(target_job.get("current_round", 0))
        max_rounds = int(target_job.get("max_rounds", 10))

        if cur_round >= max_rounds or target_job.get("status") == "COMPLETED":
            QMessageBox.information(
                self,
                "Job Completed",
                f"Job '{jid}' has already completed all {max_rounds} rounds.\n\n"
                "To run again, click 'Re-queue' to reset it or 'Launch New Job'.",
            )
            return

        # Mark job as RUNNING in SQLite and clean up pause.sig & stop.sig
        bus.set_job_status(jid, "RUNNING")
        self._log_cluster_event(f"Resumed cluster job {jid} (Round {cur_round}/{max_rounds}).")

        # Spawn coordinator daemon if not already running
        if not self._is_coordinator_running(bus, jid):
            self._start_cluster_coordinator(bus, jid)
        else:
            self._log_cluster_event(f"Coordinator daemon for {jid} is already active.")

        # Auto-launch local worker(s) if none are running on this host
        from cluster.cluster_worker import get_all_running_worker_pids
        if not get_all_running_worker_pids() and hasattr(self, "start_local_cluster_workers"):
            self.start_local_cluster_workers()

        # Synchronize Training Tab status and buttons
        if hasattr(self, "train_button"):
            self.train_button.setEnabled(False)
            self.train_button.setText("Training...")
        if hasattr(self, "fine_tune_button"):
            self.fine_tune_button.setEnabled(False)
            self.fine_tune_button.setText("Fine-Tuning...")
        if hasattr(self, "stop_training_button"):
            self.stop_training_button.setEnabled(True)
            self.stop_training_button.setText("Stop")
        if hasattr(self, "project_state"):
            self.project_state.setText("Training")
        if hasattr(self, "train_status"):
            self.train_status.setText(f"Training: Cluster (Local SGD) - Round {cur_round}/{max_rounds}")
        if hasattr(self, "cluster_status_label"):
            self.cluster_status_label.setText(f"Status: Running ({jid})")

        self.refresh_cluster_status()
        if hasattr(self, "refresh_job_manager_tab"):
            self.refresh_job_manager_tab()

    def stop_cluster_job(self) -> None:
        """Stop active or selected cluster job and terminate coordinator daemon."""
        if hasattr(self, "_coordinator_thread") and self._coordinator_thread:
            self._coordinator_thread.stop()
        if hasattr(self, "_coordinator_proc") and self._coordinator_proc:
            try:
                if self._coordinator_proc.poll() is None:
                    self._coordinator_proc.terminate()
            except Exception:
                pass
        bus = self._get_cluster_bus()
        if not bus:
            return
        target_job_id = None
        active = bus.get_active_job()
        if active:
            target_job_id = active["job_id"]
        elif getattr(self, "_selected_cluster_job_id", None):
            target_job_id = self._selected_cluster_job_id

        if target_job_id:
            bus.set_job_status(target_job_id, "STOPPED")
            self._log_cluster_event(f"Signal STOP set for job {target_job_id}.")
            pid_file = bus.jobs_dir / target_job_id / "coordinator.pid"
            if pid_file.exists():
                try:
                    cpid = int(pid_file.read_text(encoding="utf-8").strip())
                    import psutil
                    if psutil.pid_exists(cpid):
                        psutil.Process(cpid).terminate()
                    pid_file.unlink(missing_ok=True)
                except Exception:
                    pass
            self.refresh_cluster_status()
            if hasattr(self, "refresh_job_manager_tab"):
                self.refresh_job_manager_tab()
        if hasattr(self, "stop_training_button"):
            self.stop_training_button.setEnabled(False)
        if hasattr(self, "train_button"):
            self.train_button.setEnabled(True)
            self.train_button.setText("Start Training")
        if hasattr(self, "fine_tune_button"):
            self.fine_tune_button.setEnabled(True)
            self.fine_tune_button.setText("Start Fine-Tune")
        if hasattr(self, "project_state"):
            self.project_state.setText("Stopped")
        if hasattr(self, "train_status"):
            self.train_status.setText("Training: idle")

    def stop_local_cluster_workers(self) -> None:
        """Stop all local worker processes running on this machine."""
        if not hasattr(self, "_local_worker_procs"):
            self._local_worker_procs = {}

        from cluster.cluster_worker import (
            get_all_running_worker_pids,
            stop_running_worker,
        )

        running_pids = get_all_running_worker_pids()
        active_procs = {d: p for d, p in self._local_worker_procs.items() if p.poll() is None}

        total = max(len(running_pids), len(active_procs))
        if total > 0:
            self._log_cluster_event(f"Stopping {total} local worker process(es)...")

        bus = self._get_cluster_bus()
        import socket
        hostname = socket.gethostname()

        for dev, proc in active_procs.items():
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as exc:
                self._log_cluster_event(f"Error terminating worker {dev}: {exc}")

        # Stop any background worker processes and clear locks
        stop_running_worker()

        if bus:
            for tag in running_pids:
                wid = f"{hostname}_{tag}"
                bus.heartbeat(wid, status="OFFLINE", current_job_id=None)

        self._local_worker_procs.clear()
        if hasattr(self, "cluster_local_worker_btn"):
            self.cluster_local_worker_btn.setText("Start Local Worker(s)")
        self._log_cluster_event("Stopped all local worker process(es).")
        self.refresh_cluster_status()
        if hasattr(self, "refresh_job_manager_tab"):
            self.refresh_job_manager_tab()

    def start_local_cluster_workers(self) -> None:
        """Launch background worker processes on this workstation for all detected GPUs."""
        if not hasattr(self, "_local_worker_procs"):
            self._local_worker_procs = {}

        from cluster.cluster_worker import (
            detect_all_gpus,
            get_all_running_worker_pids,
            get_device_tag,
            get_running_worker_pid,
            get_worker_executable,
        )

        if not hasattr(self, "cluster_shared_dir"):
            return
        path_str = self.cluster_shared_dir.text().strip()
        if not path_str:
            bus = self._get_cluster_bus()
            if bus:
                path_str = str(bus.shared_dir)
        if not path_str:
            QMessageBox.warning(self, "Error", "Shared network directory not configured.")
            return

        try:
            shared_p = Path(path_str)
            shared_p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Failed to access shared directory:\n{exc}")
            return

        all_gpus = detect_all_gpus()
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

        flags = 0
        if sys.platform == "win32":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            CREATE_NO_WINDOW = 0x08000000
            flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW

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
                    close_fds=True,
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
            active_now = max(
                len([p for p in self._local_worker_procs.values() if p.poll() is None]),
                len(get_all_running_worker_pids()),
            )
            if active_now > 0:
                self.cluster_local_worker_btn.setText(f"Stop Local Worker(s) ({active_now} active)")
            else:
                self.cluster_local_worker_btn.setText("Start Local Worker(s)")

        QTimer.singleShot(1000, self.refresh_cluster_status)
        if hasattr(self, "refresh_job_manager_tab"):
            QTimer.singleShot(1000, self.refresh_job_manager_tab)

    def toggle_local_cluster_worker(self) -> None:
        """Start or stop independent local cluster worker background processes for all detected GPUs."""
        from cluster.cluster_worker import get_all_running_worker_pids
        running_pids = get_all_running_worker_pids()
        active_procs = {d: p for d, p in getattr(self, "_local_worker_procs", {}).items() if p.poll() is None}

        if running_pids or active_procs:
            self.stop_local_cluster_workers()
        else:
            self.start_local_cluster_workers()

    def restart_local_cluster_workers(self) -> None:
        """Cleanly terminate and re-launch local worker processes."""
        from cluster.cluster_worker import get_all_running_worker_pids
        self._log_cluster_event("Restarting local cluster worker processes...")
        # Clear any pending commands for this host in the cluster database
        bus = self._get_cluster_bus()
        if bus:
            import socket
            host = socket.gethostname().lower()
            try:
                for w in bus.list_workers():
                    wid = str(w.get("worker_id", ""))
                    if host in wid.lower():
                        bus.set_worker_command(wid, None)
            except Exception:
                pass
        self.stop_local_cluster_workers()
        # Wait up to 2 seconds for previous worker processes to exit and release lock files
        for _ in range(10):
            if not get_all_running_worker_pids():
                break
            time.sleep(0.2)
        self.start_local_cluster_workers()
        self._log_cluster_event("Restarted local cluster worker processes.")

    def clean_offline_cluster_workers(self) -> None:
        """Purge dead and offline workers from the database and fleet table."""
        bus = self._get_cluster_bus()
        if not bus:
            return
        try:
            removed = bus.delete_offline_workers(stale_threshold_seconds=60.0)
            self._log_cluster_event(f"Cleaned {removed} offline / stale worker(s) from database.")
            self.refresh_cluster_status()
            if hasattr(self, "refresh_job_manager_tab"):
                self.refresh_job_manager_tab()
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
        view_logs_act = menu.addAction(f"View Logs for '{worker_id}'")
        menu.addSeparator()
        stop_act = menu.addAction(f"Stop Worker '{worker_id}'")
        restart_act = menu.addAction(f"Restart Worker '{worker_id}'")
        menu.addSeparator()
        delete_act = menu.addAction(f"Remove '{worker_id}' from Fleet")

        viewport = self.cluster_worker_table.viewport()
        global_pos = viewport.mapToGlobal(pos) if viewport else pos
        selected_act = menu.exec(global_pos)

        if selected_act == view_logs_act:
            self._load_worker_logs_for(worker_id)
        elif selected_act == stop_act:
            self.stop_cluster_worker(worker_id)
        elif selected_act == restart_act:
            self.restart_cluster_worker(worker_id)
        elif selected_act == delete_act:
            self.delete_cluster_worker(worker_id)

    def on_cluster_worker_selected(self) -> None:
        """Handle worker table selection to load and display diagnostic logs."""
        if not hasattr(self, "cluster_worker_table") or not hasattr(self, "cluster_worker_log"):
            return
        selected_rows = self.cluster_worker_table.selectedItems()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        worker_id_item = self.cluster_worker_table.item(row, 0)
        if not worker_id_item:
            return
        worker_id = worker_id_item.text().strip()
        self._load_worker_logs_for(worker_id)

    def refresh_selected_worker_logs(self) -> None:
        """Refresh logs for the currently selected worker node."""
        if hasattr(self, "cluster_worker_table") and hasattr(self, "cluster_worker_log"):
            self.on_cluster_worker_selected()

    def _load_worker_logs_for(self, worker_id: str) -> None:
        """Query SQLite database for worker logs and render into cluster_worker_log text widget."""
        bus = self._get_cluster_bus()
        if not bus:
            if hasattr(self, "cluster_worker_log"):
                self.cluster_worker_log.setPlainText("Shared storage not accessible.")
            return

        try:
            logs = bus.get_worker_logs(worker_id, limit=250)
        except Exception as exc:
            if hasattr(self, "cluster_worker_log"):
                self.cluster_worker_log.setPlainText(f"Failed to query worker logs: {exc}")
            return

        if hasattr(self, "cluster_selected_worker_label"):
            self.cluster_selected_worker_label.setText(f"<b>WORKER DIAGNOSTIC LOGS: {worker_id}</b> ({len(logs)} entries)")

        if hasattr(self, "cluster_worker_log"):
            if not logs:
                self.cluster_worker_log.setPlainText(f"No diagnostic logs recorded yet for worker '{worker_id}'.")
                return

            formatted = []
            for item in logs:
                t_str = time.strftime("%H:%M:%S", time.localtime(item.get("timestamp", 0)))
                lvl = item.get("level", "INFO")
                msg = item.get("message", "")
                formatted.append(f"[{t_str}] [{lvl}] {msg}")

            self.cluster_worker_log.setPlainText("\n".join(formatted))
            scrollbar = self.cluster_worker_log.verticalScrollBar()
            if scrollbar:
                scrollbar.setValue(scrollbar.maximum())

    def stop_cluster_worker(self, worker_id: str) -> None:
        """Send STOP command to a specific worker and immediately mark it OFFLINE."""
        import socket
        hostname = socket.gethostname().lower()
        is_local = hostname in worker_id.lower()

        if is_local:
            parts = worker_id.split("_", 1)
            tag = parts[1] if len(parts) > 1 else "default"
            dev = tag.replace("_", ":")

            if hasattr(self, "_local_worker_procs") and dev in self._local_worker_procs:
                proc = self._local_worker_procs.pop(dev)
                if proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=1.0)
                    except Exception:
                        proc.kill()

            from cluster.cluster_worker import get_lock_file, get_running_worker_pid
            old_pid = get_running_worker_pid(tag)
            if old_pid:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/PID", str(old_pid)], check=False, capture_output=True)
                else:
                    try:
                        import signal
                        os.kill(old_pid, signal.SIGTERM)
                    except Exception:
                        pass
                get_lock_file(tag).unlink(missing_ok=True)

        bus = self._get_cluster_bus()
        if bus:
            bus.set_worker_command(worker_id, "STOP")
            bus.heartbeat(worker_id, status="OFFLINE", current_job_id=None)
            self._log_cluster_event(f"Sent STOP command to worker '{worker_id}' (marked OFFLINE).")
            self.refresh_cluster_status()
            if hasattr(self, "refresh_job_manager_tab"):
                self.refresh_job_manager_tab()

    def restart_cluster_worker(self, worker_id: str) -> None:
        """Restart a specific worker (local process respawn or remote RESTART command)."""
        import socket
        hostname = socket.gethostname().lower()
        is_local = hostname in worker_id.lower()

        from cluster.cluster_worker import (
            get_lock_file,
            get_running_worker_pid,
            get_worker_executable,
        )

        bus = self._get_cluster_bus()

        if is_local:
            if "_cuda_" in worker_id:
                tag = "cuda_" + worker_id.rsplit("_cuda_", 1)[1]
                dev = tag.replace("_", ":")
            elif worker_id.endswith("_cpu"):
                tag = "cpu"
                dev = "cpu"
            else:
                parts = worker_id.rsplit("_", 1)
                tag = parts[1] if len(parts) > 1 else "default"
                dev = tag.replace("_", ":")

            self._log_cluster_event(f"Restarting local worker '{worker_id}' (device: {dev})...")

            # 1. Terminate tracked proc if present
            if hasattr(self, "_local_worker_procs") and dev in self._local_worker_procs:
                proc = self._local_worker_procs.pop(dev)
                if proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=1.5)
                    except Exception:
                        proc.kill()

            # 2. Terminate running PID from lockfile if running
            old_pid = get_running_worker_pid(tag)
            if old_pid:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/PID", str(old_pid)], check=False, capture_output=True)
                else:
                    try:
                        import signal
                        os.kill(old_pid, signal.SIGTERM)
                    except Exception:
                        pass
                time.sleep(0.3)
                get_lock_file(tag).unlink(missing_ok=True)

            if bus:
                bus.heartbeat(worker_id, status="OFFLINE", current_job_id=None)
                bus.set_worker_command(worker_id, None)

            # 3. Launch fresh worker for this device
            path_str = self.cluster_shared_dir.text().strip() if hasattr(self, "cluster_shared_dir") else ""
            if not path_str and bus:
                path_str = str(bus.shared_dir)

            root_dir = Path(__file__).resolve().parent.parent.parent
            script_path = root_dir / "cluster_worker.py"
            if not script_path.exists():
                script_path = root_dir / "cluster" / "cluster_worker.py"

            worker_exe = get_worker_executable()
            log_path = Path(tempfile.gettempdir()) / "cluster_worker_local.log"
            env = os.environ.copy()
            env["LLM_SHARED_DIR"] = path_str

            flags = 0
            if sys.platform == "win32":
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                CREATE_NO_WINDOW = 0x08000000
                flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW

            try:
                log_file = open(log_path, "a", encoding="utf-8")
                proc = subprocess.Popen(
                    [worker_exe, str(script_path), "--shared-dir", path_str, "--device", dev],
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=flags,
                    close_fds=True,
                )
                if not hasattr(self, "_local_worker_procs"):
                    self._local_worker_procs = {}
                self._local_worker_procs[dev] = proc
                self._log_cluster_event(f"Successfully restarted local worker '{worker_id}' (PID: {proc.pid}).")
            except Exception as exc:
                self._log_cluster_event(f"Failed to restart local worker '{worker_id}': {exc}")
        else:
            if bus:
                bus.set_worker_command(worker_id, "RESTART")
                self._log_cluster_event(f"Sent RESTART command to remote worker '{worker_id}'.")

        self.refresh_cluster_status()
        if hasattr(self, "refresh_job_manager_tab"):
            self.refresh_job_manager_tab()

    def delete_cluster_worker(self, worker_id: str) -> None:
        """Delete a worker row from the database and fleet table."""
        import socket
        hostname = socket.gethostname().lower()
        if hostname in worker_id.lower():
            self.stop_cluster_worker(worker_id)

        bus = self._get_cluster_bus()
        if bus:
            bus.delete_worker(worker_id)
            self._log_cluster_event(f"Removed worker '{worker_id}' from cluster database.")
            self.refresh_cluster_status()
            if hasattr(self, "refresh_job_manager_tab"):
                self.refresh_job_manager_tab()
