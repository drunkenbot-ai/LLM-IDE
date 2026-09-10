"""Screen mixin for Thinkbox Deadline & Fox Renderfarm inspired Cluster Job Monitor."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, QPoint, Qt, Signal
from PySide6.QtWidgets import QMenu, QMessageBox, QTextEdit

from cluster.cluster_worker import get_all_running_worker_pids
from interface.tabs.job_manager_tab import set_table_rows


class JobMonitorBridge(QObject):
    """Qt signal bridge for thread-safe, non-blocking cluster job monitor telemetry."""

    data_ready = Signal(dict)


class JobManagerScreenMixin:
    """Screen mixin providing master control and telemetry inspection for cluster jobs."""

    _selected_cluster_job_id: Optional[str] = None
    _job_monitor_bridge: Optional[JobMonitorBridge] = None
    _job_monitor_poll_in_progress: bool = False

    def _ensure_job_monitor_bridge(self) -> JobMonitorBridge:
        """Lazily initialize the cross-thread job monitor signal bridge."""
        if self._job_monitor_bridge is None:
            self._job_monitor_bridge = JobMonitorBridge()
            self._job_monitor_bridge.data_ready.connect(self._apply_job_monitor_data)
        return self._job_monitor_bridge

    def refresh_job_manager_tab(self) -> None:
        """Refresh the entire Cluster Job Monitor in a background thread to prevent UI freezing."""
        if not hasattr(self, "cluster_jobs_table"):
            return

        # Always refresh cluster fleet hardware telemetry in the background
        if hasattr(self, "refresh_cluster_status"):
            self.refresh_cluster_status()

        bus = self._get_cluster_bus() if hasattr(self, "_get_cluster_bus") else None
        if not bus:
            if hasattr(self, "cluster_status_label"):
                self.cluster_status_label.setText("Status: Configure shared storage directory")
            return

        if self._job_monitor_poll_in_progress:
            return
        self._job_monitor_poll_in_progress = True

        bridge = self._ensure_job_monitor_bridge()
        target_selection = self._selected_cluster_job_id

        def _bg_poll() -> None:
            try:
                jobs = bus.list_all_jobs(limit=100)
                latest_rounds_map = bus.get_latest_rounds_for_all_jobs()
                status_symbols = {
                    "RUNNING": "▶ RUNNING",
                    "PAUSED": "⏸ PAUSED",
                    "STOPPED": "■ STOPPED",
                    "COMPLETED": "✔ COMPLETED",
                    "FAILED": "✖ FAILED",
                    "QUEUED": "⏳ QUEUED",
                }

                job_rows = []
                active_job = None
                active_job_id = None

                for j in jobs:
                    jid = str(j.get("job_id", ""))
                    st = str(j.get("status", "QUEUED")).upper()
                    if st in {"RUNNING", "QUEUED", "PAUSED"} and not active_job_id:
                        active_job_id = jid
                        active_job = j

                    st_display = status_symbols.get(st, st)

                    # Model / Architecture description
                    m_cfg = j.get("model_config", {})
                    d_model = m_cfg.get("embedding_size") or m_cfg.get("hidden_dim", "-")
                    layers = m_cfg.get("layer_count") or m_cfg.get("num_layers", "-")
                    heads = m_cfg.get("head_count") or m_cfg.get("num_heads", "-")
                    ctx = m_cfg.get("context_length", "-")
                    model_desc = f"MicroGPT ({d_model}d, {layers}L, {heads}H, ctx {ctx})"

                    # Progress & Round fractions
                    cur_round = int(j.get("current_round", 0))
                    max_rounds = int(j.get("max_rounds", 10))
                    pct = int(round((cur_round / max(max_rounds, 1)) * 100)) if max_rounds > 0 else 0
                    progress_str = f"{pct}% (Round {cur_round}/{max_rounds})"

                    # Latest Round History Telemetry from batch map
                    r_info = latest_rounds_map.get(jid, {})
                    latest_loss = "-"
                    latest_speed = "-"
                    if r_info:
                        l_val = r_info.get("avg_loss")
                        if l_val is not None:
                            try:
                                latest_loss = f"{float(l_val):.4f}"
                            except Exception:
                                latest_loss = str(l_val)
                        m_dict = r_info.get("metrics", {})
                        s_val = m_dict.get("aggregate_tokens_per_sec")
                        if s_val is not None:
                            try:
                                latest_speed = f"{float(s_val):,.0f} tok/s"
                            except Exception:
                                latest_speed = str(s_val)

                    created_ts = j.get("created_at")
                    created_str = (
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(created_ts))
                        if created_ts
                        else "-"
                    )

                    job_rows.append([
                        st_display,
                        jid,
                        model_desc,
                        progress_str,
                        latest_loss,
                        latest_speed,
                        f"{cur_round} / {max_rounds}",
                        created_str,
                    ])

                # Determine which job details to fetch
                chosen_job_id = target_selection or active_job_id
                task_rows = []
                manifest_lines = []
                rounds = []
                job_info = None
                coord_log_tail = ""

                if chosen_job_id:
                    tasks = bus.get_job_tasks(chosen_job_id)
                    for t in tasks:
                        shard_idx = int(t.get("shard_index", 0))
                        total_shards = int(t.get("total_shards", 1))
                        shard_str = f"Shard {shard_idx + 1} of {total_shards}"
                        status_str = str(t.get("participant_status", "-")).upper()
                        worker_id = str(t.get("worker_id", "-"))
                        gpu = str(t.get("gpu_name") or "-")
                        host = str(t.get("hostname") or "-")
                        dev_str = f"{gpu} ({host})" if gpu != "-" else host
                        last_round = f"Round {t.get('last_synced_round', 0)}"

                        telem = t.get("telemetry", {})
                        loss_val = telem.get("avg_loss")
                        loss_str = f"{float(loss_val):.4f}" if loss_val is not None else "-"
                        speed_val = telem.get("tokens_per_sec")
                        speed_str = f"{float(speed_val):,.0f} tok/s" if speed_val is not None else "-"

                        task_rows.append([
                            shard_str,
                            status_str,
                            worker_id,
                            dev_str,
                            last_round,
                            loss_str,
                            speed_str,
                        ])

                    rounds = bus.get_all_round_history(chosen_job_id)
                    job_info = bus.get_job(chosen_job_id)
                    manifest_lines.append(f"=== JOB EVENT MANIFEST: {chosen_job_id} ===")
                    if job_info:
                        manifest_lines.append(
                            f"Status: {job_info.get('status')} | Dataset: {job_info.get('dataset_path')}"
                        )
                        manifest_lines.append(
                            f"Sync interval K: {job_info.get('sync_interval_steps')} steps | Max rounds: {job_info.get('max_rounds')}"
                        )
                    manifest_lines.append(f"Synchronized Rounds Completed: {len(rounds)}")
                    for r in rounds:
                        r_num = r.get("round_number", 0) + 1
                        r_loss = r.get("avg_loss", 0.0)
                        r_workers = ", ".join(r.get("participating_workers", []))
                        r_metrics = r.get("metrics", {})
                        r_spd = r_metrics.get("aggregate_tokens_per_sec", 0.0)
                        manifest_lines.append(
                            f"• Round {r_num}: Global Loss {r_loss:.4f} | Speed: {r_spd:,.0f} tok/s | Workers: [{r_workers}]"
                        )

                    # Check for coordinator log tail
                    coord_log = bus.jobs_dir / chosen_job_id / "coordinator.log"
                    if coord_log.exists():
                        try:
                            lines = coord_log.read_text(encoding="utf-8", errors="replace").splitlines()
                            coord_log_tail = "\n".join(lines[-30:])
                        except Exception:
                            pass

                # Read local worker log tail
                worker_log_tail = ""
                local_log_file = Path(tempfile.gettempdir()) / "cluster_worker_local.log"
                if local_log_file.exists():
                    try:
                        w_lines = local_log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                        worker_log_tail = "\n".join(w_lines[-50:])
                    except Exception:
                        pass

                # Check running local worker PIDs
                running_pids = get_all_running_worker_pids()

                payload = {
                    "job_rows": job_rows,
                    "total_jobs": len(jobs),
                    "chosen_job_id": chosen_job_id,
                    "active_job": active_job,
                    "task_rows": task_rows,
                    "manifest_text": "\n".join(manifest_lines),
                    "rounds": rounds,
                    "coord_log_tail": coord_log_tail,
                    "worker_log_tail": worker_log_tail,
                    "running_workers_count": len(running_pids),
                }
                bridge.data_ready.emit(payload)
            except Exception as exc:
                bridge.data_ready.emit({"error": str(exc)})
            finally:
                self._job_monitor_poll_in_progress = False

        threading.Thread(target=_bg_poll, daemon=True).start()

    def _apply_job_monitor_data(self, data: dict[str, Any]) -> None:
        """Apply polled cluster data to UI widgets on the main thread."""
        if "error" in data:
            if hasattr(self, "_log_cluster_event"):
                self._log_cluster_event(f"Error refreshing job monitor: {data['error']}")
            return

        job_rows = data.get("job_rows", [])
        chosen_job_id = data.get("chosen_job_id")
        task_rows = data.get("task_rows", [])
        manifest_text = data.get("manifest_text", "")
        active_job = data.get("active_job")
        running_workers = data.get("running_workers_count", 0)

        # Update Master Jobs Table
        self.cluster_jobs_table.blockSignals(True)
        set_table_rows(self.cluster_jobs_table, job_rows)

        if chosen_job_id:
            for row_idx, r in enumerate(job_rows):
                if len(r) > 1 and r[1] == chosen_job_id:
                    if self.cluster_jobs_table.currentRow() != row_idx:
                        self.cluster_jobs_table.selectRow(row_idx)
                    break
        self.cluster_jobs_table.blockSignals(False)

        if hasattr(self, "jobs_summary_label"):
            self.jobs_summary_label.setText(f"{data.get('total_jobs', len(job_rows))} jobs")

        if hasattr(self, "job_manager_progress"):
            self.job_manager_progress.setValue(100)

        # Update Tasks & Shards Table
        if hasattr(self, "cluster_tasks_table"):
            set_table_rows(self.cluster_tasks_table, task_rows)

        if hasattr(self, "selected_job_title_label") and chosen_job_id:
            self.selected_job_title_label.setText(f"<b>TASKS & DATA SHARDS</b> (Job: {chosen_job_id})")

        # Update Job Events Log
        if hasattr(self, "job_events_log") and manifest_text:
            if self.job_events_log.toPlainText() != manifest_text:
                sb = self.job_events_log.verticalScrollBar()
                at_bottom = sb.value() >= (sb.maximum() - 8)
                self.job_events_log.setPlainText(manifest_text)
                if at_bottom:
                    sb.setValue(sb.maximum())

        # Update Worker Diagnostics Log
        if hasattr(self, "cluster_worker_log") and data.get("worker_log_tail"):
            w_tail = data["worker_log_tail"]
            if self.cluster_worker_log.toPlainText() != w_tail:
                sb = self.cluster_worker_log.verticalScrollBar()
                at_bottom = sb.value() >= (sb.maximum() - 8)
                self.cluster_worker_log.setPlainText(w_tail)
                if at_bottom:
                    sb.setValue(sb.maximum())

        # Update Local Worker button state (guard against redundant text/size updates)
        if hasattr(self, "cluster_local_worker_btn"):
            new_btn_txt = f"Stop Local Worker(s) ({running_workers} active)" if running_workers > 0 else "Start Local Worker(s)"
            if self.cluster_local_worker_btn.text() != new_btn_txt:
                self.cluster_local_worker_btn.setText(new_btn_txt)

        # Synchronize Training Tab status and Stop button if a cluster job is active
        if active_job:
            cur_st = str(active_job.get("status", "")).upper()
            c_round = int(active_job.get("current_round", 0))
            m_rounds = int(active_job.get("max_rounds", 10))
            if cur_st == "RUNNING":
                if hasattr(self, "train_status"):
                    self.train_status.setText(f"Training: Cluster (Local SGD) - Round {c_round}/{m_rounds}")
                if hasattr(self, "project_state"):
                    self.project_state.setText("Training")
                if hasattr(self, "stop_training_button"):
                    self.stop_training_button.setEnabled(True)

                # Populate Training Tab Charts from loaded rounds
                rounds = data.get("rounds", [])
                if rounds:
                    last_r = rounds[-1]
                    eff_step = int(last_r.get("metrics", {}).get("effective_step", c_round * 250))
                    g_loss = float(last_r.get("avg_loss", 0.0))
                    spd = float(last_r.get("metrics", {}).get("aggregate_tokens_per_sec", 0.0))
                    if hasattr(self, "training_loss_metric"):
                        self.training_loss_metric.setText(f"Train loss: {g_loss:.4f}")
                    if hasattr(self, "training_speed_metric"):
                        self.training_speed_metric.setText(f"Speed: {spd:,.0f} tok/s")
                    if hasattr(self, "training_step_metric"):
                        self.training_step_metric.setText(f"Step: {eff_step} (Round {c_round}/{m_rounds})")
                    if hasattr(self, "loss_chart") and self.loss_chart:
                        self.loss_chart.add_metrics(eff_step, g_loss, None)
                    if hasattr(self, "throughput_chart") and self.throughput_chart:
                        self.throughput_chart.add_values(eff_step, spd)

                # Append coordinator log tail to training log if not yet present
                coord_tail = data.get("coord_log_tail", "")
                if coord_tail and hasattr(self, "training_log"):
                    current_text = self.training_log.toPlainText()
                    if f"Job: {active_job.get('job_id')}" not in current_text:
                        self.training_log.append(
                            f"\n=== Attached to Active Cluster Job: {active_job.get('job_id')} ===\n"
                            f"{coord_tail}\n"
                        )

    def on_cluster_job_selected(self) -> None:
        """Handle selection change in the Master Jobs Table: load tasks and diagnostic logs."""
        if not hasattr(self, "cluster_jobs_table"):
            return
        selected_items = self.cluster_jobs_table.selectedItems()
        if not selected_items:
            return
        row = self.cluster_jobs_table.currentRow()
        job_id_item = self.cluster_jobs_table.item(row, 1)
        if not job_id_item:
            return
        job_id = job_id_item.text().strip()
        self._selected_cluster_job_id = job_id
        # Trigger background refresh with new selection
        self.refresh_job_manager_tab()

    def _populate_tasks_for_job(self, job_id: str) -> None:
        """Populate tasks and diagnostics for the given job ID."""
        self._selected_cluster_job_id = job_id
        self.refresh_job_manager_tab()

    def requeue_selected_cluster_job(self) -> None:
        """Re-queue the selected cluster job to resume or restart training."""
        bus = self._get_cluster_bus() if hasattr(self, "_get_cluster_bus") else None
        if not bus:
            return
        job_id = self._selected_cluster_job_id
        if not job_id:
            QMessageBox.information(self, "Select Job", "Please select a job in the table to re-queue.")
            return

        choice = QMessageBox.question(
            self,
            "Re-queue Job",
            f"Do you want to re-queue job '{job_id}'?\n\n"
            "Yes: Resume from current round\n"
            "No: Reset back to Round 0 from scratch\n"
            "Cancel: Do nothing",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if choice == QMessageBox.Cancel:
            return

        reset_rounds = (choice == QMessageBox.No)
        success = bus.requeue_job(job_id, reset_rounds=reset_rounds)
        if success:
            msg = f"Job '{job_id}' re-queued ({'Round 0' if reset_rounds else 'Resuming'})."
            if hasattr(self, "_log_cluster_event"):
                self._log_cluster_event(msg)
            QMessageBox.information(self, "Job Re-queued", msg)
            self.refresh_job_manager_tab()
        else:
            QMessageBox.warning(self, "Re-queue Failed", f"Could not re-queue job '{job_id}'.")

    def delete_selected_cluster_job(self) -> None:
        """Permanently delete the selected cluster job from database and shared storage."""
        bus = self._get_cluster_bus() if hasattr(self, "_get_cluster_bus") else None
        if not bus:
            return

        job_id = getattr(self, "_selected_cluster_job_id", None)
        if not job_id and hasattr(self, "cluster_jobs_table"):
            row = self.cluster_jobs_table.currentRow()
            if row >= 0:
                item = self.cluster_jobs_table.item(row, 1)
                if item:
                    job_id = item.text().strip()

        if not job_id:
            QMessageBox.information(self, "Select Job", "Please select a job in the table to delete.")
            return

        reply = QMessageBox.question(
            self,
            "Delete Cluster Job",
            f"Are you sure you want to permanently delete job '{job_id}'?\n\n"
            "This will remove the job record, worker history, and all stored round checkpoints from shared storage.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Stop coordinator if running for this job
        if hasattr(self, "_coordinator_proc") and self._coordinator_proc:
            try:
                self._coordinator_proc.terminate()
            except Exception:
                pass
        pid_file = bus.jobs_dir / job_id / "coordinator.pid"
        if pid_file.exists():
            try:
                cpid = int(pid_file.read_text(encoding="utf-8").strip())
                import psutil
                if psutil.pid_exists(cpid):
                    psutil.Process(cpid).terminate()
                pid_file.unlink(missing_ok=True)
            except Exception:
                pass

        success = bus.delete_job(job_id)
        if success:
            if getattr(self, "_selected_cluster_job_id", None) == job_id:
                self._selected_cluster_job_id = None
            msg = f"Permanently deleted cluster job '{job_id}'."
            if hasattr(self, "_log_cluster_event"):
                self._log_cluster_event(msg)
            if hasattr(self, "job_events_log"):
                self.job_events_log.clear()
            self.refresh_job_manager_tab()
            if hasattr(self, "refresh_cluster_status"):
                self.refresh_cluster_status()
        else:
            QMessageBox.warning(self, "Delete Failed", f"Could not delete job '{job_id}'. Check storage permissions.")

    def show_cluster_jobs_context_menu(self, pos: QPoint) -> None:
        """Display right-click context menu for selected job in the jobs table."""
        if not hasattr(self, "cluster_jobs_table"):
            return
        row = self.cluster_jobs_table.rowAt(pos.y())
        if row < 0:
            return
        self.cluster_jobs_table.selectRow(row)
        item = self.cluster_jobs_table.item(row, 1)
        if not item:
            return
        job_id = item.text().strip()
        self._selected_cluster_job_id = job_id

        menu = QMenu(self)
        resume_act = menu.addAction("▶ Resume Job")
        pause_act = menu.addAction("⏸ Pause Job")
        stop_act = menu.addAction("⏹ Stop Job")
        requeue_act = menu.addAction("🔄 Re-queue Job")
        menu.addSeparator()
        delete_act = menu.addAction("🗑 Delete Job")

        action = menu.exec(self.cluster_jobs_table.viewport().mapToGlobal(pos))
        if action == resume_act:
            if hasattr(self, "resume_cluster_job"):
                self.resume_cluster_job(job_id)
        elif action == pause_act:
            bus = self._get_cluster_bus() if hasattr(self, "_get_cluster_bus") else None
            if bus:
                bus.set_job_status(job_id, "PAUSED")
                if hasattr(self, "_log_cluster_event"):
                    self._log_cluster_event(f"Paused job {job_id}.")
                self.refresh_job_manager_tab()
        elif action == stop_act:
            bus = self._get_cluster_bus() if hasattr(self, "_get_cluster_bus") else None
            if bus:
                bus.set_job_status(job_id, "STOPPED")
                if hasattr(self, "_log_cluster_event"):
                    self._log_cluster_event(f"Stopped job {job_id}.")
                self.refresh_job_manager_tab()
        elif action == requeue_act:
            self.requeue_selected_cluster_job()
        elif action == delete_act:
            self.delete_selected_cluster_job()

    def clear_active_cluster_log(self) -> None:
        """Clear whichever diagnostic log tab is currently active."""
        tabs = getattr(self, "cluster_details_tabs", getattr(self, "cluster_log_tabs", None))
        if not tabs:
            return
        cur_widget = tabs.currentWidget()
        if isinstance(cur_widget, QTextEdit):
            cur_widget.clear()
        elif cur_widget == getattr(self, "job_events_log", None):
            self.job_events_log.clear()
        elif cur_widget == getattr(self, "cluster_worker_log", None):
            self.cluster_worker_log.clear()
        elif cur_widget == getattr(self, "cluster_log", None):
            self.cluster_log.clear()

    # -------------------------------------------------------------------------
    # Backward Compatibility Aliases & Deprecated Methods
    # -------------------------------------------------------------------------

    def pause_all_managed_jobs(self) -> None:
        """Pause active cluster job."""
        if hasattr(self, "pause_cluster_job"):
            self.pause_cluster_job()

    def resume_all_managed_jobs(self) -> None:
        """Resume active cluster job."""
        if hasattr(self, "resume_cluster_job"):
            self.resume_cluster_job()

    def stop_all_managed_jobs(self) -> None:
        """Stop active cluster job."""
        if hasattr(self, "stop_cluster_job"):
            self.stop_cluster_job()

    def mark_stale_workers_offline(self) -> None:
        """Clean offline workers from fleet."""
        if hasattr(self, "clean_offline_cluster_workers"):
            self.clean_offline_cluster_workers()

    def publish_remote_training_job(self, *args, **kwargs) -> None:
        """Legacy remote publish redirect to Cluster Local SGD launch."""
        if hasattr(self, "launch_cluster_training_job"):
            self.launch_cluster_training_job()

    def start_coordinator_server(self) -> None:
        """Deprecated HTTP coordinator API."""
        pass

    def stop_coordinator_server(self) -> None:
        """Deprecated HTTP coordinator API."""
        pass
