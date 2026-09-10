"""Screen mixin for Thinkbox Deadline & Fox Renderfarm inspired Cluster Job Monitor."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QTextEdit

from interface.tabs.job_manager_tab import set_table_rows


class JobManagerScreenMixin:
    """Screen mixin providing master control and telemetry inspection for cluster jobs."""

    _selected_cluster_job_id: Optional[str] = None

    def refresh_job_manager_tab(self) -> None:
        """Refresh the entire Cluster Job Monitor: jobs list, tasks, worker fleet, and logs."""
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

        try:
            jobs = bus.list_all_jobs(limit=100)
            job_rows = []
            status_symbols = {
                "RUNNING": "▶ RUNNING",
                "PAUSED": "⏸ PAUSED",
                "STOPPED": "■ STOPPED",
                "COMPLETED": "✔ COMPLETED",
                "FAILED": "✖ FAILED",
                "QUEUED": "⏳ QUEUED",
            }

            active_job_id = None
            for j in jobs:
                jid = str(j.get("job_id", ""))
                st = str(j.get("status", "QUEUED")).upper()
                if st in {"RUNNING", "QUEUED", "PAUSED"} and not active_job_id:
                    active_job_id = jid

                st_display = status_symbols.get(st, st)

                # Model / Architecture description
                m_cfg = j.get("model_config", {})
                d_model = m_cfg.get("embedding_size") or m_cfg.get("hidden_dim", "-")
                layers = m_cfg.get("layer_count") or m_cfg.get("num_layers", "-")
                heads = m_cfg.get("head_count") or m_cfg.get("num_heads", "-")
                ctx = m_cfg.get("context_length", "-")
                model_desc = f"MicroGPT ({d_model}d, {layers}L, {heads}H, ctx {ctx})"

                # Progress & Round fractions (Fox Renderfarm style: "80% (Round 4/5)")
                cur_round = int(j.get("current_round", 0))
                max_rounds = int(j.get("max_rounds", 10))
                pct = int(round((cur_round / max(max_rounds, 1)) * 100)) if max_rounds > 0 else 0
                progress_str = f"{pct}% (Round {cur_round}/{max_rounds})"

                # Latest Round History Telemetry for loss and speed
                all_rounds = bus.get_all_round_history(jid)
                latest_loss = "-"
                latest_speed = "-"
                if all_rounds:
                    last_r = all_rounds[-1]
                    l_val = last_r.get("avg_loss")
                    if l_val is not None:
                        try:
                            latest_loss = f"{float(l_val):.4f}"
                        except Exception:
                            latest_loss = str(l_val)
                    m_dict = last_r.get("metrics", {})
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

            set_table_rows(self.cluster_jobs_table, job_rows)

            if hasattr(self, "jobs_summary_label"):
                self.jobs_summary_label.setText(f"{len(jobs)} total jobs")

            if hasattr(self, "job_manager_progress"):
                self.job_manager_progress.setValue(100)

            # Auto-select either previous selection or the active job
            selected_id = self._selected_cluster_job_id or active_job_id
            if selected_id:
                for row_idx, r in enumerate(job_rows):
                    if r[1] == selected_id:
                        if self.cluster_jobs_table.currentRow() != row_idx:
                            self.cluster_jobs_table.selectRow(row_idx)
                        break
                self._populate_tasks_for_job(selected_id)

        except Exception as exc:
            if hasattr(self, "_log_cluster_event"):
                self._log_cluster_event(f"Error refreshing jobs: {exc}")

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
        self._populate_tasks_for_job(job_id)

    def _populate_tasks_for_job(self, job_id: str) -> None:
        """Populate the top-right Tasks & Data Shards table for the given job ID."""
        if not hasattr(self, "cluster_tasks_table"):
            return
        if hasattr(self, "selected_job_title_label"):
            self.selected_job_title_label.setText(f"<b>TASKS & DATA SHARDS</b> (Job: {job_id})")

        bus = self._get_cluster_bus() if hasattr(self, "_get_cluster_bus") else None
        if not bus:
            return

        try:
            tasks = bus.get_job_tasks(job_id)
            task_rows = []
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
            set_table_rows(self.cluster_tasks_table, task_rows)

            # Also display job history summary in Job Events Log
            if hasattr(self, "job_events_log"):
                rounds = bus.get_all_round_history(job_id)
                self.job_events_log.clear()
                self.job_events_log.append(f"=== JOB EVENT MANIFEST: {job_id} ===")
                job_info = bus.get_job(job_id)
                if job_info:
                    self.job_events_log.append(f"Status: {job_info.get('status')} | Dataset: {job_info.get('dataset_path')}")
                    self.job_events_log.append(f"Sync interval K: {job_info.get('sync_interval_steps')} steps | Max rounds: {job_info.get('max_rounds')}")
                self.job_events_log.append(f"Synchronized Rounds Completed: {len(rounds)}")
                for r in rounds:
                    r_num = r.get("round_number", 0) + 1
                    r_loss = r.get("avg_loss", 0.0)
                    r_workers = ", ".join(r.get("participating_workers", []))
                    r_metrics = r.get("metrics", {})
                    r_spd = r_metrics.get("aggregate_tokens_per_sec", 0.0)
                    self.job_events_log.append(
                        f"• Round {r_num}: Global Loss {r_loss:.4f} | Speed: {r_spd:,.0f} tok/s | Workers: [{r_workers}]"
                    )

        except Exception as exc:
            if hasattr(self, "_log_cluster_event"):
                self._log_cluster_event(f"Error loading tasks for job {job_id}: {exc}")

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
