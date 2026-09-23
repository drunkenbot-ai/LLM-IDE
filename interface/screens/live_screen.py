from __future__ import annotations

# LiveScreenMixin mixin. Shared runtime names are provided by interface.app.
import time
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class LiveScreenMixin:
    @staticmethod
    def _metric_pair(value: Optional[int], total: Optional[int]) -> str:
        """Format a metric pair.

        Args:
            value: Current value.
            total: Total value.

        Returns:
            Display text.
        """

        if value is None:
            return "-"
        if total is None:
            return str(value)
        return f"{value}/{total}"

    @staticmethod
    def _metric_float(value: Optional[float], suffix: str = "") -> str:
        """Format a floating-point metric.

        Args:
            value: Metric value.
            suffix: Optional suffix.

        Returns:
            Display text.
        """

        if value is None:
            return "-"
        return f"{value:.4g}{suffix}"

    def _init_telemetry_store(self, model_dir: Path) -> None:
        """Create or reset the SQLite telemetry store for a training run.

        Args:
            model_dir: Model output directory.
        """

        self.telemetry_db_path = initialize_store(model_dir)
        self.telemetry_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_timeline_label.setText("Timeline: live")
        self.live_scrub_active = False

    def _record_live_metric(self, event: dict[str, Any]) -> None:
        """Persist one live training metric event to SQLite.

        Args:
            event: Training progress event.
        """

        if self.telemetry_db_path is None or not self.telemetry_run_id or event.get("step") is None:
            return
        self.telemetry_latest_id = insert_metric(self.telemetry_db_path, self.telemetry_run_id, event)
        self.telemetry_latest_index += 1
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, self.telemetry_latest_index)
        if not self.live_scrub_active:
            self.live_time_slider.setValue(self.telemetry_latest_index)
            self.live_timeline_label.setText("Timeline: live")
        self.live_time_slider.blockSignals(False)

    def _load_existing_telemetry(self, model_dir: Path) -> None:
        """Load the latest saved telemetry run for an opened project.

        Args:
            model_dir: Model output directory that may contain ``training_telemetry.sqlite``.
        """

        db_path = telemetry_db_path(model_dir)
        self.telemetry_db_path = db_path if db_path.exists() else None
        self.telemetry_run_id = ""
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_scrub_active = False
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_time_slider.blockSignals(False)
        self.live_timeline_label.setText("Timeline: no saved telemetry")
        self.live_sample_text.setText("Training text: -")
        if self.telemetry_db_path is None:
            return
        try:
            run_row = latest_run(self.telemetry_db_path)
            if run_row is None:
                self.live_timeline_label.setText("Timeline: no samples")
                return
            self.telemetry_run_id = str(run_row["run_id"])
            self.telemetry_latest_index = int(run_row["sample_count"] or 0)
            self.telemetry_latest_id = int(run_row["latest_id"] or 0)
        except sqlite3.Error as exc:
            self.live_timeline_label.setText("Timeline: could not load")
            self.training_log.append(f"Telemetry load warning: {exc}")
            return
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, self.telemetry_latest_index)
        self.live_time_slider.setValue(self.telemetry_latest_index)
        self.live_time_slider.blockSignals(False)
        if (
            self.telemetry_latest_index
            and self.pages.currentIndex() == self.live_page_index
        ):
            rows = self._timeline_rows_until(self.telemetry_latest_index)
            if rows:
                self._apply_timeline_rows(rows)

    def _timeline_rows_until(self, sample_index: int) -> list[sqlite3.Row]:
        """Load telemetry rows up to a selected sample index.

        Args:
            sample_index: Maximum number of samples to load for the active run.

        Returns:
            Ordered telemetry rows for the active run.
        """

        if self.telemetry_db_path is None or not self.telemetry_run_id or sample_index <= 0:
            return []
        return rows_until(self.telemetry_db_path, self.telemetry_run_id, sample_index)

    def _begin_live_scrub(self) -> None:
        """Pause live auto-follow while the timeline slider is being dragged."""

        self.live_scrub_active = True

    def _end_live_scrub(self) -> None:
        """Apply the selected timeline snapshot after slider drag."""

        self._scrub_live_timeline(self.live_time_slider.value())

    def _jump_live_timeline_to_latest(self) -> None:
        """Return timeline display to the latest live point."""

        self.live_scrub_active = False
        self.live_time_slider.setValue(self.telemetry_latest_index)
        self._scrub_live_timeline(self.telemetry_latest_index)
        self.live_timeline_label.setText("Timeline: live")

    def _scrub_live_timeline(self, sample_index: int) -> None:
        """Replay charts and live visual widgets to a selected telemetry point.

        Args:
            sample_index: Timeline sample selected by the slider.
        """

        rows = self._timeline_rows_until(sample_index)
        if not rows:
            return
        self._apply_timeline_rows(rows)

    def _apply_timeline_rows(self, rows: list[sqlite3.Row]) -> None:
        """Apply historical telemetry rows to charts and live widgets.

        Args:
            rows: Ordered SQLite telemetry rows.
        """

        def series(name: str) -> list[tuple[int, float]]:
            return [(int(row["step"]), float(row[name])) for row in rows if row[name] is not None]

        latest = rows[-1]
        self.loss_chart.set_points(series("train_loss"), series("val_loss"))
        self.optimization_chart.set_points(series("learning_rate"), series("grad_norm"))
        self.stability_chart.set_points(series("weight_norm"), series("update_ratio"))
        self.throughput_chart.set_points(series("tokens_per_second"), series("samples_per_second"))
        self.memory_chart.set_points(series("vram_allocated_gb"), series("vram_reserved_gb"))
        snapshot = {key: latest[key] for key in latest.keys()}
        sample_text = str(snapshot.get("sample_text") or "").strip()
        if sample_text:
            self.live_sample_text.setText(f"Training text: {self._compact_preview_text(sample_text, 220)}")
        else:
            self.live_sample_text.setText("Training text: -")
        self._update_live_training_metrics(
            int(latest["step"]),
            snapshot,
            snapshot.get("train_loss"),
            snapshot.get("learning_rate"),
            snapshot.get("grad_norm"),
            snapshot.get("update_ratio"),
            snapshot.get("tokens_per_second"),
            snapshot.get("samples_per_second"),
            snapshot.get("vram_allocated_gb"),
            snapshot.get("vram_reserved_gb"),
            snapshot.get("gpu_memory_percent"),
            snapshot.get("system_cpu_percent"),
            snapshot.get("system_ram_percent"),
            snapshot.get("data_loader_workers"),
        )
        timestamp = datetime.fromtimestamp(float(latest["recorded_at"])).strftime("%H:%M:%S")
        self.live_timeline_label.setText(f"Timeline: step {int(latest['step']):,} @ {timestamp}")

    def _update_live_training_metrics(
        self,
        step: int,
        event: dict[str, Any],
        train_loss: Optional[float],
        learning_rate: Optional[float],
        grad_norm: Optional[float],
        update_ratio: Optional[float],
        tokens_per_second: Optional[float],
        samples_per_second: Optional[float],
        vram_allocated: Optional[float],
        vram_reserved: Optional[float],
        gpu_memory: Optional[float],
        system_cpu: Optional[float],
        system_ram: Optional[float],
        data_workers: Optional[int],
        val_loss: Optional[float] = None,
        eta_seconds: Optional[float] = None,
        health_label: Optional[str] = None,
    ) -> None:
        """Update live tracker widgets from one training progress event.

        Args:
            step: Current optimizer step.
            event: Progress event emitted by training.
            train_loss: Latest training loss.
            learning_rate: Current learning rate.
            grad_norm: Current gradient norm.
            update_ratio: Current parameter update ratio.
            tokens_per_second: Current token throughput.
            samples_per_second: Current sample throughput.
            vram_allocated: Current CUDA allocated memory in GB.
            vram_reserved: Current CUDA reserved memory in GB.
            gpu_memory: Current GPU memory pressure percentage.
            system_cpu: Current system CPU utilization percentage.
            system_ram: Current system RAM utilization percentage.
            data_workers: CPU data-loader worker count.
            val_loss: Optional validation loss.
            eta_seconds: Optional remaining time in seconds.
            health_label: Optional health / early stopping diagnosis label.
        """

        total_steps = event.get("total_steps")
        epoch_str = "—"
        if "epoch" in event and "total_epochs" in event:
            epoch_str = f"{event['epoch']} / {event['total_epochs']}"
            if hasattr(self, "live_epoch_metric"):
                self.live_epoch_metric.setText(epoch_str)

        step_str = f"{step:,}"
        if total_steps:
            step_str = f"{step:,} / {int(total_steps):,}"
            data_percent = min(100.0, max(0.0, (step / max(1, int(total_steps))) * 100.0))
            if hasattr(self, "live_data_metric"):
                self.live_data_metric.setText(f"Data: {data_percent:.1f}%")
            if hasattr(self, "live_progress"):
                self.live_progress.setValue(int(data_percent))

        active_mode = getattr(self, "_active_live_mode", "local")

        # Top metric cards update when in local mode
        if active_mode == "local":
            if hasattr(self, "live_step_metric"):
                self.live_step_metric.setText(step_str)
            if tokens_per_second is not None and hasattr(self, "live_tokens_metric"):
                self.live_tokens_metric.setText(f"{float(tokens_per_second):,.0f} tok/s")
            if train_loss is not None and hasattr(self, "live_loss_metric"):
                self.live_loss_metric.setText(f"{float(train_loss):.4f}")

            # Validation loss resolution
            resolved_val_loss = val_loss if val_loss is not None else event.get("val_loss")
            if resolved_val_loss is not None and hasattr(self, "live_val_loss_metric"):
                try:
                    self.live_val_loss_metric.setText(f"{float(resolved_val_loss):.4f}")
                except (ValueError, TypeError):
                    pass

            # ETA resolution
            resolved_eta = eta_seconds if eta_seconds is not None else event.get("eta_seconds")
            if resolved_eta is not None and hasattr(self, "live_eta_metric"):
                try:
                    self.live_eta_metric.setText(self._format_duration(float(resolved_eta)))
                except Exception:
                    pass

        # Health & Stability card metrics
        if learning_rate is not None and hasattr(self, "live_lr_metric"):
            self.live_lr_metric.setText(f"{float(learning_rate):.2e}")

        if grad_norm is not None and hasattr(self, "live_grad_norm_metric"):
            clip_str = " (< 1.0)" if float(grad_norm) < 1.0 else " (clipping)"
            self.live_grad_norm_metric.setText(f"{float(grad_norm):.3f}{clip_str}")

        if health_label is not None and hasattr(self, "live_early_stop_metric"):
            self.live_early_stop_metric.setText(health_label)
        elif train_loss is not None and hasattr(self, "live_early_stop_metric"):
            if self.live_early_stop_metric.text().startswith("Standby"):
                self.live_early_stop_metric.setText("Healthy (Tracking)")

        # Checkpoint Probe sample text streaming
        sample_text = str(event.get("sample_text") or "").strip()
        if sample_text:
            if hasattr(self, "live_sample_text"):
                self.live_sample_text.setText(f"Training text: {self._compact_preview_text(sample_text, 220)}")
            if hasattr(self, "live_probe_prompt"):
                self.live_probe_prompt.setText("Batch sample preview:")
            if hasattr(self, "live_probe_output"):
                self.live_probe_output.setText(f"'{self._compact_preview_text(sample_text, 180)}'")

        # Status labels
        if hasattr(self, "live_layer_status"):
            self.live_layer_status.setText(f"Layers: {self.n_layer.value()}")
        if hasattr(self, "live_head_status"):
            self.live_head_status.setText(f"Heads: {self.n_head.value()}")
        if hasattr(self, "live_hidden_status"):
            self.live_hidden_status.setText(f"Hidden size: {self.n_embd.value()}")
        if hasattr(self, "live_batch_status"):
            self.live_batch_status.setText(f"Batch size: {self.batch_size.value()}")
        if hasattr(self, "live_context_status"):
            self.live_context_status.setText(f"Context: {self.train_context_length.value()}")
        if hasattr(self, "live_device_status"):
            self.live_device_status.setText(f"Device: {self.device.currentText()}")
        if hasattr(self, "live_worker_status"):
            self.live_worker_status.setText(f"CPU workers: {data_workers if data_workers is not None else self.data_loader_workers.value()}")

        if hasattr(self, "live_cpu_bar"):
            self._set_meter(self.live_cpu_bar, "System CPU", system_cpu if system_cpu is not None else self._system_cpu_value())
        if hasattr(self, "live_gpu_bar"):
            self._set_meter(self.live_gpu_bar, "GPU memory", gpu_memory)

        if vram_allocated is not None or vram_reserved is not None:
            allocated = float(vram_allocated or 0.0)
            reserved = float(vram_reserved or 0.0)
            reserved_percent = None
            if hasattr(self, "device") and self.device.currentText().startswith("cuda") and torch.cuda.is_available():
                try:
                    _, total_vram = torch.cuda.mem_get_info()
                    reserved_percent = min(100.0, 100.0 * reserved * (1024 ** 3) / max(total_vram, 1))
                except Exception:
                    reserved_percent = None
            if hasattr(self, "live_vram_bar"):
                self._set_meter(self.live_vram_bar, "VRAM reserved", reserved_percent)
            if hasattr(self, "live_vram_label"):
                self.live_vram_label.setText(f"VRAM reserved: {reserved:.2f} GB ({allocated:.2f} GB active)")
            if hasattr(self, "live_vram_metric"):
                pct_str = f" ({reserved_percent:.1f}%)" if reserved_percent is not None else ""
                self.live_vram_metric.setText(f"{reserved:.1f} GB reserved / {allocated:.1f} GB active{pct_str}")

        if hasattr(self, "live_ram_bar"):
            self._set_meter(self.live_ram_bar, "System RAM", system_ram if system_ram is not None else self._system_ram_value())

        latest_loss = float(train_loss) if train_loss is not None else None
        if hasattr(self, "live_flow"):
            self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), step, latest_loss)
        if hasattr(self, "live_prediction_chart"):
            self.live_prediction_chart.update_distribution(step, latest_loss)
        if hasattr(self, "live_attention_chart"):
            self.live_attention_chart.update_heatmap(step, grad_norm)
        if hasattr(self, "live_activation_chart"):
            self.live_activation_chart.update_histogram(step, tokens_per_second)
        if hasattr(self, "live_gradient_chart"):
            self.live_gradient_chart.update_flow(self.n_layer.value(), grad_norm, step)

        # Cache snapshot for tab view switching
        self._last_local_telemetry = {
            "epoch_str": epoch_str,
            "step_str": step_str,
            "train_loss": f"{float(train_loss):.4f}" if train_loss is not None else "—",
            "val_loss": f"{float(val_loss):.4f}" if val_loss is not None else (f"{float(event.get('val_loss')):.4f}" if event.get("val_loss") is not None else "—"),
            "throughput": f"{float(tokens_per_second):,.0f} tok/s" if tokens_per_second is not None else "0 tok/s",
            "eta": self._format_duration(float(eta_seconds)) if eta_seconds is not None else (self._format_duration(float(event.get("eta_seconds"))) if event.get("eta_seconds") is not None else "—"),
        }

    def _set_live_training_badge(
        self,
        state: str,
        pid: Optional[int] = None,
        job_id: str = "",
        extra: str = "",
    ) -> None:
        """Update the flight deck status badge and visual styling."""
        if not hasattr(self, "live_training_badge"):
            return
        state_upper = state.upper()
        if state_upper == "LOCAL_ACTIVE":
            pid_str = f" (PID: {pid})" if pid else ""
            self.live_training_badge.setText(f"● LOCAL TRAINING ACTIVE{pid_str}")
            self.live_training_badge.setStyleSheet(
                "QLabel { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 800; }"
            )
        elif state_upper == "CLUSTER_ACTIVE":
            job_str = f" (Job: {job_id})" if job_id else ""
            extra_str = f" | {extra}" if extra else ""
            self.live_training_badge.setText(f"● CLUSTER FLEET ACTIVE{job_str}{extra_str}")
            self.live_training_badge.setStyleSheet(
                "QLabel { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #38bdf8; border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 800; }"
            )
        elif state_upper == "PAUSED":
            self.live_training_badge.setText(f"⏸ TRAINING PAUSED {extra}".strip())
            self.live_training_badge.setStyleSheet(
                "QLabel { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid #f59e0b; border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 800; }"
            )
        elif state_upper == "COMPLETED":
            self.live_training_badge.setText("✔ TRAINING COMPLETED")
            self.live_training_badge.setStyleSheet(
                "QLabel { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 800; }"
            )
        elif state_upper == "STOPPED":
            self.live_training_badge.setText("■ TRAINING STOPPED")
            self.live_training_badge.setStyleSheet(
                "QLabel { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid #ef4444; border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 800; }"
            )
        else:  # STANDBY / IDLE
            self.live_training_badge.setText("○ STANDBY - READY TO TRAIN")
            self.live_training_badge.setStyleSheet(
                "QLabel { background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid #475569; border-radius: 12px; padding: 4px 12px; font-size: 11px; font-weight: 800; }"
            )

    def _update_live_cluster_metrics(self, telemetry: dict[str, Any]) -> None:
        """Update the Live Flight Deck with distributed cluster round telemetry."""
        round_num = int(telemetry.get("round", 0))
        cur_round = round_num + 1
        max_rounds = int(telemetry.get("max_rounds", 10))
        effective_step = int(telemetry.get("effective_step", 0))
        global_loss = float(telemetry.get("global_loss", 0.0))
        val_loss = telemetry.get("val_loss")
        if val_loss is not None:
            try:
                val_loss = float(val_loss)
            except (ValueError, TypeError):
                val_loss = None
        speed = float(telemetry.get("aggregate_tokens_per_sec", 0.0))
        workers_count = int(telemetry.get("ready_workers_count", 0))
        worker_losses = telemetry.get("worker_losses", {})
        job_id = str(telemetry.get("job_id", ""))
        epoch_val = telemetry.get("epoch")
        target_epochs = telemetry.get("target_epochs", 1)

        bus = self._get_cluster_bus() if hasattr(self, "_get_cluster_bus") else None
        job = bus.get_job(job_id) if bus and job_id else None
        sync_interval = int(job.get("sync_interval_steps") or 250) if job else 250
        total_steps = max_rounds * sync_interval

        # Calculate cluster ETA
        now = time.time()
        if not hasattr(self, "_cluster_round_times") or not self._cluster_round_times:
            self._cluster_round_times = []
        self._cluster_round_times.append(now)
        remaining_rounds = max(0, max_rounds - cur_round)
        if remaining_rounds == 0:
            eta_str = "00:00"
        elif len(self._cluster_round_times) >= 2:
            avg_round_sec = (self._cluster_round_times[-1] - self._cluster_round_times[0]) / max(1, len(self._cluster_round_times) - 1)
            eta_seconds = remaining_rounds * avg_round_sec
            eta_str = self._format_duration(eta_seconds)
        else:
            eta_str = "-"

        # Store latest cluster snapshot
        self._last_cluster_telemetry = {
            "cur_round": cur_round,
            "max_rounds": max_rounds,
            "effective_step": effective_step,
            "total_steps": total_steps,
            "global_loss": global_loss,
            "val_loss": val_loss,
            "speed": speed,
            "workers_count": workers_count,
            "worker_losses": worker_losses,
            "eta_str": eta_str,
            "job_id": job_id,
            "epoch_val": epoch_val,
            "target_epochs": target_epochs,
            "sync_interval": sync_interval,
        }

        # Update Badge
        self._set_live_training_badge("CLUSTER_ACTIVE", job_id=job_id, extra=f"Round {cur_round}/{max_rounds}")

        # If current live tab view is Cluster (or auto-switched), update top cards
        active_mode = getattr(self, "_active_live_mode", "local")
        if active_mode == "cluster":
            if hasattr(self, "live_epoch_metric"):
                if epoch_val is not None:
                    self.live_epoch_metric.setText(f"Epoch: {epoch_val:.2f}/{target_epochs} (R{cur_round})")
                else:
                    self.live_epoch_metric.setText(f"Round: {cur_round}/{max_rounds}")
            if hasattr(self, "live_step_metric"):
                self.live_step_metric.setText(f"Step: {effective_step:,} / {total_steps:,}")
            if hasattr(self, "live_loss_metric"):
                self.live_loss_metric.setText(f"Loss: {global_loss:.4f}")
            if hasattr(self, "live_val_loss_metric"):
                self.live_val_loss_metric.setText(f"{val_loss:.4f}" if val_loss is not None else "—")
            if hasattr(self, "live_tokens_metric"):
                self.live_tokens_metric.setText(f"Speed: {speed:,.0f} tok/s")
            if hasattr(self, "live_eta_metric"):
                self.live_eta_metric.setText(eta_str)
            if hasattr(self, "live_progress"):
                self.live_progress.setValue(int((cur_round / max(max_rounds, 1)) * 100))

        # Update Cluster Fleet Health Widgets (on right column)
        if hasattr(self, "live_cluster_nodes_metric"):
            self.live_cluster_nodes_metric.setText(f"{workers_count} Active Workers")
        if hasattr(self, "live_cluster_sync_metric"):
            self.live_cluster_sync_metric.setText(f"Local SGD ({sync_interval} steps/rnd)")
        if hasattr(self, "live_cluster_consensus_metric"):
            consensus = "OPTIMAL" if global_loss < 7.0 else "CONVERGING"
            self.live_cluster_consensus_metric.setText(f"Synchronized ({consensus})")

        # Update VRAM summary if available
        if bus and hasattr(self, "live_cluster_vram_metric"):
            try:
                workers = bus.list_workers()
                vram_sum = sum(float(w.get("vram_used_gb") or 0.0) for w in workers)
                if vram_sum > 0:
                    self.live_cluster_vram_metric.setText(f"{vram_sum:.1f} GB Fleet Total")
            except Exception:
                pass

        # Update Cluster Worker Fleet Breakdown list
        if hasattr(self, "update_live_cluster_workers"):
            self.update_live_cluster_workers(worker_losses, workers_count)

    def _switch_live_telemetry_mode(self, mode: str) -> None:
        """Switch the Live Flight Deck between Local Process and Cluster Fleet view."""
        self._active_live_mode = mode
        is_cluster = (mode == "cluster")

        if hasattr(self, "live_mode_local_btn"):
            self.live_mode_local_btn.setChecked(not is_cluster)
        if hasattr(self, "live_mode_cluster_btn"):
            self.live_mode_cluster_btn.setChecked(is_cluster)

        # Toggle Right Column widgets
        if hasattr(self, "live_local_health_card"):
            self.live_local_health_card.setVisible(not is_cluster)
        if hasattr(self, "live_local_probe_card"):
            self.live_local_probe_card.setVisible(not is_cluster)
        if hasattr(self, "live_cluster_health_card"):
            self.live_cluster_health_card.setVisible(is_cluster)
        if hasattr(self, "live_cluster_fleet_card"):
            self.live_cluster_fleet_card.setVisible(is_cluster)

        # Update top metric cards and badge for the selected mode
        if is_cluster:
            cluster_data = getattr(self, "_last_cluster_telemetry", None)
            if cluster_data:
                cur_round = cluster_data["cur_round"]
                max_rounds = cluster_data["max_rounds"]
                effective_step = cluster_data["effective_step"]
                total_steps = cluster_data["total_steps"]
                global_loss = cluster_data["global_loss"]
                val_loss = cluster_data["val_loss"]
                speed = cluster_data["speed"]
                eta_str = cluster_data["eta_str"]
                epoch_val = cluster_data.get("epoch_val")
                target_epochs = cluster_data.get("target_epochs", 1)

                if hasattr(self, "live_epoch_metric"):
                    if epoch_val is not None:
                        self.live_epoch_metric.setText(f"Epoch: {epoch_val:.2f}/{target_epochs} (R{cur_round})")
                    else:
                        self.live_epoch_metric.setText(f"Round: {cur_round}/{max_rounds}")
                if hasattr(self, "live_step_metric"):
                    self.live_step_metric.setText(f"Step: {effective_step:,} / {total_steps:,}")
                if hasattr(self, "live_loss_metric"):
                    self.live_loss_metric.setText(f"Loss: {global_loss:.4f}")
                if hasattr(self, "live_val_loss_metric"):
                    self.live_val_loss_metric.setText(f"{val_loss:.4f}" if val_loss is not None else "—")
                if hasattr(self, "live_tokens_metric"):
                    self.live_tokens_metric.setText(f"Speed: {speed:,.0f} tok/s")
                if hasattr(self, "live_eta_metric"):
                    self.live_eta_metric.setText(eta_str)
            else:
                if hasattr(self, "live_epoch_metric"):
                    self.live_epoch_metric.setText("—")
                if hasattr(self, "live_step_metric"):
                    self.live_step_metric.setText("—")
                if hasattr(self, "live_loss_metric"):
                    self.live_loss_metric.setText("—")
                if hasattr(self, "live_val_loss_metric"):
                    self.live_val_loss_metric.setText("—")
                if hasattr(self, "live_tokens_metric"):
                    self.live_tokens_metric.setText("0 tok/s")
                if hasattr(self, "live_eta_metric"):
                    self.live_eta_metric.setText("—")
        else:
            local_data = getattr(self, "_last_local_telemetry", None)
            if local_data:
                if hasattr(self, "live_epoch_metric"):
                    self.live_epoch_metric.setText(local_data.get("epoch_str", "—"))
                if hasattr(self, "live_step_metric"):
                    self.live_step_metric.setText(local_data.get("step_str", "—"))
                if hasattr(self, "live_loss_metric"):
                    self.live_loss_metric.setText(local_data.get("train_loss", "—"))
                if hasattr(self, "live_val_loss_metric"):
                    self.live_val_loss_metric.setText(local_data.get("val_loss", "—"))
                if hasattr(self, "live_tokens_metric"):
                    self.live_tokens_metric.setText(local_data.get("throughput", "0 tok/s"))
                if hasattr(self, "live_eta_metric"):
                    self.live_eta_metric.setText(local_data.get("eta", "—"))
            elif hasattr(self, "_render_current_live_snapshot"):
                self._render_current_live_snapshot()

    def _system_ram_value(self) -> Optional[float]:
        """Read system RAM utilization for live telemetry.

        Returns:
            System RAM percentage, or None when unavailable.
        """

        if psutil is None:
            return None
        return float(psutil.virtual_memory().percent)

    def _system_cpu_value(self) -> Optional[float]:
        """Read system CPU utilization for live telemetry.

        Returns:
            System CPU percentage, or None when unavailable.
        """

        if psutil is None:
            return None
        return float(psutil.cpu_percent(interval=None))
