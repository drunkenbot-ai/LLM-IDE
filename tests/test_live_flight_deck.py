from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from interface import app as interface_app


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("license_valid", True)
    return app


def test_live_flight_deck_initial_standby_state(qapp) -> None:
    """Live tab must not have mocked numbers on initialization; displays standby state."""
    window = interface_app.MainWindow()
    try:
        # Badge
        badge_text = window.live_training_badge.text()
        assert "STANDBY" in badge_text
        assert "18492" not in badge_text

        # Top 6 Metrics
        assert window.live_epoch_metric.text() == "—"
        assert window.live_step_metric.text() == "—"
        assert window.live_loss_metric.text() == "—"
        assert window.live_val_loss_metric.text() == "—"
        assert window.live_tokens_metric.text() == "0 tok/s"
        assert window.live_eta_metric.text() == "—"

        # Health
        assert window.live_lr_metric.text() == "—"
        assert window.live_grad_norm_metric.text() == "—"
        assert "Standby" in window.live_early_stop_metric.text()

        # Checkpoint Probe
        assert "Awaiting" in window.live_probe_prompt.text()
        assert "(No checkpoint evaluated yet)" in window.live_probe_output.text()

        # Hardware Status detected
        assert "Ready" in window.live_cuda_status.text()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_live_training_metrics_update_local(qapp) -> None:
    """_update_live_training_metrics updates all cards and probe in local mode."""
    window = interface_app.MainWindow()
    try:
        event = {
            "step": 250,
            "total_steps": 1000,
            "epoch": 2,
            "total_epochs": 5,
            "train_loss": 1.4523,
            "val_loss": 1.6210,
            "eta_seconds": 125,
            "tokens_per_second": 14250,
            "sample_text": "def calculate_loss(): pass",
        }

        window._update_live_training_metrics(
            step=250,
            event=event,
            train_loss=1.4523,
            learning_rate=2.5e-4,
            grad_norm=0.42,
            update_ratio=0.01,
            tokens_per_second=14250,
            samples_per_second=32.0,
            vram_allocated=4.5,
            vram_reserved=6.2,
            gpu_memory=65.0,
            system_cpu=35.0,
            system_ram=45.0,
            data_workers=4,
            val_loss=1.6210,
            eta_seconds=125,
            health_label="Healthy (Nominal)",
        )

        assert window.live_epoch_metric.text() == "2 / 5"
        assert window.live_step_metric.text() == "250 / 1,000"
        assert window.live_loss_metric.text() == "1.4523"
        assert window.live_val_loss_metric.text() == "1.6210"
        assert window.live_tokens_metric.text() == "14,250 tok/s"
        assert window.live_eta_metric.text() == "2m 05s"
        assert "0.420" in window.live_grad_norm_metric.text()
        assert window.live_lr_metric.text() == "2.50e-04"
        assert window.live_early_stop_metric.text() == "Healthy (Nominal)"
        assert "calculate_loss" in window.live_probe_output.text()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_live_mode_switcher_and_cluster_telemetry(qapp) -> None:
    """Toggling to cluster mode switches cards and streams cluster round telemetry."""
    window = interface_app.MainWindow()
    try:
        # Initial view is local
        assert not window.live_local_health_card.isHidden()
        assert window.live_cluster_health_card.isHidden()

        # Switch to cluster mode
        window._switch_live_telemetry_mode("cluster")
        assert window.live_local_health_card.isHidden()
        assert not window.live_cluster_health_card.isHidden()
        assert not window.live_cluster_fleet_card.isHidden()

        cluster_telemetry = {
            "round": 3,
            "max_rounds": 10,
            "effective_step": 1000,
            "global_loss": 1.7845,
            "val_loss": 1.8320,
            "aggregate_tokens_per_sec": 48500,
            "ready_workers_count": 4,
            "worker_losses": {"worker_node_1": 1.76, "worker_node_2": 1.81},
            "job_id": "job_fl_42",
            "epoch": 1.5,
            "target_epochs": 3,
        }

        window._update_live_cluster_metrics(cluster_telemetry)

        # Check badge
        assert "CLUSTER FLEET ACTIVE" in window.live_training_badge.text()
        assert "job_fl_42" in window.live_training_badge.text()

        # Check top 6 cards in cluster mode
        assert "R4" in window.live_epoch_metric.text() or "Round" in window.live_epoch_metric.text()
        assert "1,000" in window.live_step_metric.text()
        assert window.live_loss_metric.text() == "1.7845"
        assert window.live_val_loss_metric.text() == "1.8320"
        assert "48,500" in window.live_tokens_metric.text()

        # Check cluster health cards
        assert "4 Active Workers" in window.live_cluster_nodes_metric.text()
        assert "Local SGD" in window.live_cluster_sync_metric.text()
        assert "Synchronized" in window.live_cluster_consensus_metric.text()

        # Switch back to local mode
        window._switch_live_telemetry_mode("local")
        assert not window.live_local_health_card.isHidden()
        assert window.live_cluster_health_card.isHidden()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
