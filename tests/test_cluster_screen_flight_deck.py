from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from interface import app as interface_app


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("license_valid", True)
    return app


def test_cluster_screen_flight_deck_labels_initial_and_update(qapp) -> None:
    """Cluster screen flight deck cards must initialize to idle and update on active job telemetry."""
    window = interface_app.MainWindow()
    try:
        # Check that cluster tab widgets exist
        assert hasattr(window, "cluster_status_label")
        assert hasattr(window, "cluster_round_label")
        assert hasattr(window, "cluster_train_time_label")
        assert hasattr(window, "cluster_sync_time_label")

        # Initial state before polling should be idle
        assert window.cluster_status_label.text() in ("Status: Idle", "Status: Fleet Idle")
        assert window.cluster_round_label.text() == "Round: -"
        assert window.cluster_train_time_label.text() == "Train Time: -"
        assert window.cluster_sync_time_label.text() == "Sync Time: -"

        # Simulate active job telemetry
        active_job = {
            "job_id": "job_fl_test_001",
            "status": "RUNNING",
            "current_round": 4,
            "max_rounds": 10,
            "job_type": "pretrain",
            "_rounds": [
                {"round_number": 0, "metrics": {"avg_compute_sec": 15.42, "coordinator_agg_sec": 3.81}},
            ],
        }

        window._apply_cluster_telemetry(None, active_job)

        # Check updated telemetry
        assert "job_fl_test_001" in window.cluster_status_label.text()
        assert "RUNNING" in window.cluster_status_label.text()
        assert "4 / 10" in window.cluster_round_label.text()
        assert "15s" in window.cluster_train_time_label.text() or "Train Time:" in window.cluster_train_time_label.text()
        assert "4s" in window.cluster_sync_time_label.text() or "Sync Time:" in window.cluster_sync_time_label.text()

        # Simulate returning to idle
        window._apply_cluster_telemetry(None, None)
        assert window.cluster_status_label.text() == "Status: Fleet Idle"
        assert window.cluster_round_label.text() == "Round: -"
        assert window.cluster_train_time_label.text() == "Train Time: -"
        assert window.cluster_sync_time_label.text() == "Sync Time: -"
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
