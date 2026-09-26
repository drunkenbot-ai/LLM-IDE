from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QWidget,
)

from cluster.bus import ClusterStorageBus
from interface.screens.cluster_screen import ClusterScreenMixin


class FakeClusterHost(ClusterScreenMixin, QWidget):
    """Minimal test host embedding ClusterScreenMixin with mock widgets."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__()
        self.tmp_path = tmp_path
        self.shared_dir = tmp_path / "shared"
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.bus = ClusterStorageBus(self.shared_dir)

        # Training Tab widgets
        self.train_data_dir = QLineEdit(str(tmp_path / "dataset"))
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(2)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(2)

        self.train_context_length = QSpinBox()
        self.train_context_length.setRange(16, 8192)
        self.train_context_length.setValue(1024)

        # Integrated Cluster settings on Training tab
        self.train_cluster_sync_steps = QSpinBox()
        self.train_cluster_sync_steps.setRange(10, 5000)
        self.train_cluster_sync_steps.setValue(250)

        self.train_cluster_max_rounds = QSpinBox()
        self.train_cluster_max_rounds.setRange(1, 100000)
        self.train_cluster_max_rounds.setValue(10)

        self.train_cluster_auto_sync = QCheckBox("Auto-sync from Target Epochs")
        self.train_cluster_auto_sync.setChecked(True)

        self.train_cluster_plan_label = QLabel()

        # Cluster Tab (Job Manager) widgets
        self.cluster_shared_dir = QLineEdit(str(self.shared_dir))
        self.cluster_sync_steps = QSpinBox()
        self.cluster_sync_steps.setRange(10, 5000)
        self.cluster_sync_steps.setValue(250)

        self.cluster_max_rounds = QSpinBox()
        self.cluster_max_rounds.setRange(1, 100000)
        self.cluster_max_rounds.setValue(10)

        self.cluster_auto_sync_epochs = QCheckBox("Auto-sync from Epochs")
        self.cluster_auto_sync_epochs.setChecked(True)

        self.cluster_sync_info_label = QLabel()
        self.cluster_min_workers = QSpinBox()
        self.cluster_min_workers.setRange(1, 64)
        self.cluster_min_workers.setValue(2)

        # Control buttons for Neural Forge and Fine Tuning
        self.train_button = QPushButton("Start Training")
        self.stop_training_button = QPushButton("Stop")
        self.stop_training_button.setEnabled(False)
        self.fine_tune_button = QPushButton("Start Fine-Tune")
        self.stop_fine_tune_button = QPushButton("Stop")
        self.stop_fine_tune_button.setEnabled(False)

        self.project_state = QLabel("Idle")
        self.train_status = QLabel("Training: idle")
        self.cluster_status_label = QLabel("Status: Idle")
        self.cluster_round_label = QLabel("Round: 0/0")
        self.training_step_metric = QLabel("Step: -")
        self.training_epoch_metric = QLabel("Epoch: -")
        self.training_progress = QProgressBar()
        self.training_health_metric = QLabel("Health: -")
        self.training_lr_metric = QLabel("LR: -")

    def _get_cluster_bus(self) -> ClusterStorageBus:
        return self.bus

    def _log_cluster_event(self, msg: str) -> None:
        pass

    def refresh_cluster_status(self) -> None:
        pass

    def refresh_job_manager_tab(self) -> None:
        pass

    def _start_cluster_coordinator(self, bus, job_id) -> None:
        pass


@pytest.fixture
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_calculate_cluster_rounds_from_dataset_summary(qt_app, tmp_path: Path) -> None:
    """Verify calculate_cluster_rounds_from_epochs accurately translates epochs to rounds."""
    host = FakeClusterHost(tmp_path)

    # Prepare dataset folder with dataset_summary.json (1,000,000 tokens)
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    summary_file = ds_dir / "dataset_summary.json"
    summary_file.write_text(json.dumps({"train_token_count": 1_000_000}), encoding="utf-8")

    # 1 active worker in bus
    host.bus.register_worker("node_01", hostname="mumws01", gpu_name="RTX 4090", vram_gb=24.0)
    host.bus.heartbeat("node_01", status="IDLE")

    # Epochs=2, Batch=2, Context=1024, K=250 -> 1 worker * 250 * 2 * 1024 = 512,000 tokens/round
    # Total tokens = 2 * 1,000,000 = 2,000,000
    # Needed rounds = ceil(2,000,000 / 512,000) = 4 rounds
    plan = host.calculate_cluster_rounds_from_epochs(quiet=True)
    assert plan is not None
    assert plan["total_tokens"] == 1_000_000
    assert plan["epochs"] == 2
    assert plan["active_workers"] == 1
    assert plan["tokens_per_round"] == 512_000
    assert plan["needed_rounds"] == 4
    assert plan["steps_per_worker"] == 1000

    assert host.train_cluster_max_rounds.value() == 4
    assert host.cluster_max_rounds.value() == 4
    assert "4 Rounds" in host.train_cluster_plan_label.text()
    assert "4 Rounds" in host.cluster_sync_info_label.text()


def test_calculate_cluster_rounds_from_npy_fallback(qt_app, tmp_path: Path) -> None:
    """Verify calculate_cluster_rounds_from_epochs falls back to train_tokens.npy."""
    host = FakeClusterHost(tmp_path)

    # Prepare dataset folder with train_tokens.npy only
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    npy_file = ds_dir / "train_tokens.npy"
    arr = np.arange(500_000, dtype=np.uint16)
    np.save(str(npy_file), arr)

    # 2 active workers in bus
    host.bus.register_worker("node_01", hostname="mumws01", gpu_name="RTX 4090", vram_gb=24.0)
    host.bus.register_worker("node_02", hostname="mumws02", gpu_name="RTX 4090", vram_gb=24.0)
    host.bus.heartbeat("node_01", status="READY")
    host.bus.heartbeat("node_02", status="READY")

    # Epochs=1, Batch=2, Context=1024, K=250 -> 2 workers * 250 * 2 * 1024 = 1,024,000 tokens/round
    # Total tokens = 500,000
    # Needed rounds = ceil(500,000 / 1,024,000) = 1 round
    host.epochs.setValue(1)
    plan = host.calculate_cluster_rounds_from_epochs(quiet=True)
    assert plan is not None
    assert plan["total_tokens"] == 500_000
    assert plan["active_workers"] == 2
    assert plan["tokens_per_round"] == 1_024_000
    assert plan["needed_rounds"] == 1


def test_reactive_two_way_synchronization(qt_app, tmp_path: Path) -> None:
    """Verify bidirectional binding and reactive updates between Training Tab and Cluster controls."""
    host = FakeClusterHost(tmp_path)

    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "dataset_summary.json").write_text(json.dumps({"train_token_count": 2_000_000}), encoding="utf-8")

    host.bus.register_worker("worker_alpha", hostname="mumws01", gpu_name="RTX 4090", vram_gb=24.0)
    host.bus.register_worker("worker_beta", hostname="mumws02", gpu_name="RTX 4090", vram_gb=24.0)
    host.bus.heartbeat("worker_alpha", status="IDLE")
    host.bus.heartbeat("worker_beta", status="IDLE")

    # Wire the reactive bindings
    host.connect_cluster_training_sync()

    # Initial state: 2 workers, K=250, B=2, C=1024 -> 1,024,000 tokens/round
    # Epochs=2 -> 4,000,000 tokens -> ceil(4,000,000 / 1,024,000) = 4 rounds
    assert host.train_cluster_max_rounds.value() == 4
    assert host.cluster_max_rounds.value() == 4

    # 1. Changing Epochs on Training tab should reactively update rounds
    host.epochs.setValue(5)
    # 5 * 2,000,000 = 10,000,000 tokens -> ceil(10,000,000 / 1,024,000) = 10 rounds
    assert host.train_cluster_max_rounds.value() == 10
    assert host.cluster_max_rounds.value() == 10
    assert "10 Rounds" in host.train_cluster_plan_label.text()
    assert "10 Rounds" in host.cluster_sync_info_label.text()

    # 2. Changing Sync Steps (K) on Training tab should sync to Cluster tab and recalculate
    host.train_cluster_sync_steps.setValue(500)
    assert host.cluster_sync_steps.value() == 500
    # K=500 -> 2 workers * 500 * 2 * 1024 = 2,048,000 tokens/round
    # 10,000,000 tokens -> ceil(10,000,000 / 2,048,000) = 5 rounds
    assert host.train_cluster_max_rounds.value() == 5
    assert host.cluster_max_rounds.value() == 5

    # 3. Changing Sync Steps (K) on Cluster tab should sync to Training tab and recalculate
    host.cluster_sync_steps.setValue(125)
    assert host.train_cluster_sync_steps.value() == 125
    # K=125 -> 2 workers * 125 * 2 * 1024 = 512,000 tokens/round
    # 10,000,000 tokens -> ceil(10,000,000 / 512,000) = 20 rounds
    assert host.train_cluster_max_rounds.value() == 20
    assert host.cluster_max_rounds.value() == 20

    # 4. Changing Batch size on Training tab should recalculate rounds
    host.batch_size.setValue(4)
    # Batch=4 -> 2 * 125 * 4 * 1024 = 1,024,000 tokens/round -> 10 rounds
    assert host.train_cluster_max_rounds.value() == 10
    assert host.cluster_max_rounds.value() == 10

    # 5. Unticking Auto-sync should preserve manual rounds setting
    host.train_cluster_auto_sync.setChecked(False)
    assert not host.cluster_auto_sync_epochs.isChecked()

    host.train_cluster_max_rounds.setValue(42)
    assert host.cluster_max_rounds.value() == 42

    # Changing epochs now should NOT overwrite manual rounds
    host.epochs.setValue(1)
    assert host.train_cluster_max_rounds.value() == 42
    assert host.cluster_max_rounds.value() == 42

    # Re-enabling Auto-sync from Cluster tab should immediately recalculate
    host.cluster_auto_sync_epochs.setChecked(True)
    assert host.train_cluster_auto_sync.isChecked()
    # Epochs=1, 2,000,000 tokens / 1,024,000 = 2 rounds
    assert host.train_cluster_max_rounds.value() == 2
    assert host.cluster_max_rounds.value() == 2


def test_cluster_start_and_resume_button_disabling_and_min_workers_clamping(qt_app, tmp_path: Path) -> None:
    """Verify that starting or resuming a cluster job disables start buttons on both Neural Forge
    and Fine-Tuning tabs, enables stop buttons, and auto-clamps min_workers when only 1 worker is online.
    """
    host = FakeClusterHost(tmp_path)

    # 1. Register only 1 active online worker
    host.bus.register_worker("worker_solo", hostname="mumws01", gpu_name="RTX 4090", vram_gb=24.0)
    host.bus.heartbeat("worker_solo", status="IDLE")

    # Verify initial button states
    assert host.train_button.isEnabled()
    assert host.fine_tune_button.isEnabled()
    assert not host.stop_training_button.isEnabled()
    assert not host.stop_fine_tune_button.isEnabled()

    # Settings specifies Min Workers = 2, but only 1 worker is online!
    host.cluster_min_workers.setValue(2)

    # Create a job with min_workers = 2
    job_id = "test_cluster_job_buttons"
    host.bus.create_job(
        job_id=job_id,
        model_config={"vocab_size": 256, "n_embd": 64, "n_layer": 2, "n_head": 2, "block_size": 128},
        training_config={"batch_size": 2, "learning_rate": 1e-3},
        dataset_path=str(tmp_path / "dataset"),
        max_rounds=5,
        sync_interval_steps=100,
        min_workers=2,
    )

    # 2. Resume the cluster job from the Cluster page
    host.resume_cluster_job(job_id=job_id)

    # Both Start buttons MUST be disabled, and both Stop buttons MUST be enabled!
    assert not host.train_button.isEnabled()
    assert host.train_button.text() == "Training..."
    assert not host.fine_tune_button.isEnabled()
    assert host.fine_tune_button.text() == "Fine-Tuning..."
    assert host.stop_training_button.isEnabled()
    assert host.stop_fine_tune_button.isEnabled()

    # Min workers must be automatically clamped from 2 to 1 in both UI and SQLite!
    assert host.cluster_min_workers.value() == 1
    job_record = host.bus.get_job(job_id)
    assert job_record["min_workers"] == 1

    # 3. Simulate background telemetry polling with st == "RUNNING"
    host._apply_cluster_telemetry(
        workers=host.bus.list_workers(),
        active_job=job_record,
    )
    # Buttons must stay disabled while running
    assert not host.train_button.isEnabled()
    assert not host.fine_tune_button.isEnabled()
    assert host.stop_training_button.isEnabled()
    assert host.stop_fine_tune_button.isEnabled()

    # 4. Stop the cluster job
    host.stop_cluster_job()

    # Buttons must revert: Start buttons enabled, Stop buttons disabled
    assert host.train_button.isEnabled()
    assert host.train_button.text() == "Start Training"
    assert host.fine_tune_button.isEnabled()
    assert host.fine_tune_button.text() == "Start Fine-Tune"
    assert not host.stop_training_button.isEnabled()
    assert not host.stop_fine_tune_button.isEnabled()


def test_coordinator_thread_lifecycle_and_stop_safety(qt_app, tmp_path: Path) -> None:
    """Verify ClusterCoordinatorThread starts, responds to stop signals, and cleanly exits without QThread crashes."""
    import time
    from interface.screens.cluster_screen import ClusterCoordinatorThread
    from cluster.bus import ClusterStorageBus

    bus = ClusterStorageBus(tmp_path)
    job_id = "job_coord_test"
    bus.create_job(
        job_id=job_id,
        model_config={"vocab_size": 32, "context_length": 8, "embedding_size": 16, "head_count": 2, "layer_count": 2},
        training_config={"learning_rate": 1e-3, "batch_size": 2},
        dataset_path=str(tmp_path / "train.npy"),
        max_rounds=5,
    )
    bus.set_job_status(job_id, "RUNNING")

    coord_thread = ClusterCoordinatorThread(bus, job_id, poll_interval=0.1)
    telemetry_received = []
    coord_thread.round_telemetry_ready.connect(lambda t: telemetry_received.append(t))

    finished_received = []
    coord_thread.job_finished.connect(lambda jid, success: finished_received.append((jid, success)))

    coord_thread.start()
    assert coord_thread.isRunning() is True

    # Allow it a brief moment to enter loop
    time.sleep(0.2)

    # Now stop the thread
    coord_thread.stop()
    clean_exit = coord_thread.wait(timeout_ms=3000)

    assert clean_exit is True
    assert coord_thread.isRunning() is False

    qt_app.processEvents()
    assert len(finished_received) == 1
    assert finished_received[0][0] == job_id

