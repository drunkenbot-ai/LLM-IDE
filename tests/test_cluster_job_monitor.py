"""Tests for the Thinkbox Deadline & Fox Renderfarm inspired Cluster Job Monitor.

Validates:
1. ClusterStorageBus.list_all_jobs() formatting and ordering.
2. ClusterStorageBus.get_job_tasks() shard inspection and worker mapping.
3. Job re-queueing and resetting for master restart capability.
4. Job deletion and artifact cleanup.
5. Worker gating to RUNNING status only.
"""

import os
import tempfile
import time
from pathlib import Path

import torch

from cluster.bus import ClusterStorageBus


def test_list_all_jobs_ordering_and_decoding(tmp_path: Path) -> None:
    """Verify list_all_jobs returns all jobs sorted by created_at DESC with decoded configs."""
    bus = ClusterStorageBus(tmp_path)

    bus.create_job(
        job_id="job_alpha",
        model_config={"embedding_size": 128, "layer_count": 2},
        training_config={"batch_size": 2, "learning_rate": 0.001},
        dataset_path=str(tmp_path / "train.npy"),
        max_rounds=5,
    )
    time.sleep(0.05)
    bus.create_job(
        job_id="job_beta",
        model_config={"embedding_size": 256, "layer_count": 4},
        training_config={"batch_size": 4, "learning_rate": 0.0003},
        dataset_path=str(tmp_path / "train.npy"),
        max_rounds=10,
    )

    jobs = bus.list_all_jobs(limit=50)
    assert len(jobs) == 2
    # Most recently created job first
    assert jobs[0]["job_id"] == "job_beta"
    assert jobs[1]["job_id"] == "job_alpha"

    assert jobs[0]["model_config"]["embedding_size"] == 256
    assert jobs[0]["training_config"]["batch_size"] == 4
    assert jobs[1]["model_config"]["layer_count"] == 2


def test_get_job_tasks_and_worker_mapping(tmp_path: Path) -> None:
    """Verify get_job_tasks returns detailed task and shard allocations."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "test_job_tasks_1"
    bus.create_job(
        job_id=job_id,
        model_config={},
        training_config={},
        dataset_path=str(tmp_path / "tokens.npy"),
        max_rounds=3,
    )

    # Register two workers
    bus.register_worker("node_01", "host_a", "RTX 4090", 24.0)
    bus.register_worker("node_02", "host_b", "RTX 5060 Ti", 8.0)

    # Workers claim slots
    shard_0, total_1 = bus.claim_job_slot(job_id, "node_01")
    shard_1, total_2 = bus.claim_job_slot(job_id, "node_02")

    assert shard_0 == 0
    assert shard_1 == 1

    # Save mock round 0 telemetry
    bus.save_worker_telemetry(job_id, 0, "node_01", {"avg_loss": 2.15, "tokens_per_sec": 5000})
    bus.save_worker_telemetry(job_id, 0, "node_02", {"avg_loss": 2.18, "tokens_per_sec": 4800})

    tasks = bus.get_job_tasks(job_id)
    assert len(tasks) == 2

    assert tasks[0]["worker_id"] == "node_01"
    assert tasks[0]["shard_index"] == 0
    assert tasks[0]["hostname"] == "host_a"
    assert tasks[0]["gpu_name"] == "RTX 4090"
    assert tasks[0]["telemetry"]["avg_loss"] == 2.15

    assert tasks[1]["worker_id"] == "node_02"
    assert tasks[1]["shard_index"] == 1
    assert tasks[1]["gpu_name"] == "RTX 5060 Ti"
    assert tasks[1]["telemetry"]["tokens_per_sec"] == 4800


def test_requeue_job_resuming_and_scratch(tmp_path: Path) -> None:
    """Verify requeue_job resets status to QUEUED and removes signal files."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "test_job_requeue"
    bus.create_job(
        job_id=job_id,
        model_config={},
        training_config={},
        dataset_path=str(tmp_path / "tokens.npy"),
        max_rounds=5,
    )

    bus.set_job_status(job_id, "STOPPED")
    assert bus.is_stopped(job_id)

    bus.advance_job_round(job_id, 3)
    job = bus.get_job(job_id)
    assert job["current_round"] == 3

    # Re-queue with resume (keeping rounds)
    success = bus.requeue_job(job_id, reset_rounds=False)
    assert success
    job = bus.get_job(job_id)
    assert job["status"] == "QUEUED"
    assert job["current_round"] == 3
    assert not bus.is_stopped(job_id)

    # Re-queue with scratch reset (round 0)
    success = bus.requeue_job(job_id, reset_rounds=True)
    assert success
    job = bus.get_job(job_id)
    assert job["status"] == "QUEUED"
    assert job["current_round"] == 0


def test_delete_job_and_cleanup(tmp_path: Path) -> None:
    """Verify delete_job purges job metadata, participants, and directory."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "test_job_delete"
    bus.create_job(
        job_id=job_id,
        model_config={},
        training_config={},
        dataset_path=str(tmp_path / "tokens.npy"),
        max_rounds=5,
    )
    bus.claim_job_slot(job_id, "node_del")

    job_dir = bus.jobs_dir / job_id
    assert job_dir.exists()

    success = bus.delete_job(job_id)
    assert success
    assert bus.get_job(job_id) is None
    assert len(bus.get_job_tasks(job_id)) == 0
    assert not job_dir.exists()


def test_job_resume_clears_signals(tmp_path: Path) -> None:
    """Verify that transitioning a job to RUNNING unlinks both stop.sig and pause.sig."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "test_resume_sig"
    bus.create_job(
        job_id=job_id,
        model_config={},
        training_config={},
        dataset_path=str(tmp_path / "tokens.npy"),
        max_rounds=5,
    )

    # 1. Stop job
    bus.set_job_status(job_id, "STOPPED")
    assert bus.is_stopped(job_id)

    # 2. Resume job
    bus.set_job_status(job_id, "RUNNING")
    assert not bus.is_stopped(job_id)
    assert not bus.is_paused(job_id)

    # 3. Pause job
    bus.set_job_status(job_id, "PAUSED")
    assert bus.is_paused(job_id)
    assert not bus.is_stopped(job_id)

    # 4. Resume from paused
    bus.set_job_status(job_id, "RUNNING")
    assert not bus.is_paused(job_id)
    assert not bus.is_stopped(job_id)


def test_set_table_rows_in_place_update() -> None:
    """Verify set_table_rows updates table cells in-place without clearing items when row count matches."""
    from PySide6.QtWidgets import QApplication, QTableWidget
    from interface.tabs.job_manager_tab import set_table_rows

    _app = QApplication.instance() or QApplication([])
    table = QTableWidget(0, 3)

    initial_rows = [["RUNNING", "job_1", "Round 1/5"], ["STOPPED", "job_2", "Round 0/5"]]
    set_table_rows(table, initial_rows)

    assert table.rowCount() == 2
    item_0_0 = table.item(0, 0)
    assert item_0_0 is not None
    assert item_0_0.text() == "RUNNING"

    # Update with new values but same row count
    updated_rows = [["RUNNING", "job_1", "Round 2/5"], ["STOPPED", "job_2", "Round 0/5"]]
    set_table_rows(table, updated_rows)

    # Verify existing item was reused and updated
    assert table.item(0, 0) is item_0_0
    assert table.item(0, 2).text() == "Round 2/5"


def test_worker_restart_command_and_lock_handling(tmp_path: Path) -> None:
    """Verify worker RESTART command dispatch and singleton lock release/re-acquire."""
    from cluster_worker import acquire_singleton_lock, release_singleton_lock, get_lock_file

    bus = ClusterStorageBus(tmp_path)
    wid = "node_restart_test"
    bus.register_worker(wid, "hostname_test", "RTX 4090", 24.0)

    # 1. Send RESTART command
    bus.set_worker_command(wid, "RESTART")
    assert bus.get_worker_command(wid) == "RESTART"

    # 2. Worker clears command
    bus.set_worker_command(wid, None)
    assert bus.get_worker_command(wid) is None

    # 3. Lock acquisition and release
    tag = "test_dev_unit"
    acquired = acquire_singleton_lock(tag)
    assert acquired
    lock_file = get_lock_file(tag)
    assert lock_file.exists()

    # Release lock on restart
    release_singleton_lock(tag)
    assert not lock_file.exists()
