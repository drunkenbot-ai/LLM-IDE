"""Comprehensive unit and integration tests for port-blocked distributed Local SGD cluster.

Tests cover:
1. SQLite WAL central storage bus (ephemeral connection, CRUD, worker tracking).
2. Cooperative signal detection (pause.sig, stop.sig) without network ports.
3. Atomic tensor checkpoint serialization and ready-flag triggers.
4. Token dataset sharding and aligned boundary calculations.
5. Arithmetic state dict averaging (vectorized floating point, integer buffer preservation).
6. Coordinator multi-worker round synchronization and straggler timeout handling.
7. End-to-end Local SGD training round with workers on disjoint shards.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cluster.bus import ClusterStorageBus
from cluster.coordinator import ClusterCoordinator, average_state_dicts
from cluster.sharding import ShardedTokenDataset, compute_shard_boundaries
from cluster.worker import ClusterWorker, build_model_from_config, get_hardware_info


def test_hardware_detection() -> None:
    """Verify hardware discovery returns valid device string, GPU name, and VRAM."""
    device, gpu_name, vram_gb = get_hardware_info("cpu")
    assert device == "cpu"
    assert isinstance(gpu_name, str) and len(gpu_name) > 0
    assert isinstance(vram_gb, float)


def test_storage_bus_crud_and_heartbeat(tmp_path: Path) -> None:
    """Verify worker registration, status updates, and heartbeat expiration."""
    bus = ClusterStorageBus(tmp_path)

    # Register workers
    bus.register_worker("node_01", "host_a", "NVIDIA RTX 4090", 24.0)
    bus.register_worker("node_02", "host_b", "NVIDIA RTX 3090", 24.0)

    workers = bus.list_workers()
    assert len(workers) == 2
    w1 = next(w for w in workers if w["worker_id"] == "node_01")
    assert w1["gpu_name"] == "NVIDIA RTX 4090"
    assert w1["is_online"] is True
    assert w1["status"] in ("IDLE", "INITIALIZING")

    # Update heartbeat
    bus.heartbeat("node_01", status="TRAINING", current_job_id="job_abc")
    w1_updated = next(w for w in bus.list_workers() if w["worker_id"] == "node_01")
    assert w1_updated["status"] == "TRAINING"
    assert w1_updated["current_job_id"] == "job_abc"


def test_storage_bus_job_lifecycle_and_signals(tmp_path: Path) -> None:
    """Verify job creation, status transitions, and cooperative signal files."""
    bus = ClusterStorageBus(tmp_path)

    job_id = "job_test_01"
    model_cfg = {"vocab_size": 100, "context_length": 32, "embedding_size": 64, "head_count": 2, "layer_count": 2}
    training_cfg = {"learning_rate": 1e-3, "batch_size": 2}

    bus.create_job(
        job_id=job_id,
        model_config=model_cfg,
        training_config=training_cfg,
        dataset_path=str(tmp_path / "train_tokens.npy"),
        max_rounds=5,
        sync_interval_steps=50,
        min_workers=2,
        sync_timeout_seconds=30.0,
    )

    job = bus.get_job(job_id)
    assert job is not None
    assert job["job_id"] == job_id
    assert job["status"] == "QUEUED"
    assert job["model_config"]["vocab_size"] == 100

    # Shard slot claiming
    shard_0, total_0 = bus.claim_job_slot(job_id, "node_01")
    assert shard_0 == 0
    assert total_0 == 1

    shard_1, total_1 = bus.claim_job_slot(job_id, "node_02")
    assert shard_1 == 1
    assert total_1 == 2

    # Re-claiming returns existing slot
    shard_0_again, total_re = bus.claim_job_slot(job_id, "node_01")
    assert shard_0_again == 0

    # Signal handling: PAUSE
    bus.set_job_status(job_id, "PAUSED")
    assert bus.is_paused(job_id) is True
    assert bus.is_stopped(job_id) is False

    # RESUME
    bus.set_job_status(job_id, "RUNNING")
    assert bus.is_paused(job_id) is False

    # STOP
    bus.set_job_status(job_id, "STOPPED")
    assert bus.is_stopped(job_id) is True


def test_storage_bus_atomic_tensor_checkpoints(tmp_path: Path) -> None:
    """Verify atomic saving and loading of worker and global weights."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "job_tensors"

    weights_w1 = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    weights_w2 = {"weight": torch.tensor([4.0, 5.0, 6.0])}

    assert bus.is_worker_weights_ready(job_id, 0, "node_01") is False

    bus.save_worker_weights(job_id, 0, "node_01", weights_w1)
    bus.save_worker_weights(job_id, 0, "node_02", weights_w2)

    assert bus.is_worker_weights_ready(job_id, 0, "node_01") is True
    assert bus.is_worker_weights_ready(job_id, 0, "node_02") is True

    ready_list = bus.get_ready_workers_for_round(job_id, 0)
    assert set(ready_list) == {"node_01", "node_02"}

    loaded_w1 = bus.load_worker_weights(job_id, 0, "node_01")
    assert torch.equal(loaded_w1["weight"], weights_w1["weight"])

    # Global weights
    global_weights = {"weight": torch.tensor([2.5, 3.5, 4.5])}
    assert bus.is_global_weights_ready(job_id, 0) is False
    bus.save_global_weights(job_id, 0, global_weights)
    assert bus.is_global_weights_ready(job_id, 0) is True

    # Confirm ready workers list strictly excludes global_model
    ready_list_after_global = bus.get_ready_workers_for_round(job_id, 0)
    assert "global_model" not in ready_list_after_global
    assert set(ready_list_after_global) == {"node_01", "node_02"}

    loaded_global = bus.load_global_weights(job_id, 0)
    assert torch.equal(loaded_global["weight"], global_weights["weight"])


def test_hybrid_is_stopped_and_paused_without_signal_files(tmp_path: Path) -> None:
    """Verify is_stopped and is_paused return True when database status is STOPPED/PAUSED even if .sig file is missing."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "job_db_only"
    bus.create_job(
        job_id=job_id,
        model_config={},
        training_config={},
        dataset_path="dummy.npy",
    )

    # Delete signal file to simulate SMB metadata caching or DB-only status updates
    stop_sig = bus.jobs_dir / job_id / "signals" / "stop.sig"
    pause_sig = bus.jobs_dir / job_id / "signals" / "pause.sig"

    bus.set_job_status(job_id, "STOPPED")
    if stop_sig.exists():
        stop_sig.unlink()
    assert not stop_sig.exists()
    assert bus.is_stopped(job_id) is True  # Successfully detected via SQLite fallback!

    bus.set_job_status(job_id, "PAUSED")
    if pause_sig.exists():
        pause_sig.unlink()
    assert not pause_sig.exists()
    assert bus.is_paused(job_id) is True  # Successfully detected via SQLite fallback!


def test_shard_boundary_calculations() -> None:
    """Verify non-overlapping token slice boundaries aligned to context_length."""
    total_tokens = 2048
    context_length = 256

    # 1 worker: full dataset
    s0, e0 = compute_shard_boundaries(total_tokens, 0, 1, context_length)
    assert (s0, e0) == (0, 2048)

    # 2 workers: each gets 4 windows of 256 tokens = 1024 tokens each
    s0, e0 = compute_shard_boundaries(total_tokens, 0, 2, context_length)
    s1, e1 = compute_shard_boundaries(total_tokens, 1, 2, context_length)
    assert s0 == 0 and e0 == 1024
    assert s1 == 1024 and e1 == 2048
    assert e0 == s1  # Seamless boundary, no overlap

    # 3 workers with remainder tokens: last worker captures remainder
    s0, e0 = compute_shard_boundaries(total_tokens, 0, 3, context_length)
    s1, e1 = compute_shard_boundaries(total_tokens, 1, 3, context_length)
    s2, e2 = compute_shard_boundaries(total_tokens, 2, 3, context_length)
    assert s0 == 0
    assert e0 == s1
    assert e1 == s2
    assert e2 == total_tokens


def test_sharded_token_dataset() -> None:
    """Verify ShardedTokenDataset produces valid next-token prediction pairs."""
    tokens = np.arange(1000, dtype=np.int64)
    context_length = 32

    ds_w0 = ShardedTokenDataset(tokens, context_length=context_length, shard_index=0, total_shards=2)
    ds_w1 = ShardedTokenDataset(tokens, context_length=context_length, shard_index=1, total_shards=2)

    assert len(ds_w0) > 0
    assert len(ds_w1) > 0

    x0, y0 = ds_w0[0]
    assert x0.shape == (context_length,)
    assert y0.shape == (context_length,)
    # Verify next-token shift
    assert torch.equal(y0[:-1], x0[1:])

    # Verify worker 0 and worker 1 samples are disjoint
    x1, y1 = ds_w1[0]
    assert not torch.equal(x0, x1)


def test_arithmetic_weight_averaging() -> None:
    """Verify vectorized averaging across PyTorch state dicts."""
    state_a = {
        "fc.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "fc.bias": torch.tensor([10.0, 20.0]),
        "buffer_int": torch.tensor([1, 2, 3], dtype=torch.long),
    }
    state_b = {
        "fc.weight": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        "fc.bias": torch.tensor([30.0, 40.0]),
        "buffer_int": torch.tensor([9, 9, 9], dtype=torch.long),
    }

    averaged = average_state_dicts([state_a, state_b])

    expected_weight = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    expected_bias = torch.tensor([20.0, 30.0])

    assert torch.allclose(averaged["fc.weight"], expected_weight)
    assert torch.allclose(averaged["fc.bias"], expected_bias)
    # Non-float buffer preserved from first state dict
    assert torch.equal(averaged["buffer_int"], state_a["buffer_int"])


def test_coordinator_wait_and_average(tmp_path: Path) -> None:
    """Verify coordinator completes a round when workers deposit weights."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "job_coord"

    bus.create_job(
        job_id=job_id,
        model_config={},
        training_config={},
        dataset_path="dummy.npy",
        max_rounds=3,
        min_workers=2,
        sync_timeout_seconds=5.0,
    )
    bus.claim_job_slot(job_id, "worker_1")
    bus.claim_job_slot(job_id, "worker_2")

    # Workers deposit weights for round 0
    w1_state = {"param": torch.tensor([2.0, 4.0])}
    w2_state = {"param": torch.tensor([4.0, 8.0])}
    bus.save_worker_weights(job_id, 0, "worker_1", w1_state)
    bus.save_worker_weights(job_id, 0, "worker_2", w2_state)

    coordinator = ClusterCoordinator(bus=bus, job_id=job_id)
    global_model = coordinator.wait_and_average_round(round_num=0, poll_interval_seconds=0.1)

    assert global_model is not None
    assert torch.allclose(global_model["param"], torch.tensor([3.0, 6.0]))

    # Confirm global model is ready on bus
    assert bus.is_global_weights_ready(job_id, 0) is True
    # Confirm round was advanced in database
    job = bus.get_job(job_id)
    assert job["current_round"] == 1


def test_coordinator_straggler_timeout(tmp_path: Path) -> None:
    """Verify coordinator averages available models when min_workers is reached and timeout expires."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "job_straggler"

    bus.create_job(
        job_id=job_id,
        model_config={},
        training_config={},
        dataset_path="dummy.npy",
        max_rounds=3,
        min_workers=1,  # Minimum 1 worker needed
        sync_timeout_seconds=0.3,  # Short timeout for testing
    )
    bus.claim_job_slot(job_id, "fast_worker")
    bus.claim_job_slot(job_id, "slow_straggler")  # Registered but won't deposit

    # Fast worker deposits
    bus.save_worker_weights(job_id, 0, "fast_worker", {"param": torch.tensor([10.0])})

    coordinator = ClusterCoordinator(bus=bus, job_id=job_id)
    start_t = time.time()
    global_model = coordinator.wait_and_average_round(round_num=0, poll_interval_seconds=0.05)
    elapsed = time.time() - start_t

    assert global_model is not None
    assert torch.equal(global_model["param"], torch.tensor([10.0]))
    assert elapsed >= 0.3  # Waited for straggler timeout before proceeding


def test_end_to_end_cluster_local_sgd_round(tmp_path: Path) -> None:
    """Simulate 2 workers executing 1 complete Local SGD round on disjoint shards."""
    bus = ClusterStorageBus(tmp_path)

    # 1. Prepare dummy dataset
    token_data = np.random.randint(0, 100, size=2048, dtype=np.int64)
    dataset_file = tmp_path / "train_tokens.npy"
    np.save(dataset_file, token_data)

    model_cfg = {
        "vocab_size": 100,
        "context_length": 32,
        "embedding_size": 32,
        "head_count": 2,
        "layer_count": 1,
    }
    training_cfg = {
        "learning_rate": 1e-3,
        "batch_size": 2,
    }

    job_id = "e2e_job"
    bus.create_job(
        job_id=job_id,
        model_config=model_cfg,
        training_config=training_cfg,
        dataset_path=str(dataset_file),
        max_rounds=1,
        sync_interval_steps=3,  # 3 fast local steps
        min_workers=2,
    )
    bus.set_job_status(job_id, "RUNNING")

    # 2. Instantiate 2 workers
    worker_1 = ClusterWorker(bus=bus, worker_id="node_e2e_1", device="cpu", heartbeat_interval=1.0)
    worker_2 = ClusterWorker(bus=bus, worker_id="node_e2e_2", device="cpu", heartbeat_interval=1.0)

    try:
        # Both workers claim slot
        shard_1, total_1 = bus.claim_job_slot(job_id, worker_1.worker_id)
        shard_2, total_2 = bus.claim_job_slot(job_id, worker_2.worker_id)
        assert (shard_1, shard_2) == (0, 1)

        # Worker 1 trains round 0
        model_1 = build_model_from_config(model_cfg, "cpu")
        opt_1 = torch.optim.SGD(model_1.parameters(), lr=0.01)
        ds_1 = ShardedTokenDataset(token_data, context_length=32, shard_index=shard_1, total_shards=2)
        dl_1 = torch.utils.data.DataLoader(ds_1, batch_size=2)
        loss_1, _ = worker_1.run_training_round(
            job_id, 0, model_1, opt_1, dl_1, iter(dl_1), steps_per_round=3
        )
        assert isinstance(loss_1, float)
        bus.save_worker_weights(job_id, 0, worker_1.worker_id, model_1.state_dict())

        # Worker 2 trains round 0
        model_2 = build_model_from_config(model_cfg, "cpu")
        opt_2 = torch.optim.SGD(model_2.parameters(), lr=0.01)
        ds_2 = ShardedTokenDataset(token_data, context_length=32, shard_index=shard_2, total_shards=2)
        dl_2 = torch.utils.data.DataLoader(ds_2, batch_size=2)
        loss_2, _ = worker_2.run_training_round(
            job_id, 0, model_2, opt_2, dl_2, iter(dl_2), steps_per_round=3
        )
        assert isinstance(loss_2, float)
        bus.save_worker_weights(job_id, 0, worker_2.worker_id, model_2.state_dict())

        # Coordinator averages round 0
        coordinator = ClusterCoordinator(bus=bus, job_id=job_id)
        global_state = coordinator.wait_and_average_round(round_num=0, poll_interval_seconds=0.05)
        assert global_state is not None
        assert bus.is_global_weights_ready(job_id, 0) is True

    finally:
        worker_1.stop()
        worker_2.stop()


def test_standalone_worker_lock_and_lifecycle() -> None:
    """Verify singleton lock acquisition, duplicate rejection, and per-device isolation."""
    import os
    from cluster.cluster_worker import (
        acquire_singleton_lock,
        get_running_worker_pid,
        is_pid_running,
        release_singleton_lock,
    )

    dev_a = "cuda_0"
    dev_b = "cuda_1"

    release_singleton_lock(dev_a)
    release_singleton_lock(dev_b)
    try:
        assert acquire_singleton_lock(dev_a) is True
        # Cannot acquire again on same device tag
        assert acquire_singleton_lock(dev_a) is False
        # But CAN acquire simultaneously on a different device tag
        assert acquire_singleton_lock(dev_b) is True

        pid_a = get_running_worker_pid(dev_a)
        pid_b = get_running_worker_pid(dev_b)
        assert pid_a == os.getpid()
        assert pid_b == os.getpid()
        assert is_pid_running(pid_a) is True
    finally:
        release_singleton_lock(dev_a)
        release_singleton_lock(dev_b)

    assert get_running_worker_pid(dev_a) is None
    assert get_running_worker_pid(dev_b) is None


def test_storage_bus_worker_commands_and_purge(tmp_path: Path) -> None:
    """Verify set_worker_command, get_worker_command, and delete_offline_workers."""
    bus = ClusterStorageBus(tmp_path)

    bus.register_worker("w1", "host_a", "NVIDIA RTX 4090", 24.0)
    bus.register_worker("w2", "host_b", "NVIDIA RTX 3090", 24.0)

    # Worker command lifecycle
    assert bus.get_worker_command("w1") is None
    bus.set_worker_command("w1", "STOP")
    assert bus.get_worker_command("w1") == "STOP"
    bus.set_worker_command("w1", None)
    assert bus.get_worker_command("w1") is None

    # Offline worker purging
    # Simulate w2 having an old heartbeat > 60s ago
    with bus._connect() as conn:
        conn.execute("UPDATE workers SET last_heartbeat = ? WHERE worker_id = ?", (time.time() - 100.0, "w2"))

    deleted = bus.delete_offline_workers(stale_threshold_seconds=30.0)
    assert deleted == 1
    workers = bus.list_workers()
    assert len(workers) == 1
    assert workers[0]["worker_id"] == "w1"

    # Single worker deletion
    assert bus.delete_worker("w1") is True
    assert len(bus.list_workers()) == 0


def test_get_worker_executable() -> None:
    """Verify worker executable resolution returns a valid executable path."""
    from cluster.cluster_worker import get_worker_executable

    exe = get_worker_executable()
    assert isinstance(exe, str)
    assert Path(exe).exists()


def test_cluster_telemetry_bridge() -> None:
    """Verify ClusterTelemetryBridge emits Qt signals across threads safely."""
    import threading
    from PySide6.QtWidgets import QApplication
    from interface.screens.cluster_screen import ClusterTelemetryBridge

    app = QApplication.instance() or QApplication([])
    bridge = ClusterTelemetryBridge()
    received = []

    bridge.telemetry_ready.connect(lambda w, j, err: received.append((w, j, err)))

    def _emitter():
        time.sleep(0.02)
        bridge.telemetry_ready.emit([{"worker_id": "test_node"}], None, None)

    t = threading.Thread(target=_emitter)
    t.start()
    t.join()

    app.processEvents()
    assert len(received) == 1
    assert received[0][0][0]["worker_id"] == "test_node"
    assert received[0][1] is None
    assert received[0][2] is None


def test_dynamic_fault_tolerant_shard_reassignment(tmp_path: Path) -> None:
    """Verify that when a worker drops out, surviving workers automatically re-partition and take over 100% of data."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "job_fault_tol"
    total_tokens = 3072
    context_length = 256

    bus.create_job(
        job_id=job_id,
        model_config={},
        training_config={},
        dataset_path="dummy.npy",
        max_rounds=5,
        min_workers=1,
    )

    # 3 workers join the job
    bus.claim_job_slot(job_id, "worker_a")
    bus.claim_job_slot(job_id, "worker_b")
    bus.claim_job_slot(job_id, "worker_c")

    # Round 0 begins: workers retrieve their dynamic shard assignments
    s0, tot0 = bus.get_worker_shard_assignment(job_id, "worker_a", round_num=0)
    s1, tot1 = bus.get_worker_shard_assignment(job_id, "worker_b", round_num=0)
    s2, tot2 = bus.get_worker_shard_assignment(job_id, "worker_c", round_num=0)

    assert (s0, tot0) == (0, 3)
    assert (s1, tot1) == (1, 3)
    assert (s2, tot2) == (2, 3)

    # Initial boundaries
    b0 = compute_shard_boundaries(total_tokens, s0, tot0, context_length)
    b1 = compute_shard_boundaries(total_tokens, s1, tot1, context_length)
    b2 = compute_shard_boundaries(total_tokens, s2, tot2, context_length)
    assert b0 == (0, 1024)
    assert b1 == (1024, 2048)
    assert b2 == (2048, 3072)

    # Worker B crashes/dies and is marked DROPPED
    bus.mark_worker_dropped(job_id, "worker_b", reason="crash")

    # In round 1, surviving workers re-query their shard assignments
    new_s0, new_tot0 = bus.get_worker_shard_assignment(job_id, "worker_a", round_num=1)
    new_s2, new_tot2 = bus.get_worker_shard_assignment(job_id, "worker_c", round_num=1)

    assert (new_s0, new_tot0) == (0, 2)
    assert (new_s2, new_tot2) == (1, 2)

    # Reallocated boundaries: 2 workers now cover the entire dataset without gaps!
    re_b0 = compute_shard_boundaries(total_tokens, new_s0, new_tot0, context_length)
    re_b2 = compute_shard_boundaries(total_tokens, new_s2, new_tot2, context_length)

    assert re_b0 == (0, 1536)
    assert re_b2 == (1536, 3072)
    assert re_b0[1] == re_b2[0]  # Seamless boundary
    assert re_b2[1] == total_tokens  # 100% data covered


def test_coordinator_telemetry_aggregation_and_summary(tmp_path: Path) -> None:
    """Verify coordinator collects worker telemetries, aggregates loss/speed, and records round summary."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "job_telemetry"

    bus.create_job(
        job_id=job_id,
        model_config={},
        training_config={},
        dataset_path="dummy.npy",
        max_rounds=2,
        sync_interval_steps=100,
        min_workers=2,
    )

    bus.get_worker_shard_assignment(job_id, "node_1", 0)
    bus.get_worker_shard_assignment(job_id, "node_2", 0)

    # Deposit weights
    bus.save_worker_weights(job_id, 0, "node_1", {"w": torch.tensor([1.0])})
    bus.save_worker_weights(job_id, 0, "node_2", {"w": torch.tensor([3.0])})

    # Deposit telemetry
    bus.save_worker_telemetry(job_id, 0, "node_1", {
        "worker_id": "node_1",
        "avg_loss": 2.0,
        "tokens_per_sec": 5000.0,
        "tokens_processed": 50000,
    })
    bus.save_worker_telemetry(job_id, 0, "node_2", {
        "worker_id": "node_2",
        "avg_loss": 3.0,
        "tokens_per_sec": 7000.0,
        "tokens_processed": 70000,
    })

    coordinator = ClusterCoordinator(bus, job_id)
    collected_callbacks = []

    global_model = coordinator.wait_and_average_round(
        round_num=0,
        poll_interval_seconds=0.05,
        progress_callback=lambda m: collected_callbacks.append(m),
    )

    assert global_model is not None
    assert len(collected_callbacks) >= 1
    summary = collected_callbacks[-1]

    # Global loss is average of 2.0 and 3.0 = 2.5
    assert summary["global_loss"] == 2.5
    # Aggregate speed is sum of 5000 and 7000 = 12000
    assert summary["aggregate_tokens_per_sec"] == 12000.0
    assert summary["total_tokens_round"] == 120000
    assert summary["ready_workers_count"] == 2
    assert summary["worker_losses"]["node_1"] == 2.0
    assert summary["worker_losses"]["node_2"] == 3.0

    # Verify persisted in SQLite
    history = bus.get_all_round_history(job_id)
    assert len(history) == 1
    assert history[0]["round_number"] == 0
    assert history[0]["avg_loss"] == 2.5


def test_coordinator_drops_straggler_and_reallocates(tmp_path: Path) -> None:
    """Verify coordinator drops dead workers upon timeout and reallocates workload."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "job_straggler_drop"

    bus.create_job(
        job_id=job_id,
        model_config={},
        training_config={},
        dataset_path="dummy.npy",
        max_rounds=2,
        min_workers=1,
        sync_timeout_seconds=0.2,  # Short timeout for testing
    )

    bus.get_worker_shard_assignment(job_id, "live_worker", 0)
    bus.get_worker_shard_assignment(job_id, "dead_worker", 0)

    # Only live_worker deposits weights
    bus.save_worker_weights(job_id, 0, "live_worker", {"w": torch.tensor([5.0])})

    coordinator = ClusterCoordinator(bus, job_id)
    global_model = coordinator.wait_and_average_round(round_num=0, poll_interval_seconds=0.05)

    assert global_model is not None
    assert torch.equal(global_model["w"], torch.tensor([5.0]))

    # Verify dead_worker was marked DROPPED
    active_participants = bus.get_job_participants(job_id)
    assert len(active_participants) == 1
    assert active_participants[0]["worker_id"] == "live_worker"

    # For next round, live_worker gets assigned all shards (slot 0 of 1)
    slot, total = bus.get_worker_shard_assignment(job_id, "live_worker", round_num=1)
    assert (slot, total) == (0, 1)


def test_storage_bus_heartbeat_resource_metrics(tmp_path: Path) -> None:
    """Verify heartbeat accepts and updates CPU, RAM, and VRAM telemetry."""
    bus = ClusterStorageBus(tmp_path)
    bus.register_worker("node_alpha", "host-1", "NVIDIA GeForce GTX 1650", 4.0)

    # Send heartbeat with metrics
    bus.heartbeat("node_alpha", status="IDLE", metrics={
        "cpu_percent": 24.5,
        "ram_used_gb": 6.8,
        "ram_total_gb": 16.0,
        "vram_used_gb": 3.2,
        "vram_total_gb": 4.0,
    })

    workers = bus.list_workers()
    assert len(workers) == 1
    w = workers[0]
    assert w["worker_id"] == "node_alpha"
    assert w["cpu_percent"] == 24.5
    assert w["ram_used_gb"] == 6.8
    assert w["ram_total_gb"] == 16.0
    assert w["vram_used_gb"] == 3.2


def test_collect_system_metrics_standalone() -> None:
    """Verify standalone system metrics collection helper."""
    from cluster.cluster_worker import collect_system_metrics
    metrics = collect_system_metrics("cpu", total_vram_gb=4.0)
    assert "cpu_percent" in metrics
    assert "ram_used_gb" in metrics
    assert "ram_total_gb" in metrics
    assert "vram_used_gb" in metrics
    assert "vram_total_gb" in metrics
    assert metrics["vram_total_gb"] == 4.0


def test_durable_checkpoints_storage(tmp_path: Path) -> None:
    """Verify durable checkpoints storage on shared drive."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "job_ckpt_test"

    state_dict = {"weight": torch.tensor([10.0, 20.0])}

    # Step checkpoint
    saved_path = bus.save_checkpoint(job_id, step=250, state_dict=state_dict, is_final=False)
    assert saved_path.exists()
    assert saved_path.name == "checkpoint_step_000250.pt"

    # Latest checkpoint pointer
    latest = bus.load_latest_checkpoint(job_id)
    assert latest is not None
    assert torch.equal(latest["weight"], state_dict["weight"])

    # Final model checkpoint
    bus.save_checkpoint(job_id, step=500, state_dict=state_dict, is_final=True)
    final_path = bus.get_checkpoints_dir(job_id) / "final_model.pt"
    assert final_path.exists()


def test_worker_sqlite_logging(tmp_path: Path) -> None:
    """Verify worker log entries are written to and retrieved from SQLite bus."""
    bus = ClusterStorageBus(tmp_path)
    wid = "node_gpu_0"

    bus.write_worker_log(wid, "Starting worker engine...", level="INFO")
    bus.write_worker_log(wid, "CUDA out of memory", level="ERROR")

    logs = bus.get_worker_logs(wid)
    assert len(logs) == 2
    assert logs[0]["message"] == "Starting worker engine..."
    assert logs[0]["level"] == "INFO"
    assert logs[1]["message"] == "CUDA out of memory"
    assert logs[1]["level"] == "ERROR"


def test_dataset_vocab_size_clamping() -> None:
    """Verify ShardedTokenDataset and StandaloneTokenDataset clamp out-of-range token IDs to avoid CUDA asserts."""
    from cluster_worker import StandaloneTokenDataset

    # Tokens with IDs up to 5000
    raw_tokens = np.array([10, 50, 500, 1200, 4999, 5000, 25, 30], dtype=np.int64)
    vocab_size = 1000

    # 1. ShardedTokenDataset
    ds = ShardedTokenDataset(raw_tokens, context_length=4, shard_index=0, total_shards=1, vocab_size=vocab_size)
    x, y = ds[0]
    assert int(x.max().item()) <= vocab_size - 1
    assert int(y.max().item()) <= vocab_size - 1
    assert int(x.min().item()) >= 0
    assert int(y.min().item()) >= 0

    # 2. StandaloneTokenDataset
    s_ds = StandaloneTokenDataset(raw_tokens, context_length=4, shard_index=0, total_shards=1, vocab_size=vocab_size)
    sx, sy = s_ds[0]
    assert int(sx.max().item()) <= vocab_size - 1
    assert int(sy.max().item()) <= vocab_size - 1
    assert int(sx.min().item()) >= 0
    assert int(sy.min().item()) >= 0


def test_storage_bus_truncate_journal_mode(tmp_path: Path) -> None:
    """Verify ClusterStorageBus initializes in network-share compatible TRUNCATE journal mode."""
    bus = ClusterStorageBus(tmp_path)
    with bus._connect() as conn:
        cursor = conn.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.upper() in {"TRUNCATE", "MEMORY", "DELETE"}


def test_get_active_job_sweeps_stale_jobs(tmp_path: Path) -> None:
    """Verify get_active_job automatically marks uncoordinated jobs with stale updated_at as STOPPED."""
    bus = ClusterStorageBus(tmp_path)

    # 1. Create a job that is stale (updated_at 100 seconds ago)
    bus.create_job("stale_job", {}, {}, "/dummy.npy")
    with bus._connect() as conn:
        conn.execute("UPDATE jobs SET status = 'RUNNING', updated_at = ? WHERE job_id = 'stale_job';", (time.time() - 100.0,))

    # When querying with max_stale_seconds=30.0, the stale job should be swept to STOPPED
    active = bus.get_active_job(max_stale_seconds=30.0)
    assert active is None

    # Verify in DB that it was marked STOPPED
    stale_rec = bus.get_job("stale_job")
    assert stale_rec is not None
    assert stale_rec["status"] == "STOPPED"

    # 2. Create a fresh active job
    bus.create_job("fresh_job", {}, {}, "/dummy.npy")
    bus.set_job_status("fresh_job", "RUNNING")
    active = bus.get_active_job(max_stale_seconds=30.0)
    assert active is not None
    assert active["job_id"] == "fresh_job"


def test_touch_job_coordinator_liveness(tmp_path: Path) -> None:
    """Verify touch_job updates updated_at timestamp to signal coordinator activity."""
    bus = ClusterStorageBus(tmp_path)
    bus.create_job("coordinated_job", {}, {}, "/dummy.npy")

    old_time = time.time() - 50.0
    with bus._connect() as conn:
        conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = 'coordinated_job';", (old_time,))

    # Touch job
    bus.touch_job("coordinated_job")

    rec = bus.get_job("coordinated_job")
    assert rec is not None
    assert rec["updated_at"] > old_time + 40.0


def test_list_workers_offline_override(tmp_path: Path) -> None:
    """Verify list_workers marks explicitly OFFLINE or STOPPED workers as OFFLINE regardless of heartbeat freshness."""
    bus = ClusterStorageBus(tmp_path)
    bus.register_worker("w_offline", "host1", "RTX 3090", 24.0)

    # Heartbeat with fresh timestamp (0 seconds ago) but status='OFFLINE'
    bus.heartbeat("w_offline", status="OFFLINE")

    workers = bus.list_workers(active_within_seconds=60.0)
    assert len(workers) == 1
    assert workers[0]["status"] == "OFFLINE"
    assert workers[0]["is_online"] is False

    # Set to IDLE -> should now be online
    bus.heartbeat("w_offline", status="IDLE")
    workers = bus.list_workers(active_within_seconds=60.0)
    assert workers[0]["status"] == "IDLE"
    assert workers[0]["is_online"] is True


def test_worker_deposits_telemetry_and_coordinator_aggregates(tmp_path: Path) -> None:
    """Verify workers deposit telemetry and coordinator computes valid global loss and throughput."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "telemetry_test_job"
    bus.create_job(
        job_id=job_id,
        model_config={"vocab_size": 256, "embedding_size": 64, "head_count": 2, "layer_count": 2, "context_length": 16},
        training_config={"learning_rate": 1e-3, "batch_size": 2},
        dataset_path="/dummy.npy",
        max_rounds=1,
        sync_interval_steps=5,
        min_workers=2,
    )

    worker_1 = ClusterWorker(bus, worker_id="node_mumws4351", device="cpu")
    worker_2 = ClusterWorker(bus, worker_id="node_mumws4857", device="cpu")

    model = build_model_from_config({"vocab_size": 256, "embedding_size": 64, "head_count": 2, "layer_count": 2}, "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Synthetic tokens
    tokens = np.random.randint(0, 255, size=500, dtype=np.int64)
    ds = ShardedTokenDataset(tokens, context_length=16, shard_index=0, total_shards=2, vocab_size=256)
    loader = DataLoader(ds, batch_size=2)

    # Worker 1 runs round
    loss_1, _ = worker_1.run_training_round(
        job_id=job_id,
        round_num=0,
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        dataloader_iter=iter(loader),
        steps_per_round=5,
    )
    assert loss_1 > 0.0
    metrics_1 = worker_1._last_round_metrics
    assert metrics_1["tokens_processed"] > 0
    assert metrics_1["tokens_per_sec"] > 0.0

    # Save weights & telemetry for worker 1
    bus.save_worker_weights(job_id, 0, "node_mumws4351", {k: v.clone() for k, v in model.state_dict().items()})
    bus.save_worker_telemetry(job_id, 0, "node_mumws4351", {
        "worker_id": "node_mumws4351",
        "round": 0,
        "steps_completed": 5,
        "avg_loss": round(loss_1, 4),
        "tokens_processed": metrics_1["tokens_processed"],
        "tokens_per_sec": round(metrics_1["tokens_per_sec"], 1),
        "timestamp": time.time(),
    })

    # Save weights & telemetry for worker 2
    bus.save_worker_weights(job_id, 0, "node_mumws4857", {k: v.clone() for k, v in model.state_dict().items()})
    bus.save_worker_telemetry(job_id, 0, "node_mumws4857", {
        "worker_id": "node_mumws4857",
        "round": 0,
        "steps_completed": 5,
        "avg_loss": 5.4321,
        "tokens_processed": 160,
        "tokens_per_sec": 3200.0,
        "timestamp": time.time(),
    })

    # Verify telemetry files exist on disk
    t1 = bus.load_worker_telemetry(job_id, 0, "node_mumws4351")
    t2 = bus.load_worker_telemetry(job_id, 0, "node_mumws4857")
    assert t1 is not None and t1["avg_loss"] == round(loss_1, 4)
    assert t2 is not None and t2["avg_loss"] == 5.4321

    # Coordinator averages round
    coordinator = ClusterCoordinator(bus, job_id)
    avg_state = coordinator.wait_and_average_round(0)
    assert avg_state is not None

    # Verify global summary in database
    rounds = bus.get_all_round_history(job_id)
    assert len(rounds) == 1
    r0 = rounds[0]
    assert r0["avg_loss"] > 0.0
    assert r0["metrics"]["aggregate_tokens_per_sec"] > 0.0
    assert "node_mumws4351" in r0["participating_workers"]
    assert "node_mumws4857" in r0["participating_workers"]

    worker_1.stop()
    worker_2.stop()


def test_cmd_worker_rejects_duplicate(tmp_path: Path) -> None:
    """Verify cmd_worker returns non-zero when a worker is already running for the device."""
    import argparse
    import subprocess
    import sys
    from cluster.cli import cmd_worker
    from cluster.cluster_worker import get_device_tag, get_lock_file, release_singleton_lock
    from cluster.worker import get_hardware_info

    dev_str, _, _ = get_hardware_info("cpu")
    tag = get_device_tag(dev_str)
    release_singleton_lock(tag)

    args = argparse.Namespace(
        shared_dir=str(tmp_path),
        worker_id="dup_test_worker",
        device="cpu",
        heartbeat_interval=5.0,
        poll_interval=1.0,
        detach=False,
    )

    # Spawn an independent background dummy process to simulate an already-running worker
    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"])
    lock_file = get_lock_file(tag)
    lock_file.write_text(str(sub.pid))

    try:
        ret = cmd_worker(args)
        assert ret != 0, "cmd_worker must exit with non-zero when another worker is active"
    finally:
        sub.terminate()
        sub.wait(timeout=3.0)
        lock_file.unlink(missing_ok=True)
        release_singleton_lock(tag)


def test_worker_job_compatibility_gatekeeping(tmp_path: Path) -> None:
    """Verify that ClusterWorker gatekeeping rejects incompatible jobs and refuses slot claim."""
    bus = ClusterStorageBus(tmp_path)
    worker = ClusterWorker(bus=bus, worker_id="compat_node_01", device="cpu")

    try:
        # 1. Non-existent dataset
        incompat_job_1 = {
            "job_id": "job_missing_ds",
            "dataset_path": str(tmp_path / "nonexistent.npy"),
            "training_config": {},
            "model_config": {},
        }
        compat, reason = worker.is_job_compatible(incompat_job_1)
        assert compat is False
        assert "not accessible" in reason.lower()

        # 2. Target worker whitelist exclusion
        valid_dataset_path = tmp_path / "train_tokens.npy"
        np.save(valid_dataset_path, np.arange(100, dtype=np.int32))

        incompat_job_2 = {
            "job_id": "job_whitelist",
            "dataset_path": str(valid_dataset_path),
            "target_workers": ["other_node_99"],
            "training_config": {},
            "model_config": {},
        }
        compat, reason = worker.is_job_compatible(incompat_job_2)
        assert compat is False
        assert "target_workers" in reason

        # 3. Degraded worker status
        worker._current_status = "DEGRADED"
        compat, reason = worker.is_job_compatible({
            "job_id": "job_valid",
            "dataset_path": str(valid_dataset_path),
            "training_config": {},
            "model_config": {},
        })
        assert compat is False
        assert "degraded" in reason.lower()

        # 4. Bus level slot claiming rejection for degraded worker
        bus.heartbeat(worker.worker_id, status="DEGRADED")
        bus.create_job(
            job_id="job_bus_guard",
            model_config={"vocab_size": 100},
            training_config={"batch_size": 2},
            dataset_path=str(valid_dataset_path),
        )
        import pytest
        with pytest.raises(RuntimeError, match="status 'DEGRADED'"):
            bus.claim_job_slot("job_bus_guard", worker.worker_id)
    finally:
        worker.stop()


def test_cluster_validation_loss_evaluation(tmp_path: Path) -> None:
    """Verify that worker evaluates validation loss on val_tokens.npy and coordinator aggregates it."""
    bus = ClusterStorageBus(tmp_path)

    # Create dummy train and val datasets
    train_npy = tmp_path / "train_tokens.npy"
    val_npy = tmp_path / "val_tokens.npy"
    np.save(train_npy, np.random.randint(0, 50, size=2048, dtype=np.int32))
    np.save(val_npy, np.random.randint(0, 50, size=512, dtype=np.int32))

    job_id = "job_val_test"
    model_cfg = {
        "vocab_size": 64,
        "context_length": 32,
        "embedding_size": 32,
        "head_count": 2,
        "layer_count": 2,
    }
    training_cfg = {
        "learning_rate": 1e-3,
        "batch_size": 2,
        "precision": "float32",
        "use_amp": False,
    }

    bus.create_job(
        job_id=job_id,
        model_config=model_cfg,
        training_config=training_cfg,
        dataset_path=str(train_npy),
        max_rounds=1,
        sync_interval_steps=5,
        min_workers=1,
    )

    worker = ClusterWorker(bus=bus, worker_id="val_node_01", device="cpu")
    job = bus.get_job(job_id)
    assert job is not None

    import threading
    worker_thread = threading.Thread(target=lambda: worker.execute_job(job, poll_interval=0.1), daemon=True)
    worker_thread.start()

    try:
        # Coordinator aggregates round and includes val_loss in summary
        coordinator = ClusterCoordinator(bus=bus, job_id=job_id)
        global_state = coordinator.wait_for_round_and_aggregate(round_num=0, poll_interval_seconds=0.1)
        assert global_state is not None
        worker_thread.join(timeout=10.0)

        # Check worker telemetry recorded val_loss
        tel = bus.load_worker_telemetry(job_id, round_num=0, worker_id=worker.worker_id)
        assert tel is not None
        assert "val_loss" in tel
        assert tel["val_loss"] is not None
        assert isinstance(tel["val_loss"], float)
        assert tel["val_loss"] > 0.0

        history = bus.get_all_round_history(job_id)
        assert len(history) == 1
        summary_metrics = history[0]["metrics"]
        assert "val_loss" in summary_metrics
        assert summary_metrics["val_loss"] == tel["val_loss"]
    finally:
        worker.stop()
        worker_thread.join(timeout=2.0)


def test_worker_disabled_state_management(tmp_path: Path) -> None:
    """Verify workers can be disabled to pause job pickup while keeping telemetry active."""
    bus = ClusterStorageBus(tmp_path)
    wid = "test_worker_mumws9999"

    # Register worker
    bus.register_worker(wid, "mumws9999", "RTX 5060 Ti", 16.0)
    bus.heartbeat(wid, status="IDLE", metrics={"cpu_percent": 12.0, "ram_used_gb": 4.5, "vram_used_gb": 2.0})

    # By default, worker is enabled
    assert bus.is_worker_enabled(wid) is True
    workers = bus.list_workers()
    assert len(workers) == 1
    assert workers[0]["enabled"] is True
    assert workers[0]["status"] == "IDLE"

    # Disable worker
    bus.set_worker_enabled(wid, False)
    assert bus.is_worker_enabled(wid) is False

    # Status in list_workers becomes DISABLED while still online
    workers = bus.list_workers()
    assert workers[0]["enabled"] is False
    assert workers[0]["status"] == "DISABLED"
    assert workers[0]["is_online"] is True

    # Submitting a job and attempting to claim slot raises RuntimeError
    job_id = "job_test_disabled_123"
    bus.create_job(
        job_id=job_id,
        model_config={"vocab_size": 256, "embedding_size": 64},
        training_config={"batch_size": 2},
        dataset_path=str(tmp_path / "train.npy"),
        max_rounds=1,
    )
    import pytest
    with pytest.raises(RuntimeError, match="DISABLED"):
        bus.claim_job_slot(job_id, wid)

    # Worker compatibility check rejects disabled worker
    worker = ClusterWorker(bus, wid, device="cpu")
    job = bus.get_job(job_id)
    is_compat, reason = worker.is_job_compatible(job)
    assert is_compat is False
    assert "disabled" in reason.lower()

    # Re-enable worker
    bus.set_worker_enabled(wid, True)
    assert bus.is_worker_enabled(wid) is True
    workers = bus.list_workers()
    assert workers[0]["enabled"] is True
    assert workers[0]["status"] == "IDLE"

    # Now can claim job slot successfully
    shard_idx, total_shards = bus.claim_job_slot(job_id, wid)
    assert shard_idx == 0
    assert total_shards == 1


def test_job_manifest_log_formatting_with_validation_loss(tmp_path: Path) -> None:
    """Verify Job Events Log formatting matches: • Round 7: Global Loss 5.9043 | Validation Loss 6.9520 | Speed: 64,488 tok/s | Workers: [...]"""
    bus = ClusterStorageBus(tmp_path)
    job_id = "cluster_job_log_format_test"
    bus.create_job(
        job_id=job_id,
        model_config={"vocab_size": 256, "embedding_size": 64},
        training_config={"batch_size": 2},
        dataset_path=str(tmp_path / "train.npy"),
        max_rounds=10,
    )

    workers = ["mumws4351", "mumws4857_cuda_0", "mumws4886", "mumws4886_slot2"]
    bus.record_round_summary(
        job_id=job_id,
        round_num=6,  # 0-indexed round 6 -> Round 7
        participating_workers=workers,
        avg_loss=5.9043,
        metrics={
            "round": 6,
            "val_loss": 6.9520,
            "aggregate_tokens_per_sec": 64488.0,
            "effective_step": 1750,
        },
    )

    rounds = bus.get_all_round_history(job_id)
    assert len(rounds) == 1
    r = rounds[0]
    r_num = r.get("round_number", 0) + 1
    r_loss = r.get("avg_loss", 0.0)
    r_workers = ", ".join(r.get("participating_workers", []))
    r_metrics = r.get("metrics", {})
    r_spd = r_metrics.get("aggregate_tokens_per_sec", 0.0)
    r_val = r_metrics.get("val_loss")
    val_str = f" | Validation Loss {float(r_val):.4f}" if r_val is not None else ""
    log_line = f"• Round {r_num}: Global Loss {r_loss:.4f}{val_str} | Speed: {r_spd:,.0f} tok/s | Workers: [{r_workers}]"

    expected = "• Round 7: Global Loss 5.9043 | Validation Loss 6.9520 | Speed: 64,488 tok/s | Workers: [mumws4351, mumws4857_cuda_0, mumws4886, mumws4886_slot2]"
    assert log_line == expected


def test_cluster_fine_tuning_job_creation_and_staging(tmp_path: Path) -> None:
    """Verify cluster fine-tuning job creation, base model staging, and metadata persistence."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "cluster_fine_tune_staging_test"

    # Create dummy base checkpoint
    base_ckpt_path = tmp_path / "pretrained_base.pt"
    dummy_weights = {
        "transformer.wte.weight": torch.randn(256, 64),
        "transformer.h.0.attn.c_attn.weight": torch.randn(192, 64),
    }
    torch.save({"model_state_dict": dummy_weights, "model_config": {"vocab_size": 256, "embedding_size": 64}}, base_ckpt_path)

    lora_cfg = {
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.05,
        "target_modules": "attention",
    }

    bus.create_job(
        job_id=job_id,
        model_config={"vocab_size": 256, "embedding_size": 64},
        training_config={"learning_rate": 5e-5, "batch_size": 4},
        dataset_path=str(tmp_path / "instruction_tokens.npy"),
        max_rounds=5,
        sync_interval_steps=100,
        min_workers=1,
        job_type="fine_tune",
        base_checkpoint_path=str(base_ckpt_path),
        peft_method="lora",
        lora_config=lora_cfg,
    )

    job = bus.get_job(job_id)
    assert job is not None
    assert job["job_type"] == "fine_tune"
    assert job["peft_method"] == "lora"
    assert job["lora_config"]["rank"] == 8
    assert job["lora_config"]["alpha"] == 16.0
    assert job["base_checkpoint_path"] is not None

    # Verify base model was staged to shared storage jobs folder
    staged_base = tmp_path / "jobs" / job_id / "base_model.pt"
    assert staged_base.exists()

    # Verify load_base_model_weights loads the exact weights
    loaded_base = bus.load_base_model_weights(job_id, device="cpu")
    assert loaded_base is not None
    assert "transformer.wte.weight" in loaded_base
    torch.testing.assert_close(loaded_base["transformer.wte.weight"], dummy_weights["transformer.wte.weight"])


def test_cluster_worker_lora_adapters_and_freezing() -> None:
    """Verify LoRA adapter application and non-LoRA parameter freezing on workers."""
    from cluster.worker import LoRALinear, apply_lora_adapters, freeze_non_lora_parameters, lora_state_dict

    class SimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Module()
            self.attn.q_proj = nn.Linear(32, 32)
            self.attn.v_proj = nn.Linear(32, 32)
            self.mlp = nn.Module()
            self.mlp.fc = nn.Linear(32, 64)

    net = SimpleNet()
    total_before = sum(p.numel() for p in net.parameters())

    # Apply LoRA adapters to attention projections
    num_wrapped = apply_lora_adapters(net, rank=4, alpha=8.0, dropout=0.0, target_modules="attention")
    assert num_wrapped == 2
    assert isinstance(net.attn.q_proj, LoRALinear)
    assert isinstance(net.attn.v_proj, LoRALinear)
    assert not isinstance(net.mlp.fc, LoRALinear)

    # Freeze non-LoRA parameters
    freeze_non_lora_parameters(net)

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    trainable_names = [name for name, p in net.named_parameters() if p.requires_grad]

    assert len(trainable_params) > 0
    # Trainable parameters must strictly be LoRA parameters
    for name in trainable_names:
        assert "lora_a" in name or "lora_b" in name

    # Base linear weights must be frozen
    assert net.attn.q_proj.base.weight.requires_grad is False
    assert net.mlp.fc.weight.requires_grad is False

    # Extract adapter state dict
    adapters = lora_state_dict(net)
    assert len(adapters) == 4  # q_proj.lora_a, q_proj.lora_b, v_proj.lora_a, v_proj.lora_b


def test_coordinator_fine_tune_lora_checkpoint_export(tmp_path: Path) -> None:
    """Verify coordinator exports adapter_model.pt and final_model_merged.pt on final round of fine-tune job."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "coordinator_lora_export_test"

    model_cfg = {"vocab_size": 256, "context_length": 32, "embedding_size": 32, "head_count": 2, "layer_count": 1}
    lora_cfg = {"rank": 4, "alpha": 8.0, "dropout": 0.0, "target_modules": "attention"}

    bus.create_job(
        job_id=job_id,
        model_config=model_cfg,
        training_config={"batch_size": 2},
        dataset_path=str(tmp_path / "train.npy"),
        max_rounds=1,
        sync_interval_steps=10,
        min_workers=1,
        job_type="fine_tune",
        peft_method="lora",
        lora_config=lora_cfg,
    )

    from cluster.worker import build_model_from_config, apply_lora_adapters, freeze_non_lora_parameters
    worker_model = build_model_from_config(model_cfg, "cpu")
    apply_lora_adapters(worker_model, rank=4, alpha=8.0, dropout=0.0, target_modules="attention")
    freeze_non_lora_parameters(worker_model)

    bus.claim_job_slot(job_id, "worker_01")
    bus.save_worker_weights(job_id, 0, "worker_01", worker_model.state_dict())
    bus.save_worker_telemetry(job_id, 0, "worker_01", {
        "worker_id": "worker_01",
        "round": 0,
        "steps_completed": 10,
        "avg_loss": 3.456,
        "tokens_processed": 500,
        "tokens_per_sec": 1200.0,
        "timestamp": time.time(),
    })

    coordinator = ClusterCoordinator(bus, job_id)
    global_weights = coordinator.wait_and_average_round(round_num=0)
    assert global_weights is not None

    ckpt_dir = bus.get_checkpoints_dir(job_id)
    assert (ckpt_dir / "latest_checkpoint.pt").exists()
    assert (ckpt_dir / "final_model.pt").exists()
    assert (ckpt_dir / "adapter_model.pt").exists()
    assert (ckpt_dir / "final_model_merged.pt").exists()
    assert (ckpt_dir / "training_summary.json").exists()
    assert (ckpt_dir / "model_lineage.json").exists()

    # Verify adapter checkpoint contents
    loaded_adapter = torch.load(ckpt_dir / "adapter_model.pt", map_location="cpu")
    assert "adapter_state_dict" in loaded_adapter
    assert len(loaded_adapter["adapter_state_dict"]) > 0


def test_cluster_telemetry_reflection_on_fine_tuning_screen(tmp_path: Path) -> None:
    """Verify _apply_cluster_telemetry updates all Fine-Tuning screen metric chips and log."""
    from interface.screens.cluster_screen import ClusterScreenMixin

    class _MockChip:
        def __init__(self) -> None:
            self.text = ""
        def setText(self, t: str) -> None:
            self.text = t

    class _MockProgress:
        def __init__(self) -> None:
            self.value = 0
        def setValue(self, v: int) -> None:
            self.value = v

    class _MockButton:
        def __init__(self) -> None:
            self.enabled = True
            self.text = ""
        def setEnabled(self, e: bool) -> None:
            self.enabled = e
        def setText(self, t: str) -> None:
            self.text = t

    class _MockLog:
        def __init__(self) -> None:
            self.lines: list[str] = []
        def append(self, t: str) -> None:
            self.lines.append(t)

    class _ScreenHost(ClusterScreenMixin):
        def __init__(self) -> None:
            self.fine_tune_loss_metric = _MockChip()
            self.fine_tune_val_metric = _MockChip()
            self.fine_tune_step_metric = _MockChip()
            self.fine_tune_epoch_metric = _MockChip()
            self.fine_tune_speed_metric = _MockChip()
            self.fine_tune_lr_metric = _MockChip()
            self.fine_tune_eta_metric = _MockChip()
            self.fine_tune_progress = _MockProgress()
            self.fine_tune_process_status = _MockChip()
            self.fine_tune_log = _MockLog()
            self.fine_tune_button = _MockButton()
            self.stop_fine_tune_button = _MockButton()
            self.cluster_status_label = _MockChip()
            self.cluster_round_label = _MockChip()

    host = _ScreenHost()
    active_job = {
        "job_id": "cluster_ft_telemetry_test",
        "job_type": "fine_tune",
        "status": "RUNNING",
        "current_round": 2,
        "max_rounds": 10,
        "sync_interval_steps": 250,
        "training_config": {"learning_rate": 3e-5},
        "_latest_round": {
            "round_number": 1,
            "avg_loss": 4.1234,
            "participating_workers": ["worker_a", "worker_b"],
            "metrics": {
                "val_loss": 4.5678,
                "aggregate_tokens_per_sec": 32000.0,
            },
        },
    }

    host._apply_cluster_telemetry(workers=[{"is_online": True}], active_job=active_job)

    assert "4.1234" in host.fine_tune_loss_metric.text
    assert "4.5678" in host.fine_tune_val_metric.text
    assert "32,000" in host.fine_tune_speed_metric.text
    assert "Round 2/10" in host.fine_tune_step_metric.text
    assert "2/10" in host.fine_tune_epoch_metric.text
    assert host.fine_tune_progress.value == 20
    assert "Cluster Local SGD" in host.fine_tune_process_status.text
    assert host.fine_tune_button.enabled is False
    assert host.stop_fine_tune_button.enabled is True
    assert len(host.fine_tune_log.lines) == 1
    assert "Global Loss 4.1234" in host.fine_tune_log.lines[0]
    assert "Validation Loss 4.5678" in host.fine_tune_log.lines[0]

    # Verify that when fleet is idle (no active job, only running_pids or empty dict),
    # _apply_cluster_telemetry does not raise KeyError: 'job_id' and resets labels to idle.
    host._apply_cluster_telemetry(workers=[{"is_online": True}], active_job={"_running_pids": {}})
    assert host.cluster_status_label.text == "Status: Fleet Idle"
    assert host.cluster_round_label.text == "Round: -"

    host._apply_cluster_telemetry(workers=[], active_job=None)
    assert host.cluster_status_label.text == "Status: Fleet Idle"
    assert host.cluster_round_label.text == "Round: -"


def test_cluster_screen_resilient_worker_bus_fallbacks() -> None:
    """Verify cluster_screen handles legacy or out-of-sync buses missing is_worker_enabled without AttributeError."""
    from interface.screens.cluster_screen import ClusterScreenMixin

    class _LegacyBusWithoutNewMethods:
        """Simulates an older ClusterStorageBus before worker enable/disable was added."""
        pass

    class _TestHost(ClusterScreenMixin):
        def __init__(self) -> None:
            self._bus = _LegacyBusWithoutNewMethods()
            self.logged_events: list[str] = []

        def _get_cluster_bus(self):
            return self._bus

        def _log_cluster_event(self, msg: str) -> None:
            self.logged_events.append(msg)

        def refresh_cluster_status(self) -> None:
            pass

    host = _TestHost()

    # Must not raise AttributeError: 'ClusterStorageBus' object has no attribute 'is_worker_enabled'
    host.toggle_cluster_worker_enabled("worker_legacy_01")
    assert len(host.logged_events) == 1
    assert "worker_legacy_01" in host.logged_events[0]

    # Must not raise AttributeError on delete_worker
    host.delete_cluster_worker("worker_legacy_01")
    assert any("Removed worker 'worker_legacy_01'" in ev for ev in host.logged_events)


def test_storage_bus_malformed_db_self_healing(tmp_path: Path) -> None:
    """Verify that when SQLite encounters a malformed database disk image, it self-heals without raising an unhandled DatabaseError."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "job_corrupt_test"

    # Create job and worker
    bus.create_job(
        job_id=job_id,
        model_config={"vocab_size": 100},
        training_config={"lr": 0.01},
        dataset_path=str(tmp_path / "data.npy"),
    )
    bus.heartbeat("worker_heal_test", status="IDLE")

    # Corrupt the database file by overwriting it with invalid bytes
    with open(bus.db_path, "r+b") as f:
        f.seek(100)
        f.write(b"CORRUPTED_GARBAGE_BYTES" * 20)

    # Calling delete_job or list_all_jobs on the malformed DB should trigger self-healing
    res = bus.delete_job(job_id)
    assert res is True

    # Check that a corrupt backup was generated
    backups = list(tmp_path.glob("cluster.db.corrupt_*"))
    assert len(backups) >= 1

    # Check that a fresh working database was restored and can process new queries
    bus.register_worker("worker_after_healing", "host1", "RTX 4090", 24.0)
    bus.heartbeat("worker_after_healing", status="IDLE")
    workers = bus.list_workers()
    assert any(w["worker_id"] == "worker_after_healing" for w in workers)


def test_cluster_worker_micro_batch_and_gradient_accumulation(tmp_path: Path) -> None:
    """Verify that run_training_round handles gradient accumulation and micro-batching correctly."""
    bus = ClusterStorageBus(tmp_path)
    worker = ClusterWorker(bus, worker_id="test_worker_accum", device="cpu")

    model_cfg = {
        "vocab_size": 128,
        "context_length": 16,
        "embedding_size": 32,
        "head_count": 2,
        "layer_count": 2,
    }
    model = build_model_from_config(model_cfg, "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokens = np.random.randint(0, 128, size=512, dtype=np.int32)
    dataset = ShardedTokenDataset(tokens, context_length=16, shard_index=0, total_shards=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

    job_spec = {
        "training_config": {
            "batch_size": 4,
            "gradient_accumulation": 2,
            "activation_checkpointing": True,
        },
        "model_config": model_cfg,
    }

    avg_loss, _ = worker.run_training_round(
        job_id="job_accum_test",
        round_num=0,
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        dataloader_iter=iter(dataloader),
        steps_per_round=4,
        job=job_spec,
    )

    assert isinstance(avg_loss, float)
    assert avg_loss > 0.0
    # Activation checkpointing must be enabled on the model
    assert getattr(model, "gradient_checkpointing", False) is True










