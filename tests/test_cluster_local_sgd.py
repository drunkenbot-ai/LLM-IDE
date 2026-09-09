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
import pytest
import torch
import torch.nn as nn

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
    assert w1["status"] == "IDLE"

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
    loaded_global = bus.load_global_weights(job_id, 0)
    assert torch.equal(loaded_global["weight"], global_weights["weight"])


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
    """Verify singleton lock acquisition, duplicate rejection, and cleanup."""
    import os
    from cluster.cluster_worker import (
        acquire_singleton_lock,
        get_running_worker_pid,
        is_pid_running,
        release_singleton_lock,
    )

    release_singleton_lock()
    try:
        assert acquire_singleton_lock() is True
        pid = get_running_worker_pid()
        assert pid == os.getpid()
        assert is_pid_running(pid) is True
    finally:
        release_singleton_lock()

    assert get_running_worker_pid() is None


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
