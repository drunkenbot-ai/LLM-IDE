"""Unit tests for local SSD dataset caching and dynamic VRAM multi-worker slots."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cluster.worker import (
    calculate_optimal_worker_slots,
    cache_dataset_to_local,
    purge_local_dataset_cache,
    is_network_path,
)


def test_is_network_path():
    """Verify detection of UNC paths and local drives."""
    assert is_network_path(r"\\mumnfs19\user_data\shared\train.npy") is True
    assert is_network_path("//mumnfs19/user_data/shared/train.npy") is True
    # C: drive should be False on Windows
    assert is_network_path(r"C:\Users\test\train.npy") is False


def test_purge_local_dataset_cache():
    """Verify stale .npy files are deleted while preserving active files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        # Create dummy npy files
        f1 = td / "old_tokens.npy"
        f2 = td / "stale_tokens.npy"
        f3 = td / "active_train.npy"
        f1.write_bytes(b"123")
        f2.write_bytes(b"456")
        f3.write_bytes(b"789")

        logs = []
        purged = purge_local_dataset_cache(
            cache_dir=td,
            keep_files={"active_train.npy"},
            log_fn=lambda msg, **kw: logs.append(msg),
        )

        assert purged == 2
        assert not f1.exists()
        assert not f2.exists()
        assert f3.exists()
        assert any("Purged 2 stale" in log for log in logs)


def test_cache_dataset_to_local():
    """Verify remote dataset is cached to local SSD and reused on second call."""
    with tempfile.TemporaryDirectory() as net_dir, tempfile.TemporaryDirectory() as local_dir:
        net_file = Path(net_dir) / "train_tokens.npy"
        arr = np.arange(1000, dtype=np.int32)
        np.save(str(net_file), arr)

        # Mock is_network_path to treat net_file as remote
        with patch("cluster.worker.is_network_path", return_value=True):
            with patch.dict(os.environ, {"LOCALAPPDATA": local_dir}):
                logs = []
                cached_path = cache_dataset_to_local(
                    str(net_file),
                    log_fn=lambda msg, **kw: logs.append(msg),
                )
                assert os.path.exists(cached_path)
                assert Path(cached_path).parent.name == "llm_cluster_cache"
                loaded = np.load(cached_path)
                np.testing.assert_array_equal(loaded, arr)
                assert any("Streaming network dataset" in log for log in logs)

                # Second call should reuse without re-downloading
                logs2 = []
                cached_path2 = cache_dataset_to_local(
                    str(net_file),
                    log_fn=lambda msg, **kw: logs2.append(msg),
                )
                assert cached_path2 == cached_path
                assert any("Verified existing local SSD dataset cache" in log for log in logs2)


def test_calculate_optimal_worker_slots():
    """Verify dynamic VRAM slot calculation for various GPU capacities with safety headroom."""
    # Test with simulated CUDA device info
    with patch("torch.cuda.is_available", return_value=True):
        # 1. 8 GB GPU with 6.5 GB job requirement -> should spawn 1 slot
        mock_props_8g = MagicMock()
        mock_props_8g.total_memory = int(8.0 * (1024 ** 3))
        with patch("torch.cuda.get_device_properties", return_value=mock_props_8g):
            slots, est_gb, node_gb = calculate_optimal_worker_slots(
                model_cfg={},
                training_cfg={"vram_required_gb": 6.5},
                device_str="cuda:0",
            )
            assert slots == 1
            assert node_gb == 8.0

        # 2. 16 GB GPU with 6.5 GB job requirement -> should spawn 2 slots (capped to avoid OOM)
        mock_props_16g = MagicMock()
        mock_props_16g.total_memory = int(15.93 * (1024 ** 3))
        with patch("torch.cuda.get_device_properties", return_value=mock_props_16g):
            slots, est_gb, node_gb = calculate_optimal_worker_slots(
                model_cfg={},
                training_cfg={"vram_required_gb": 6.5},
                device_str="cuda:0",
            )
            assert slots == 2

        # 3. 24 GB GPU with 6.5 GB job requirement -> should spawn 3 slots
        mock_props_24g = MagicMock()
        mock_props_24g.total_memory = int(24.0 * (1024 ** 3))
        with patch("torch.cuda.get_device_properties", return_value=mock_props_24g):
            slots, est_gb, node_gb = calculate_optimal_worker_slots(
                model_cfg={},
                training_cfg={"vram_required_gb": 6.5},
                device_str="cuda:0",
            )
            assert slots == 3

        # 4. Measured VRAM > 45% (e.g. 10 GB on 15.93 GB 5060 Ti) -> MUST return 1 slot
        with patch("torch.cuda.get_device_properties", return_value=mock_props_16g):
            slots, est_gb, node_gb = calculate_optimal_worker_slots(
                model_cfg={},
                training_cfg={},
                device_str="cuda:0",
                measured_vram_gb=10.0,
            )
            assert slots == 1

        # 5. 100% VRAM utilization (15.9 GB on 15.93 GB 5060 Ti) -> MUST return 1 slot
        with patch("torch.cuda.get_device_properties", return_value=mock_props_16g):
            slots, est_gb, node_gb = calculate_optimal_worker_slots(
                model_cfg={},
                training_cfg={},
                device_str="cuda:0",
                measured_vram_gb=15.9,
            )
            assert slots == 1


def test_check_and_spawn_auxiliary_slots_guards(tmp_path):
    """Verify that auxiliary slots are NOT spawned when GPU is heavily utilized or disabled."""
    from cluster.bus import ClusterStorageBus
    from cluster.worker import ClusterWorker

    bus = ClusterStorageBus(shared_dir=tmp_path / "shared")
    worker = ClusterWorker(bus=bus, worker_id="test_node_1", device="cpu")

    # Simulate CUDA device
    worker.device_str = "cuda:0"

    # 1. When GPU is 100% utilized (0 free memory), no slots should be spawned
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.synchronize"), \
         patch("torch.cuda.max_memory_reserved", return_value=int(15.9 * (1024 ** 3))), \
         patch("torch.cuda.mem_get_info", return_value=(int(0.1 * (1024 ** 3)), int(16.0 * (1024 ** 3)))), \
         patch("subprocess.Popen") as mock_popen:
        
        worker._check_and_spawn_auxiliary_slots(
            job_id="test_job_1",
            job={"model_config": {}, "training_config": {}},
        )
        assert len(worker._child_worker_procs) == 0
        mock_popen.assert_not_called()

    # 2. When CLUSTER_DISABLE_AUTO_SLOTS is set, no slots should be spawned
    with patch.dict(os.environ, {"CLUSTER_DISABLE_AUTO_SLOTS": "1"}), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("subprocess.Popen") as mock_popen:
        
        worker._check_and_spawn_auxiliary_slots(
            job_id="test_job_1",
            job={"model_config": {}, "training_config": {}},
        )
        assert len(worker._child_worker_procs) == 0
        mock_popen.assert_not_called()

    # 3. Ephemeral auxiliary child worker should NEVER spawn further slots
    worker.ephemeral_job_id = "test_job_1"
    with patch("torch.cuda.is_available", return_value=True), \
         patch("subprocess.Popen") as mock_popen:
        
        worker._check_and_spawn_auxiliary_slots(
            job_id="test_job_1",
            job={"model_config": {}, "training_config": {"allow_worker_slots": True}},
        )
        assert len(worker._child_worker_procs) == 0
        mock_popen.assert_not_called()

    # 4. Default configuration (allow_worker_slots omitted) must suppress slot spawning (dedicated GPU mode)
    worker.ephemeral_job_id = None
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.synchronize"), \
         patch("torch.cuda.max_memory_reserved", return_value=int(4.0 * (1024 ** 3))), \
         patch("torch.cuda.mem_get_info", return_value=(int(12.0 * (1024 ** 3)), int(16.0 * (1024 ** 3)))), \
         patch("subprocess.Popen") as mock_popen:
        
        worker._check_and_spawn_auxiliary_slots(
            job_id="test_job_1",
            job={"model_config": {}, "training_config": {}},  # allow_worker_slots omitted
        )
        assert len(worker._child_worker_procs) == 0
        mock_popen.assert_not_called()

    # 5. Explicit allow_worker_slots: True allows spawning when headroom permits
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.synchronize"), \
         patch("torch.cuda.max_memory_reserved", return_value=int(4.0 * (1024 ** 3))), \
         patch("torch.cuda.mem_get_info", return_value=(int(12.0 * (1024 ** 3)), int(16.0 * (1024 ** 3)))), \
         patch("subprocess.Popen") as mock_popen:
        
        worker._check_and_spawn_auxiliary_slots(
            job_id="test_job_1",
            job={"model_config": {}, "training_config": {"allow_worker_slots": True}},
        )
        mock_popen.assert_called()


