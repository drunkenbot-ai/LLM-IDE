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
    """Verify dynamic VRAM slot calculation for various GPU capacities."""
    # Test with simulated CUDA device info
    with patch("torch.cuda.is_available", return_value=True):
        # 1. 8 GB GPU with 8 GB job requirement
        mock_props_8g = MagicMock()
        mock_props_8g.total_memory = int(8.0 * (1024 ** 3))
        with patch("torch.cuda.get_device_properties", return_value=mock_props_8g):
            slots, est_gb, node_gb = calculate_optimal_worker_slots(
                model_cfg={},
                training_cfg={"vram_required_gb": 7.5},
                device_str="cuda:0",
            )
            assert slots == 1
            assert node_gb == 8.0

        # 2. 16 GB GPU with 7.5 GB job requirement -> should spawn 2 slots
        mock_props_16g = MagicMock()
        mock_props_16g.total_memory = int(15.93 * (1024 ** 3))
        with patch("torch.cuda.get_device_properties", return_value=mock_props_16g):
            slots, est_gb, node_gb = calculate_optimal_worker_slots(
                model_cfg={},
                training_cfg={"vram_required_gb": 7.5},
                device_str="cuda:0",
            )
            assert slots == 2

        # 3. 24 GB GPU with 7.5 GB job requirement -> should spawn 3 slots
        mock_props_24g = MagicMock()
        mock_props_24g.total_memory = int(24.0 * (1024 ** 3))
        with patch("torch.cuda.get_device_properties", return_value=mock_props_24g):
            slots, est_gb, node_gb = calculate_optimal_worker_slots(
                model_cfg={},
                training_cfg={"vram_required_gb": 7.5},
                device_str="cuda:0",
            )
            assert slots == 3
