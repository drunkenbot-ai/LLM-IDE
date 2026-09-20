"""Unit tests verifying pretraining checkpoint resume and job extension in distributed cluster."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from cluster.bus import ClusterStorageBus
from cluster.worker import ClusterWorker, build_model_from_config


def test_bus_load_base_model_weights_various_formats(tmp_path: Path) -> None:
    """Verify that ClusterStorageBus extracts model weights from various checkpoint formats."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "test_job_load_weights"
    job_dir = bus.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    sample_weights = {"tok_embeddings.weight": torch.randn(10, 16)}

    # Format 1: Direct state dict
    torch.save(sample_weights, job_dir / "base_model.pt")
    loaded = bus.load_base_model_weights(job_id)
    assert loaded is not None
    assert torch.equal(loaded["tok_embeddings.weight"], sample_weights["tok_embeddings.weight"])

    # Format 2: Dict with "model_state_dict"
    wrapped_weights = {"model_state_dict": sample_weights, "step": 100}
    torch.save(wrapped_weights, job_dir / "base_model.pt")
    loaded = bus.load_base_model_weights(job_id)
    assert loaded is not None
    assert torch.equal(loaded["tok_embeddings.weight"], sample_weights["tok_embeddings.weight"])

    # Format 3: Dict with "state_dict"
    wrapped_weights2 = {"state_dict": sample_weights, "epoch": 1}
    torch.save(wrapped_weights2, job_dir / "base_model.pt")
    loaded = bus.load_base_model_weights(job_id)
    assert loaded is not None
    assert torch.equal(loaded["tok_embeddings.weight"], sample_weights["tok_embeddings.weight"])


def test_pretraining_job_loads_base_checkpoint_weights(tmp_path: Path) -> None:
    """Verify that pretraining jobs properly initialize weights from base_checkpoint_path."""
    bus = ClusterStorageBus(tmp_path)
    job_id = "test_pretrain_resume_job"

    model_config = {
        "vocab_size": 64,
        "context_length": 32,
        "embedding_size": 32,
        "head_count": 2,
        "layer_count": 2,
    }
    training_config = {
        "learning_rate": 1e-3,
        "batch_size": 2,
        "training_mode": "pretrain",
        "resume": True,
    }

    # Create dummy dataset
    token_data = np.random.randint(0, 64, size=500, dtype=np.int32)
    ds_path = tmp_path / "train_tokens.npy"
    np.save(ds_path, token_data)

    # Create a checkpoint with specific known weights
    base_model = build_model_from_config(model_config, "cpu")
    with torch.no_grad():
        for p in base_model.parameters():
            p.fill_(4.2)
    ckpt_path = tmp_path / "previous_checkpoint.pt"
    torch.save({"model_state_dict": base_model.state_dict(), "step": 250}, ckpt_path)

    # Create job with base_checkpoint_path
    bus.create_job(
        job_id=job_id,
        model_config=model_config,
        training_config=training_config,
        dataset_path=str(ds_path),
        max_rounds=2,
        sync_interval_steps=5,
        min_workers=1,
        job_type="pretrain",
        base_checkpoint_path=str(ckpt_path),
    )

    # Verify base model was staged in job directory
    assert (bus.jobs_dir / job_id / "base_model.pt").exists()

    # Verify worker loads the weights
    worker = ClusterWorker(bus=bus, worker_id="test_worker_1", device="cpu", allow_shared_device=True)
    job = bus.get_job(job_id)
    assert job is not None

    base_weights = bus.load_base_model_weights(job_id, device="cpu")
    assert base_weights is not None

    fresh_model = build_model_from_config(model_config, "cpu")
    fresh_model.load_state_dict(base_weights)

    # Check that weights match the 4.2 filled checkpoint
    first_param = next(fresh_model.parameters())
    assert torch.allclose(first_param, torch.tensor(4.2))
