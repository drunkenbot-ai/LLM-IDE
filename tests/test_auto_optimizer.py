import pytest
from engine.config import ModelConfig
from engine.training_planning import optimize_training_hyperparameters

def test_compute_optimal_hyperparameters_base_model():
    cfg = ModelConfig(
        vocab_size=50257,
        context_length=2048,
        embedding_size=4096,
        head_count=32,
        layer_count=32,
    )
    res = optimize_training_hyperparameters(
        model_config=cfg,
        target_vram_gb=16.0,
        train_tokens=1_000_000,
        training_mode="pretrain",
        device_type="cuda",
    )
    assert res["batch_size"] >= 1
    assert res["gradient_accumulation"] >= 1
    assert res["epochs"] >= 1
    assert res["learning_rate"] > 0
    assert res["weight_decay"] > 0
    assert res["warmup_steps"] >= 10
    assert res["eval_interval"] >= 1
    assert res["save_interval"] >= 1
    assert "Pre-training" in res["summary_text"]

def test_compute_optimal_hyperparameters_finetune_lora():
    cfg = ModelConfig(
        vocab_size=128256,
        context_length=4096,
        embedding_size=4096,
        head_count=32,
        layer_count=32,
    )
    res = optimize_training_hyperparameters(
        model_config=cfg,
        target_vram_gb=16.0,
        train_tokens=200_000,
        training_mode="fine_tune",
        device_type="cuda",
    )
    assert res["batch_size"] >= 1
    assert res["epochs"] >= 1
    assert res["learning_rate"] == 0.0002
    assert "LoRA" in res["summary_text"]

def test_compute_optimal_hyperparameters_microllm():
    cfg = ModelConfig(
        vocab_size=8000,
        context_length=1024,
        embedding_size=768,
        head_count=12,
        layer_count=12,
    )
    res = optimize_training_hyperparameters(
        model_config=cfg,
        target_vram_gb=24.0,
        train_tokens=50_000,
        training_mode="pretrain",
        device_type="cuda",
    )
    assert res["batch_size"] >= 8
    assert res["epochs"] >= 1
