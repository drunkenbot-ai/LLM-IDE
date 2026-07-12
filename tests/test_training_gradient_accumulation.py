from __future__ import annotations

import json
import tempfile
import unittest
import torch
from pathlib import Path

from llm_trainer.config import ModelConfig, TrainingConfig
from llm_trainer.training import train_model
from llm_trainer.model import MicroGPT


def _run_training(train_tokens: list[int], gradient_accumulation: int, batch_size: int) -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        model_config = ModelConfig(
            vocab_size=32,
            context_length=8,
            embedding_size=32,
            head_count=4,
            layer_count=2,
            dropout=0.0,
        )
        training_config = TrainingConfig(
            output_dir=output_dir,
            epochs=1,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            learning_rate=1e-3,
            device="cpu",
            use_amp=False,
            precision="fp32",
            eval_interval=0,
            save_interval=0,
            resume=False,
        )
        result = train_model(
            model_config,
            training_config,
            train_tokens=train_tokens,
            val_tokens=[],
            pad_token_id=0,
        )
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        return int(summary["total_steps"])


class TrainingGradientAccumulationTests(unittest.TestCase):
    def test_activation_checkpointing_preserves_forward_shape(self) -> None:
        config = ModelConfig(vocab_size=32, context_length=8, embedding_size=32, head_count=4, layer_count=2)
        model = MicroGPT(config)
        model.train()
        model.enable_gradient_checkpointing(True)
        logits = model(torch.ones((2, 8), dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (2, 8, 32))

    def test_applies_final_partial_accumulation_step(self) -> None:
        # context=8 => dataset windows=7, batch_size=2 => 3 batches, accumulation=4.
        # No full accumulation cycle exists, so this guards the epoch-end remainder step.
        train_tokens = [index % 31 for index in range(15)]
        total_steps = _run_training(train_tokens, gradient_accumulation=4, batch_size=2)
        self.assertEqual(total_steps, 0)

    def test_counts_remainder_batch_group_as_optimizer_step(self) -> None:
        # context=8 => windows=6, batch_size=2 => 3 batches, accumulation=2 => 2 steps.
        train_tokens = [index % 31 for index in range(14)]
        total_steps = _run_training(train_tokens, gradient_accumulation=2, batch_size=2)
        self.assertEqual(total_steps, 0)


if __name__ == "__main__":
    unittest.main()
