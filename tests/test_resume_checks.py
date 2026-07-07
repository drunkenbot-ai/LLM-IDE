from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from llm_trainer.config import ModelConfig, TrainingConfig
from llm_trainer.resume_checks import _resume_checkpoint_for, _validate_resume_compatibility


class ResumeChecksTests(unittest.TestCase):
    def test_resume_checkpoint_for_returns_none_when_resume_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(output_dir=Path(temp_dir), resume=False)
            self.assertIsNone(_resume_checkpoint_for(config))

    def test_validate_resume_compatibility_returns_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "model_out"
            checkpoints = output_dir / "checkpoints"
            data_dir = root / "data"
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoints.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": {"<pad>": 0}, "unk_token": "<pad>"},
            }
            (data_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json), encoding="utf-8")
            (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json), encoding="utf-8")

            model_config = ModelConfig(vocab_size=64, context_length=16, embedding_size=32, head_count=4, layer_count=2)
            model_config.validate()
            checkpoint_path = checkpoints / "checkpoint_1.pt"
            torch.save({"model_config": {"vocab_size": 64, "context_length": 16, "embedding_size": 32, "head_count": 4, "layer_count": 2}}, checkpoint_path)

            training_config = TrainingConfig(output_dir=output_dir, resume=True, require_compatible_resume=True)
            resolved = _validate_resume_compatibility(data_dir, data_dir / "tokenizer.json", model_config, training_config)
            self.assertEqual(resolved, checkpoint_path)


if __name__ == "__main__":
    unittest.main()
