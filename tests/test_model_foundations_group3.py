"""Unit tests for Group 3 Model & Training Foundations improvements.

Tests:
1. Parameter weight decay decoupling (2D matrix weights decayed vs 1D biases/norms non-decayed).
2. Tied parameter deduplication in optimizer parameter groups.
3. 8-Bit AdamW (adamw_8bit) config validation and graceful fallback behavior.
4. Synthetic agent & reasoning dataset generation (generate_agent_data).
5. Target masking fidelity on generated multi-hop agent trajectories.
6. Contrastive negative sample ratio and structural correctness.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
import torch

from engine.config import ModelConfig, TrainingConfig
from engine.generate_agent_data import (
    generate_agent_dataset,
    CONTRASTIVE_NEGATIVE_SCENARIOS,
    MULTIHOP_SCENARIOS,
    SEARCH_SCENARIOS,
    PYTHON_SCENARIOS,
)
from engine.model_gpt import MicroGPT
from engine.target_masking import find_completion_spans
from engine.tool_call_data import format_tool_call_record
from engine.training_runtime import make_optimizer


class TestModelFoundationsGroup3(unittest.TestCase):
    """Test suite for Group 3 core model & training improvements."""

    def setUp(self) -> None:
        self.model_config = ModelConfig(
            vocab_size=1000,
            embedding_size=128,
            head_count=4,
            layer_count=2,
            context_length=64,
            norm_type="rmsnorm",
            mlp_type="swiglu",
        )
        self.model = MicroGPT(self.model_config)

    def test_weight_decay_decoupling_groups(self) -> None:
        """Verify 2D matrix weights get weight decay while 1D normalization weights get 0.0."""
        training_config = TrainingConfig(
            output_dir=Path(tempfile.mkdtemp()),
            learning_rate=1e-3,
            weight_decay=0.1,
            optimizer_name="adamw",
            device="cpu",
        )
        optimizer = make_optimizer(self.model, training_config)

        # There should be 2 parameter groups: decayed and non-decayed
        self.assertEqual(len(optimizer.param_groups), 2)

        decay_group = optimizer.param_groups[0]
        nodecay_group = optimizer.param_groups[1]

        self.assertEqual(decay_group["weight_decay"], 0.1)
        self.assertEqual(nodecay_group["weight_decay"], 0.0)

        # Verify all parameters in decay_group are 2D or higher
        for p in decay_group["params"]:
            self.assertGreaterEqual(p.dim(), 2)

        # Verify all parameters in nodecay_group are 1D (RMSNorm scale weights)
        for p in nodecay_group["params"]:
            self.assertLess(p.dim(), 2)

    def test_tied_weights_not_duplicated_in_optimizer(self) -> None:
        """Verify tied weights (token_embedding.weight == lm_head.weight) are not added twice."""
        training_config = TrainingConfig(
            output_dir=Path(tempfile.mkdtemp()),
            learning_rate=1e-3,
            weight_decay=0.05,
            optimizer_name="adamw",
            device="cpu",
        )
        optimizer = make_optimizer(self.model, training_config)

        total_optimizer_params = sum(len(g["params"]) for g in optimizer.param_groups)
        unique_model_params = len(list(self.model.parameters()))

        self.assertEqual(total_optimizer_params, unique_model_params)

    def test_config_validation_accepts_adamw_8bit(self) -> None:
        """Verify TrainingConfig accepts adamw_8bit without raising ValueError."""
        config = TrainingConfig(
            output_dir=Path(tempfile.mkdtemp()),
            optimizer_name="adamw_8bit",
        )
        # Should validate cleanly
        config.validate()

    def test_adamw_8bit_cpu_fallback(self) -> None:
        """Verify requesting adamw_8bit on CPU emits a UserWarning and falls back to AdamW."""
        training_config = TrainingConfig(
            output_dir=Path(tempfile.mkdtemp()),
            optimizer_name="adamw_8bit",
            device="cpu",
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            optimizer = make_optimizer(self.model, training_config)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertTrue(any("8-bit AdamW" in str(w.message) or "bitsandbytes" in str(w.message) for w in captured))

    def test_generate_agent_dataset_creates_valid_jsonl(self) -> None:
        """Verify generate_agent_dataset writes properly structured OpenAI agent records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "agent_corpus.jsonl"
            count = generate_agent_dataset(
                output_path=out_file,
                sample_count=20,
                contrastive_ratio=0.35,
                multihop_ratio=0.35,
                seed=999,
            )
            self.assertEqual(count, 20)
            self.assertTrue(out_file.exists())

            lines = out_file.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 20)

            has_contrastive = False
            has_multihop = False
            has_single_hop = False

            for line in lines:
                record = json.loads(line)
                self.assertIn("tools", record)
                self.assertIn("messages", record)
                self.assertGreaterEqual(len(record["messages"]), 2)

                # Check if any message is assistant
                assistant_msgs = [m for m in record["messages"] if m.get("role") == "assistant"]
                self.assertTrue(len(assistant_msgs) >= 1)

                # Check thought presence
                self.assertTrue(any("<thought>" in m.get("content", "") for m in assistant_msgs))

                tool_msgs = [m for m in record["messages"] if m.get("role") == "tool"]
                if len(tool_msgs) == 0:
                    has_contrastive = True
                elif len(tool_msgs) == 1:
                    has_single_hop = True
                elif len(tool_msgs) >= 2:
                    has_multihop = True

            self.assertTrue(has_contrastive, "Expected at least one contrastive sample")
            self.assertTrue(has_single_hop or has_multihop, "Expected at least one tool call sample")

    def test_agent_data_target_masking_integration(self) -> None:
        """Verify generated agent records are cleanly masked with prompt targets excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "agent_corpus.jsonl"
            generate_agent_dataset(out_file, sample_count=5, contrastive_ratio=0.2, multihop_ratio=0.5, seed=123)

            for line in out_file.read_text(encoding="utf-8").strip().split("\n"):
                record = json.loads(line)
                rendered = format_tool_call_record(record)
                spans = find_completion_spans(rendered)
                self.assertGreaterEqual(len(spans), 1)

                completion_text = "".join(rendered[s:e] for s, e in spans)
                # Ensure <thought> is unmasked (in completion)
                self.assertIn("<thought>", completion_text)
                # Ensure User: prompt header is masked (not in completion)
                self.assertNotIn("User: ", completion_text)


if __name__ == "__main__":
    unittest.main()
