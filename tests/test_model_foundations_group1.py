"""Unit tests for Group 1 Model & Training Foundations improvements.

Tests:
1. SwiGLU canonical intermediate dimension calculation (2/3 * 4 * d = 8/3 * d rounded to 64).
2. ModelConfig resolved_intermediate_size behavior for SwiGLU, GELU, and explicit overrides.
3. SwiGLU MLP instantiation, parameter count savings, and forward pass.
4. Updated default rope_theta (500000.0).
5. Reasoning tokens (<thought>, </thought>) registration in tokenizer and target masking.
6. Contrastive negative tool-call sample creation and prompt/completion formatting.
"""

from __future__ import annotations

import unittest
import torch

from engine.config import ModelConfig
from engine.model_transformer import MLP, Block
from engine.target_masking import find_completion_spans
from engine.tokenizer import REASONING_TOKENS, SPECIAL_TOKENS, TOOL_TOKENS
from engine.tool_call_data import (
    STANDARD_AGENT_TOOLS,
    create_contrastive_negative_sample,
    format_tool_call_record,
)
from engine.training_planning import estimate_parameter_breakdown


class TestModelFoundationsGroup1(unittest.TestCase):
    """Test suite for Group 1 core model improvements."""

    def test_swiglu_resolved_intermediate_size_canonical(self) -> None:
        """Verify SwiGLU intermediate size matches 8/3 * d rounded to multiple of 64."""
        # For d=256: 8/3 * 256 = 682.66 -> rounded up to multiple of 64 is 704
        config = ModelConfig(vocab_size=1000, embedding_size=256, mlp_type="swiglu")
        self.assertEqual(config.resolved_intermediate_size(), 704)

        # For d=512: 8/3 * 512 = 1365.33 -> rounded up to multiple of 64 is 1408
        config_512 = ModelConfig(vocab_size=1000, embedding_size=512, mlp_type="swiglu")
        self.assertEqual(config_512.resolved_intermediate_size(), 1408)

        # For GELU: defaults to 4 * d = 1024
        config_gelu = ModelConfig(vocab_size=1000, embedding_size=256, mlp_type="gelu")
        self.assertEqual(config_gelu.resolved_intermediate_size(), 1024)

    def test_swiglu_explicit_intermediate_size_override(self) -> None:
        """Verify explicit intermediate_size override is respected."""
        config = ModelConfig(vocab_size=1000, embedding_size=256, mlp_type="swiglu", intermediate_size=800)
        self.assertEqual(config.resolved_intermediate_size(), 800)

    def test_swiglu_mlp_forward_and_parameter_savings(self) -> None:
        """Verify SwiGLU MLP instantiates, executes forward pass, and uses fewer parameters than 4x."""
        config_swiglu = ModelConfig(
            vocab_size=1000,
            embedding_size=256,
            mlp_type="swiglu",
            bias=False,
        )
        mlp = MLP(config_swiglu)
        x = torch.randn(2, 16, 256)
        out = mlp(x)
        self.assertEqual(out.shape, (2, 16, 256))

        # Check parameter savings:
        # Canonical SwiGLU: 3 * (256 * 704) = 540,672
        # Naive 4x SwiGLU: 3 * (256 * 1024) = 786,432
        swiglu_params = sum(p.numel() for p in mlp.parameters())
        self.assertEqual(swiglu_params, 3 * 256 * 704)
        self.assertLess(swiglu_params, 3 * 256 * 1024)

    def test_rope_theta_default(self) -> None:
        """Verify default rope_theta is updated to 500000.0 for long context stability."""
        config = ModelConfig(vocab_size=1000)
        self.assertEqual(config.rope_theta, 500000.0)

    def test_reasoning_tokens_registered(self) -> None:
        """Verify <thought> and </thought> are present in REASONING_TOKENS, TOOL_TOKENS, and SPECIAL_TOKENS."""
        self.assertIn("<thought>", REASONING_TOKENS)
        self.assertIn("</thought>", REASONING_TOKENS)
        self.assertIn("<thought>", TOOL_TOKENS)
        self.assertIn("</thought>", TOOL_TOKENS)
        self.assertIn("<thought>", SPECIAL_TOKENS)
        self.assertIn("</thought>", SPECIAL_TOKENS)

    def test_target_masking_unmasks_thought_blocks(self) -> None:
        """Verify find_completion_spans unmasks standalone <thought> completions."""
        transcript = (
            "User: Calculate 25 * 40\n"
            "<thought>Let me multiply 25 by 40: 25 * 4 = 100, so 25 * 40 = 1000.</thought>\n"
            "<tool_calls>[{\"name\": \"python_interpreter\", \"arguments\": {\"code\": \"25 * 40\"}}]</tool_calls>\n"
            "<eos>"
        )
        spans = find_completion_spans(transcript)
        self.assertTrue(len(spans) >= 1)
        start, end = spans[0]
        completion = transcript[start:end]
        self.assertIn("<thought>", completion)
        self.assertIn("<tool_calls>", completion)
        # Ensure user prompt is excluded from completion
        self.assertNotIn("User: Calculate", completion)

    def test_contrastive_negative_sample_creation(self) -> None:
        """Verify create_contrastive_negative_sample produces valid OpenAI-shaped record."""
        sample = create_contrastive_negative_sample(
            prompt="What is the capital of Japan?",
            direct_answer="The capital of Japan is Tokyo.",
        )
        self.assertIn("tools", sample)
        self.assertEqual(len(sample["tools"]), 2)
        self.assertEqual(sample["messages"][0], {"role": "user", "content": "What is the capital of Japan?"})
        self.assertEqual(sample["messages"][1], {"role": "assistant", "content": "The capital of Japan is Tokyo."})

        # Format through format_tool_call_record
        rendered = format_tool_call_record(sample)
        self.assertIn("Tools:\n<tools>", rendered)
        self.assertIn("User: What is the capital of Japan?", rendered)
        self.assertIn("Assistant: The capital of Japan is Tokyo.", rendered)
        # Crucially: it contains tools in system prompt, but NO <tool_calls> tag!
        self.assertNotIn("<tool_calls>", rendered)

    def test_prompt_completion_negative_sample_formatting(self) -> None:
        """Verify flat prompt-response records with tools are rendered as negative contrastive samples."""
        flat_record = {
            "tools": STANDARD_AGENT_TOOLS,
            "prompt": "Explain photosynthesis in one sentence.",
            "response": "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        }
        rendered = format_tool_call_record(flat_record)
        self.assertIn("Tools:\n<tools>", rendered)
        self.assertIn("User: Explain photosynthesis in one sentence.", rendered)
        self.assertIn("Assistant: Photosynthesis is the process by which plants convert sunlight into chemical energy.", rendered)
        self.assertNotIn("<tool_calls>", rendered)


if __name__ == "__main__":
    unittest.main()
