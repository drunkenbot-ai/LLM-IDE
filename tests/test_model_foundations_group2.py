"""Unit tests for Group 2 Model & Training Foundations improvements.

Tests:
1. PackedInstructionDataset: Sequence packing multiple discrete samples into fixed context_length bins.
2. Static tensor shapes, target masking fidelity, and padding elimination in PackedInstructionDataset.
3. ChatML Role Boundary tokens (<|im_start|>, <|im_end|>) in tokenizer and target masking.
4. Multi-hop agent trajectory formatting with sequential thought, tool_calls, and tool_results.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from engine.target_masking import (
    IGNORE_INDEX,
    PackedInstructionDataset,
    find_completion_spans,
)
from engine.tokenizer import (
    CHATML_CHAT_TEMPLATE,
    ROLE_TOKENS,
    SPECIAL_TOKENS,
)
from engine.tool_call_data import (
    STANDARD_AGENT_TOOLS,
    create_multihop_agent_trajectory,
    format_tool_call_record,
)


class TestModelFoundationsGroup2(unittest.TestCase):
    """Test suite for Group 2 core model improvements."""

    def test_role_tokens_registered_in_tokenizer(self) -> None:
        """Verify <|im_start|> and <|im_end|> are registered in ROLE_TOKENS and SPECIAL_TOKENS."""
        self.assertIn("<|im_start|>", ROLE_TOKENS)
        self.assertIn("<|im_end|>", ROLE_TOKENS)
        self.assertIn("<|im_start|>", SPECIAL_TOKENS)
        self.assertIn("<|im_end|>", SPECIAL_TOKENS)
        self.assertIn("<|im_start|>", CHATML_CHAT_TEMPLATE)
        self.assertIn("<|im_end|>", CHATML_CHAT_TEMPLATE)

    def test_find_completion_spans_with_chatml(self) -> None:
        """Verify find_completion_spans detects <|im_start|>assistant spans and includes <|im_end|>."""
        chatml_text = (
            "<|im_start|>system\nYou are a helpful frontier assistant.<|im_end|>\n"
            "<|im_start|>user\nWhat is the square of 9?<|im_end|>\n"
            "<|im_start|>assistant\n<thought>9 * 9 = 81</thought>\nThe square of 9 is 81.<|im_end|>\n"
        )
        spans = find_completion_spans(chatml_text)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        completion = chatml_text[start:end]
        self.assertIn("<thought>9 * 9 = 81</thought>", completion)
        self.assertIn("The square of 9 is 81.", completion)
        self.assertIn("<|im_end|>", completion)
        # Ensure system and user prompts are excluded
        self.assertNotIn("What is the square of 9?", completion)
        self.assertNotIn("helpful frontier assistant", completion)

    def test_packed_instruction_dataset_packs_multiple_samples(self) -> None:
        """Verify PackedInstructionDataset packs multiple short discrete samples into context_length bins."""
        # Create 4 discrete samples separated by EOS (token id 3)
        # Sample 1: 20 tokens, Sample 2: 30 tokens, Sample 3: 25 tokens, Sample 4: 15 tokens
        # Total tokens = 90. Context length = 64.
        # Bin 1 should pack Sample 1 + Sample 2 (20 + 30 = 50 tokens <= 64).
        # Bin 2 should pack Sample 3 + Sample 4 (25 + 15 = 40 tokens <= 64).
        tokens_list: list[int] = []
        targets_list: list[int] = []

        # Sample 1 (20 tokens)
        for i in range(19):
            tokens_list.append(100 + i)
            targets_list.append(-100 if i < 10 else 100 + i)
        tokens_list.append(3)  # EOS
        targets_list.append(3)

        # Sample 2 (30 tokens)
        for i in range(29):
            tokens_list.append(200 + i)
            targets_list.append(-100 if i < 15 else 200 + i)
        tokens_list.append(3)  # EOS
        targets_list.append(3)

        # Sample 3 (25 tokens)
        for i in range(24):
            tokens_list.append(300 + i)
            targets_list.append(-100 if i < 12 else 300 + i)
        tokens_list.append(3)  # EOS
        targets_list.append(3)

        # Sample 4 (15 tokens)
        for i in range(14):
            tokens_list.append(400 + i)
            targets_list.append(-100 if i < 7 else 400 + i)
        tokens_list.append(3)  # EOS
        targets_list.append(3)

        tokens = np.array(tokens_list, dtype=np.int64)
        targets = np.array(targets_list, dtype=np.int64)

        packed_ds = PackedInstructionDataset(
            tokens=tokens,
            targets=targets,
            context_length=64,
            eos_token_id=3,
            pad_token_id=0,
        )

        self.assertEqual(packed_ds.total_unpacked_samples, 4)
        self.assertEqual(len(packed_ds), 2)  # Packed into exactly 2 bins

        # Inspect Bin 0
        inp_0, tgt_0 = packed_ds[0]
        self.assertEqual(inp_0.shape, (64,))
        self.assertEqual(tgt_0.shape, (64,))

        # Inspect Bin 1
        inp_1, tgt_1 = packed_ds[1]
        self.assertEqual(inp_1.shape, (64,))
        self.assertEqual(tgt_1.shape, (64,))

        # Verify trailing padding uses IGNORE_INDEX for targets
        self.assertEqual(tgt_0[-1].item(), IGNORE_INDEX)
        self.assertEqual(inp_0[-1].item(), 0)

    def test_multihop_agent_trajectory_formatting(self) -> None:
        """Verify create_multihop_agent_trajectory formats sequential thought, tool call, and tool result turns."""
        trajectory = [
            {"role": "user", "content": "Find the weather in Tokyo and calculate wind chill if temp is 5C and wind is 20km/h."},
            {
                "role": "assistant",
                "thought": "I should search for current weather in Tokyo first.",
                "tool_calls": [
                    {
                        "id": "call_tokyo_weather",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query": "current weather Tokyo"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_tokyo_weather",
                "content": '{"temperature": 5, "wind_speed": 20, "conditions": "Clear"}',
            },
            {
                "role": "assistant",
                "thought": "Now I will compute the wind chill index using Python.",
                "tool_calls": [
                    {
                        "id": "call_wind_chill",
                        "type": "function",
                        "function": {
                            "name": "python_interpreter",
                            "arguments": '{"code": "13.12 + 0.6215*5 - 11.37*(20**0.16) + 0.3965*5*(20**0.16)"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_wind_chill",
                "content": "2.4",
            },
            {
                "role": "assistant",
                "content": "The current temperature in Tokyo is 5°C with 20 km/h winds, giving an effective wind chill of approximately 2.4°C.",
            },
        ]

        record = create_multihop_agent_trajectory(trajectory, tools=STANDARD_AGENT_TOOLS)
        rendered = format_tool_call_record(record)

        # Verify key trajectory elements are present
        self.assertIn("Tools:\n<tools>", rendered)
        self.assertIn("User: Find the weather in Tokyo", rendered)
        self.assertIn("<thought>I should search for current weather in Tokyo first.</thought>", rendered)
        self.assertIn("call_tokyo_weather", rendered)
        self.assertIn('<tool_result id="call_tokyo_weather">', rendered)
        self.assertIn("<thought>Now I will compute the wind chill index using Python.</thought>", rendered)
        self.assertIn("call_wind_chill", rendered)
        self.assertIn('<tool_result id="call_wind_chill">', rendered)
        self.assertIn("The current temperature in Tokyo is 5°C", rendered)

        # Verify completion spans unmask assistant turns while masking tool results
        spans = find_completion_spans(rendered)
        self.assertTrue(len(spans) >= 2)
        extracted_text = "".join(rendered[s:e] for s, e in spans)
        self.assertIn("<thought>I should search", extracted_text)
        self.assertIn("<thought>Now I will compute", extracted_text)
        self.assertIn("effective wind chill", extracted_text)
        # Verify tool execution results from environment are masked out (not in completion)
        self.assertNotIn('<tool_result id="call_tokyo_weather">', extracted_text)
        self.assertNotIn('<tool_result id="call_wind_chill">', extracted_text)


if __name__ == "__main__":
    unittest.main()
