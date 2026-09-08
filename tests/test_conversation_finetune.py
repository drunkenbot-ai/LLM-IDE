"""Unit tests for conversation fine-tuning and multi-turn chat handling."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from engine.config import ModelConfig
from engine.data_core import _extract_structured_text
from engine.microgpt_chat import STOP_SEQUENCES, MicroGPTChatSession
from engine.target_masking import (
    IGNORE_INDEX,
    InstructionDataset,
    find_completion_spans,
    mask_prompt_targets,
)


class TestConversationFineTune(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(special_tokens=["<unk>", "<|endoftext|>", "<pad>", "<bos>", "<eos>"], vocab_size=500)
        sample_corpus = [
            "System: You are a conversational assistant.\n"
            "User: Hi\nAssistant: Hello! How can I help?\n"
            "User: Tell me more.\nAssistant: Certainly, here is more detail.\n<eos>\n",
        ]
        self.tokenizer.train_from_iterator(sample_corpus, trainer)

    def test_multi_turn_spans_detection(self) -> None:
        record = {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"},
            ]
        }
        text = _extract_structured_text(record, "conversation")
        self.assertIn("System: Be concise.", text)
        self.assertIn("User: Question 1", text)
        self.assertIn("Assistant: Answer 1", text)

        spans = find_completion_spans(text)
        self.assertEqual(len(spans), 2)
        span1 = text[spans[0][0]:spans[0][1]]
        span2 = text[spans[1][0]:spans[1][1]]
        self.assertEqual(span1.strip(), "Answer 1")
        self.assertEqual(span2.strip(), "Answer 2")

    def test_multi_turn_target_masking(self) -> None:
        text = (
            "System: Sys\n"
            "User: Q1\n"
            "Assistant: A1\n"
            "User: Q2\n"
            "Assistant: A2\n"
            "<eos>"
        )
        enc = self.tokenizer.encode(text)
        targets = mask_prompt_targets(text, self.tokenizer, enc.ids, enc.offsets)
        self.assertEqual(len(targets), len(enc.ids))

        # First tokens (System: Sys) must be -100
        self.assertEqual(targets[0], IGNORE_INDEX)

        # Unmasked tokens must exist for both A1 and A2
        unmasked = [tid for tid in targets if tid != IGNORE_INDEX]
        self.assertGreater(len(unmasked), 0)

        # Trailing <eos> must be unmasked
        self.assertEqual(targets[-1], enc.ids[-1])

    def test_instruction_dataset_keeps_multi_turn_intact(self) -> None:
        # Single 5-turn conversation ending in EOS=4
        tokens = np.array([10, 11, 12, 13, 14, 15, 16, 17, 4], dtype=np.int64)
        targets = np.array([-100, -100, 12, -100, -100, 15, 16, 17, 4], dtype=np.int64)
        ds = InstructionDataset(tokens, targets=targets, context_length=32, eos_token_id=4)
        # Should be exactly 1 discrete sample containing all turns
        self.assertEqual(len(ds), 1)
        x, y = ds[0]
        self.assertEqual(len(x), len(tokens) - 1)
        self.assertEqual(len(y), len(targets) - 1)

    def test_microgpt_chat_stop_sequences_contain_user(self) -> None:
        self.assertIn("\nUser:", STOP_SEQUENCES)
        self.assertIn("\nSystem:", STOP_SEQUENCES)
        self.assertIn("<eos>", STOP_SEQUENCES)

    def test_microgpt_chat_render_prompt_pruning(self) -> None:
        session = MicroGPTChatSession.__new__(MicroGPTChatSession)
        session.config = ModelConfig(vocab_size=500, context_length=64)
        session.tokenizer = self.tokenizer
        session._messages = [
            {"role": "user", "content": "Very old question that should be dropped"},
            {"role": "assistant", "content": "Very old answer that should be dropped"},
            {"role": "user", "content": "Recent question"},
            {"role": "assistant", "content": "Recent answer"},
        ]
        prompt = "Current question"
        system_prompt = "Permanent System Instruction"

        rendered = session._render_prompt(
            prompt, system_prompt, reasoning_effort="None", thinking_enabled=False, max_tokens=16
        )

        # Permanent system instruction and latest turn MUST be preserved
        self.assertIn("System: Permanent System Instruction", rendered)
        self.assertIn("User: Current question", rendered)
        self.assertIn("Assistant:", rendered)

        # Token length must fit inside context_length - 16
        enc = self.tokenizer.encode(rendered)
        self.assertLessEqual(len(enc.ids), session.config.context_length - 16)


if __name__ == "__main__":
    unittest.main()
