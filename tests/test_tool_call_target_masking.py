"""Unit tests for tool-call fine-tuning target masking, classification, and runtime handling."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from engine.data_core import _structured_record_kind, load_structured_json_documents
from engine.microgpt_chat import STOP_SEQUENCES, MicroGPTChatSession
from engine.target_masking import IGNORE_INDEX, find_completion_spans, mask_prompt_targets
from engine.tokenizer import SPECIAL_TOKENS, TOOL_TOKENS, Tokenizer
from engine.tool_call_data import _format_message, _is_tool_message
from tokenizers import Tokenizer as FastTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


class TestToolCallTargetMasking(unittest.TestCase):
    def setUp(self) -> None:
        self.fast_tokenizer = FastTokenizer(BPE(unk_token="<unk>"))
        self.fast_tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(
            special_tokens=["<unk>", "<|endoftext|>", "<pad>", "<bos>", "<eos>", *TOOL_TOKENS],
            vocab_size=500,
        )
        sample_corpus = [
            "Tools:\n<tools>[]</tools>\nUser: What is weather?\nAssistant: <tool_calls>[{\"name\":\"get_weather\"}]</tool_calls>\n"
            "<tool_result id=\"1\">{\"temp\": 72}</tool_result>\nAssistant: It is 72 degrees.\n<eos>\n",
            "User: Calculate\nAssistant: <CALL>calc(2+2)</CALL>\n<tool_result>4</tool_result>\nAssistant: 4\n<eos>\n",
        ]
        self.fast_tokenizer.train_from_iterator(sample_corpus, trainer)

    def test_tool_tokens_registered_in_tokenizer(self) -> None:
        for token in [
            "<tools>",
            "</tools>",
            "<tool_calls>",
            "</tool_calls>",
            "<tool_result>",
            "</tool_result>",
            "<CALL>",
            "</CALL>",
            "<thought>",
            "</thought>",
        ]:
            self.assertIn(token, TOOL_TOKENS)
            self.assertIn(token, SPECIAL_TOKENS)

    def test_record_kind_classification(self) -> None:
        # Standard OpenAI record with top-level tools
        openai_rec = {
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "messages": [
                {"role": "user", "content": "Search for news"},
                {"role": "assistant", "tool_calls": [{"id": "1", "type": "function"}]},
            ],
        }
        self.assertEqual(_structured_record_kind(openai_rec), "tool_call")

        # Tag-based record with nested <CALL> (e.g. tool_call_training_part_1.jsonl)
        tag_rec = {
            "messages": [
                {"role": "system", "content": "You can call functions with <CALL>"},
                {"role": "user", "content": "What time is it?"},
                {"role": "assistant", "content": "<CALL>get_time()</CALL>"},
            ]
        }
        self.assertEqual(_structured_record_kind(tag_rec), "tool_call")

        # Record with role='tool'
        tool_role_rec = {
            "messages": [
                {"role": "user", "content": "Fetch"},
                {"role": "tool", "content": "data"},
            ]
        }
        self.assertEqual(_structured_record_kind(tool_role_rec), "tool_call")

        # Generic conversation record without tool elements
        conv_rec = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        self.assertEqual(_structured_record_kind(conv_rec), "conversation")

    def test_format_message_assistant_with_empty_content_preserves_label(self) -> None:
        # When assistant generates tool_calls with None/empty content
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_stock_price", "arguments": '{"ticker": "AAPL"}'},
                }
            ],
        }
        formatted = _format_message(msg)
        self.assertTrue(formatted.startswith("Assistant: <tool_calls>"))
        self.assertIn("call_abc", formatted)
        self.assertIn("get_stock_price", formatted)

    def test_format_message_assistant_with_content_and_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": "Checking the price for you...",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "check_price", "arguments": "{}"},
                }
            ],
        }
        formatted = _format_message(msg)
        self.assertTrue(formatted.startswith("Assistant: Checking the price for you...\n<tool_calls>\n"))
        self.assertTrue(formatted.endswith("\n</tool_calls>"))
        self.assertIn("call_123", formatted)
        self.assertIn("check_price", formatted)

    def test_single_tool_call_target_masking(self) -> None:
        text = (
            "Tools:\n<tools>[{\"name\": \"get_weather\"}]</tools>\n"
            "User: What's the weather in Tokyo?\n"
            "Assistant: <tool_calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}]</tool_calls>\n"
            "<tool_result id=\"call_1\">{\"temperature\": 18, \"condition\": \"Clear\"}</tool_result>\n"
            "Assistant: The weather in Tokyo is clear and 18°C.\n"
            "<eos>"
        )

        spans = find_completion_spans(text)
        self.assertEqual(len(spans), 2)

        # Span 1: Tool call completion target
        span1_text = text[spans[0][0]:spans[0][1]].strip()
        self.assertTrue(span1_text.startswith("<tool_calls>"))
        self.assertTrue(span1_text.endswith("</tool_calls>"))
        self.assertNotIn("tool_result", span1_text)

        # Span 2: Final response completion target
        span2_text = text[spans[1][0]:spans[1][1]].strip()
        self.assertEqual(span2_text, "The weather in Tokyo is clear and 18°C.\n<eos>")

        # Check token level masking
        enc = self.fast_tokenizer.encode(text)
        targets = mask_prompt_targets(text, self.fast_tokenizer, enc.ids, enc.offsets)

        # Ensure tools block and user query are masked (-100)
        self.assertEqual(targets[0], IGNORE_INDEX)

        # Tool result block must be masked (-100)
        result_offset = text.find("<tool_result")
        result_end_offset = text.find("</tool_result>") + len("</tool_result>")
        for tid, (start, end) in zip(targets, enc.offsets):
            if start >= result_offset and end <= result_end_offset:
                self.assertEqual(
                    tid,
                    IGNORE_INDEX,
                    f"Observation token in range [{start}, {end}] was not masked!",
                )

        # Final EOS token must NOT be masked
        self.assertNotEqual(targets[-1], IGNORE_INDEX)

    def test_sequential_tool_calls_target_masking(self) -> None:
        text = (
            "Tools:\n<tools>[...]</tools>\n"
            "User: Plan my trip\n"
            "Assistant: <tool_calls>[{\"name\": \"book_flight\"}]</tool_calls>\n"
            "<tool_result id=\"1\">{\"status\": \"booked\"}</tool_result>\n"
            "Assistant: <tool_calls>[{\"name\": \"book_hotel\"}]</tool_calls>\n"
            "<tool_result id=\"2\">{\"hotel\": \"confirmed\"}</tool_result>\n"
            "Assistant: Flight and hotel are booked!\n"
            "<eos>"
        )

        spans = find_completion_spans(text)
        self.assertEqual(len(spans), 3)

        span1 = text[spans[0][0]:spans[0][1]].strip()
        span2 = text[spans[1][0]:spans[1][1]].strip()
        span3 = text[spans[2][0]:spans[2][1]].strip()

        self.assertIn("book_flight", span1)
        self.assertNotIn("tool_result", span1)

        self.assertIn("book_hotel", span2)
        self.assertNotIn("tool_result", span2)

        self.assertIn("Flight and hotel are booked!", span3)

    def test_tag_based_call_format_target_masking(self) -> None:
        text = (
            "User: Calculate sqrt(144)\n"
            "Assistant: <CALL>math.sqrt(144)</CALL>\n"
            "<tool_result>12</tool_result>\n"
            "Assistant: The answer is 12.\n"
            "<eos>"
        )

        spans = find_completion_spans(text)
        self.assertEqual(len(spans), 2)

        span1 = text[spans[0][0]:spans[0][1]].strip()
        span2 = text[spans[1][0]:spans[1][1]].strip()

        self.assertEqual(span1, "<CALL>math.sqrt(144)</CALL>")
        self.assertEqual(span2, "The answer is 12.\n<eos>")

    def test_stop_sequences_contain_tool_tags(self) -> None:
        self.assertIn("</tool_calls>", STOP_SEQUENCES)
        self.assertIn("</CALL>", STOP_SEQUENCES)

    def test_microgpt_chat_preserves_closing_tool_tag(self) -> None:
        reply = 'Some thoughts <tool_calls>[{"name": "lookup"}]</tool_calls> extra prompt text'
        s = "</tool_calls>"
        idx = reply.find(s)
        cleaned = reply[: idx + len(s)].strip()
        self.assertTrue(cleaned.endswith("</tool_calls>"))
        self.assertNotIn("extra prompt text", cleaned)


if __name__ == "__main__":
    unittest.main()
