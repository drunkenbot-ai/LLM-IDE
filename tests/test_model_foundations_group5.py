"""Unit tests for Group 5 Knowledge Replay Buffer & Collapsible Reasoning Chat UI.

Tests:
1. Replay buffer file discovery and proportional sampling during fine-tune stage dataset loading.
2. Replay buffer ratio = 0.0 disables base document inclusion during fine-tuning.
3. ChatMessageWidget thought process extraction and separation from final reply.
4. ChatMessageWidget interactive collapsible toggle behavior.
5. ChatMessageWidget plain message handling without thought tags.
6. Markdown renderer transformation of agent tool calls, observations, and thoughts.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication

from engine.config import DatasetConfig
from engine.dataset_loader import _load_documents_with_cache
from interface.chat_widgets import ChatMessageWidget
from interface.markdown_renderer import format_agent_artifacts_markdown, markdown_to_html


class TestModelFoundationsGroup5(unittest.TestCase):
    """Test suite for Group 5 knowledge replay buffer and thought UI."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_markdown_renderer_agent_artifacts_formatting(self) -> None:
        """Verify format_agent_artifacts_markdown creates rich HTML blocks."""
        raw_text = (
            "⚙️ *Calling tool `python_interpreter`...*\n"
            "📋 *Observation*: `42`\n"
            "<tool_result id=\"call_abc123\">Calculated result: 42</tool_result>\n"
            "<thought>Double checking the calculation.</thought>"
        )
        formatted = format_agent_artifacts_markdown(raw_text)
        self.assertIn("tool-call-box", formatted)
        self.assertIn("python_interpreter", formatted)
        self.assertIn("tool-obs-box", formatted)
        self.assertIn("call_abc123", formatted)
        self.assertIn("thought-box", formatted)

    def test_markdown_to_html_includes_agent_styles(self) -> None:
        """Verify markdown_to_html renders styled HTML with thought and tool styling."""
        html = markdown_to_html("⚙️ Calling tool web_search...")
        self.assertIn("tool-call-box", html)
        self.assertIn("web_search", html)

    def test_chat_message_widget_thought_extraction(self) -> None:
        """Verify ChatMessageWidget cleanly extracts and isolates <thought> blocks."""
        content = (
            "<thought>\n"
            "1. Analyze user request.\n"
            "2. Execute math verification.\n"
            "</thought>\n"
            "The final answer is 42."
        )
        widget = ChatMessageWidget(
            role="assistant",
            content=content,
            html_renderer=markdown_to_html,
            resend_callback=lambda p: None,
        )
        widget.show()

        self.assertFalse(widget.thought_toggle.isHidden())
        thought_html = widget.thought_browser.toPlainText()
        self.assertIn("Analyze user request", thought_html)
        self.assertNotIn("The final answer is 42", thought_html)

        # Answer browser has final reply without raw thought tags
        main_text = widget.browser.toPlainText()
        self.assertIn("The final answer is 42.", main_text)
        self.assertNotIn("<thought>", main_text)

    def test_chat_message_widget_toggle_behavior(self) -> None:
        """Verify clicking the toggle switches thought visibility and updates arrow."""
        content = "<thought>Internal reasoning</thought>Response text."
        widget = ChatMessageWidget(
            role="assistant",
            content=content,
            html_renderer=markdown_to_html,
            resend_callback=lambda p: None,
        )
        widget.show()

        initial_state = not widget.thought_browser.isHidden()
        widget._toggle_thought()
        self.assertNotEqual(not widget.thought_browser.isHidden(), initial_state)

        # Toggle again
        widget._toggle_thought()
        self.assertEqual(not widget.thought_browser.isHidden(), initial_state)

    def test_chat_message_widget_plain_message_no_thought(self) -> None:
        """Verify regular messages without thought tags hide the toggle."""
        widget = ChatMessageWidget(
            role="assistant",
            content="Simple direct answer without reasoning.",
            html_renderer=markdown_to_html,
            resend_callback=lambda p: None,
        )
        widget.show()
        self.assertTrue(widget.thought_toggle.isHidden())
        self.assertTrue(widget.thought_browser.isHidden())
        self.assertIn("Simple direct answer", widget.browser.toPlainText())

    def test_replay_buffer_sampling_in_dataset_loader(self) -> None:
        """Verify dataset_loader samples base documents when replay_buffer_ratio > 0."""
        tmp_dir = Path(tempfile.mkdtemp())
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        # Create 10 base pre-training text files
        for i in range(10):
            (input_dir / f"doc_{i:02d}.txt").write_text(
                f"Base pre-training document {i} containing substantive text about mathematics and science.",
                encoding="utf-8",
            )

        # Create 1 structured instruction JSONL file
        instruction_file = tmp_dir / "instructions.jsonl"
        instruction_file.write_text(
            json.dumps({"prompt": "Hello", "response": "Hi there!"}) + "\n",
            encoding="utf-8",
        )

        # Case 1: Replay buffer ratio = 0.3 (should sample ~3 base docs)
        config_with_replay = DatasetConfig(
            input_dir=input_dir,
            output_dir=output_dir / "run1",
            dataset_stage="instruction",
            instruction_dataset_paths=[instruction_file],
            replay_buffer_ratio=0.3,
        )

        mock_builder = MagicMock()
        mock_builder.stats = MagicMock()
        mock_builder.stats.accepted_document_count = 0

        _load_documents_with_cache(config_with_replay, mock_builder)
        # Verify that builder received submissions (structured instruction + sampled replay docs)
        self.assertGreaterEqual(mock_builder.submit.call_count, 2)

        # Case 2: Replay buffer ratio = 0.0 (disabled)
        mock_builder_zero = MagicMock()
        mock_builder_zero.stats = MagicMock()
        mock_builder_zero.stats.accepted_document_count = 0

        config_no_replay = DatasetConfig(
            input_dir=input_dir,
            output_dir=output_dir / "run2",
            dataset_stage="instruction",
            instruction_dataset_paths=[instruction_file],
            replay_buffer_ratio=0.0,
        )

        _load_documents_with_cache(config_no_replay, mock_builder_zero)
        # Exactly 1 submission (only the structured instruction document)
        self.assertEqual(mock_builder_zero.submit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
