"""Unit tests for Group 4 Autonomous Agent Tool Execution & Model Capabilities.

Tests:
1. Tool call parsing (parse_tool_calls) for JSON and tag-based formats.
2. Resilience of tool call parsing on malformed or absent markup.
3. Python interpreter execution sandbox (execute_python_code) with timeouts and errors.
4. Web search execution (execute_web_search) with query validation and fallback.
5. Tool router (execute_agent_tool) for python_interpreter and web_search.
6. DatasetConfig replay_buffer_ratio default and validation constraints.
7. Synthetic agent and identity corpus generator outputs.
8. MicroGPTChatSession tool prompt rendering with <tools> block.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from engine.agent_executor import (
    execute_agent_tool,
    execute_python_code,
    execute_web_search,
    parse_tool_calls,
)
from engine.config import DatasetConfig, ModelConfig
from engine.generate_agent_data import generate_agent_dataset
from engine.generate_identity_data import generate_identity_corpus
from engine.microgpt_chat import MicroGPTChatSession
from engine.model_gpt import MicroGPT


class TestModelFoundationsGroup4(unittest.TestCase):
    """Test suite for Group 4 autonomous agent execution and capabilities."""

    def test_parse_tool_calls_json_block(self) -> None:
        """Verify extraction of tool calls from JSON <tool_calls> blocks."""
        text = (
            "I need to search for this information.\n"
            "<tool_calls>[{\"name\": \"web_search\", \"arguments\": {\"query\": \"quantum computing 2026\"}}]</tool_calls>"
        )
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "web_search")
        self.assertEqual(calls[0]["arguments"], {"query": "quantum computing 2026"})
        self.assertTrue("id" in calls[0])

    def test_parse_tool_calls_json_string_arguments(self) -> None:
        """Verify extraction when arguments are a serialized JSON string."""
        text = (
            "<tool_calls>[\n"
            "  {\"name\": \"python_interpreter\", \"arguments\": \"{\\\"code\\\": \\\"print(42)\\\"}\"}\n"
            "]</tool_calls>"
        )
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "python_interpreter")
        self.assertEqual(calls[0]["arguments"], {"code": "print(42)"})

    def test_parse_tool_calls_tag_block(self) -> None:
        """Verify extraction of tool calls from <CALL> tags."""
        text = (
            "Let's execute code:\n"
            "<CALL>tool=python_interpreter\n"
            "code=x = 10 * 5\nprint(x)\n"
            "</CALL>"
        )
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "python_interpreter")
        self.assertIn("10 * 5", str(calls[0]["arguments"]))

    def test_parse_tool_calls_resilience(self) -> None:
        """Verify graceful return of empty list for text without tools or invalid JSON."""
        # Plain text
        self.assertEqual(parse_tool_calls("Hello, how can I help you today?"), [])
        # Malformed JSON
        self.assertEqual(parse_tool_calls("<tool_calls>[{invalid json</tool_calls>"), [])

    def test_python_interpreter_execution_success(self) -> None:
        """Verify safe execution of standard Python computation snippets."""
        output = execute_python_code("print(2 ** 10)")
        self.assertEqual(output.strip(), "1024")

        # Multi-line with math library
        output2 = execute_python_code("import math\nprint(round(math.sqrt(144)))")
        self.assertEqual(output2.strip(), "12")

    def test_python_interpreter_execution_error_handling(self) -> None:
        """Verify that Python execution errors are caught cleanly without raising exceptions."""
        output = execute_python_code("1 / 0")
        self.assertIn("ZeroDivisionError", output)

    def test_python_interpreter_timeout(self) -> None:
        """Verify that infinite loops in Python code hit the timeout cleanly."""
        output = execute_python_code("while True: pass", timeout_seconds=0.5)
        self.assertIn("timed out", output.lower())

    def test_web_search_empty_query(self) -> None:
        """Verify web search handles empty query with error JSON."""
        res_str = execute_web_search("")
        data = json.loads(res_str)
        self.assertIn("error", data)

    def test_web_search_valid_query(self) -> None:
        """Verify web search returns valid JSON structure with query and results."""
        res_str = execute_web_search("Python programming language", timeout_seconds=3.0)
        data = json.loads(res_str)
        self.assertEqual(data["query"], "Python programming language")
        self.assertIn("results", data)
        self.assertIsInstance(data["results"], list)

    def test_execute_agent_tool_router(self) -> None:
        """Verify execute_agent_tool correctly routes to interpreter and search."""
        # Route python_interpreter
        res_py = execute_agent_tool("python_interpreter", {"code": "print('hello from tool')"})
        self.assertIn("hello from tool", res_py)

        # Route web_search
        res_search = execute_agent_tool("web_search", {"query": "artificial intelligence"})
        data = json.loads(res_search)
        self.assertIn("results", data)

        # Route unknown tool
        res_unknown = execute_agent_tool("unknown_tool_xyz", {})
        self.assertIn("Unknown tool", res_unknown)

    def test_dataset_config_replay_buffer_ratio(self) -> None:
        """Verify DatasetConfig has replay_buffer_ratio with default 0.05 and validation."""
        config = DatasetConfig(
            input_dir=Path(tempfile.mkdtemp()),
            output_dir=Path(tempfile.mkdtemp()),
        )
        self.assertEqual(config.replay_buffer_ratio, 0.05)
        config.validate()

        # Valid custom ratio
        config.replay_buffer_ratio = 0.15
        config.validate()

        # Invalid negative ratio
        config.replay_buffer_ratio = -0.1
        with self.assertRaises(ValueError):
            config.validate()

        # Invalid ratio > 1.0
        config.replay_buffer_ratio = 1.5
        with self.assertRaises(ValueError):
            config.validate()

    def test_generate_identity_corpus_callable(self) -> None:
        """Verify generate_identity_corpus generates valid identity facts file."""
        tmp_dir = Path(tempfile.mkdtemp())
        project_json = tmp_dir / "project.json"
        project_json.write_text(
            json.dumps({
                "project_name": "TestAgentModel",
                "created_at": "2026-09-01T12:00:00",
            }),
            encoding="utf-8",
        )

        out_path = tmp_dir / "training_data" / "identity" / "identity_facts.txt"
        result_path = generate_identity_corpus(
            project_dir=tmp_dir,
            output_path=out_path,
            creator="DrunkenBot AI",
            maker="Nilesh Jadhav",
            role="Autonomous Research Agent",
            sentence_count=50,
        )

        self.assertTrue(result_path.exists())
        content = result_path.read_text(encoding="utf-8")
        self.assertIn("TestAgentModel", content)
        self.assertIn("DrunkenBot AI", content)
        self.assertGreater(len(content), 200)

    def test_microgpt_chat_tools_system_prompt_declaration(self) -> None:
        """Verify MicroGPTChatSession injects tools schema when enable_tools=True."""
        model_config = ModelConfig(
            vocab_size=1000,
            embedding_size=64,
            head_count=2,
            layer_count=1,
            context_length=512,
        )
        model = MicroGPT(model_config)

        # Mock tokenizer with encode/decode
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 0
        tokenizer.pad_token_id = 1
        tokenizer.encode.return_value = MagicMock(ids=[1, 2, 3])
        tokenizer.decode.return_value = "Hello"

        session = MicroGPTChatSession.__new__(MicroGPTChatSession)
        session.model = model
        session.tokenizer = tokenizer
        session.device = "cpu"
        session.config = model_config
        session._messages = []

        # Prompt with enable_tools=True
        prompt_with_tools = session._render_prompt(
            prompt="What is 2+2?",
            system_prompt="",
            reasoning_effort="medium",
            thinking_enabled=False,
            max_tokens=64,
            enable_tools=True,
        )
        self.assertIn("<tools>", prompt_with_tools)
        self.assertIn("python_interpreter", prompt_with_tools)
        self.assertIn("web_search", prompt_with_tools)

        # Prompt with enable_tools=False
        prompt_without_tools = session._render_prompt(
            prompt="What is 2+2?",
            system_prompt="",
            reasoning_effort="medium",
            thinking_enabled=False,
            enable_tools=False,
        )
        self.assertNotIn("<tools>", prompt_without_tools)


if __name__ == "__main__":
    unittest.main()
