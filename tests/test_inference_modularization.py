"""Unit tests for the modularized inference package (`inference.core` and `inference.ui`).

Verifies:
1. Headless core execution without GUI dependencies.
2. Autonomous agent tool router and parsing in `inference.core.agent_executor`.
3. Inference types and dataclasses in `inference.core.types`.
4. Chat and evaluation utilities in `inference.core`.
5. Desktop UI widgets and Markdown rendering in `inference.ui`.
6. 100% backwards-compatible re-exports in `engine.*` and `interface.*`.
"""

from __future__ import annotations

import json
import pytest
from PySide6.QtWidgets import QApplication

# 1. Headless core imports (Must succeed without Qt)
import inference.core as inf_core
from inference.core.types import InferenceMessage, GenerationOptions, StreamChunk
from inference.core.agent_executor import (
    parse_tool_calls,
    execute_python_code,
    execute_web_search,
    execute_agent_tool,
)
from inference.core.evaluation import (
    DEFAULT_BENCHMARK_PROMPTS,
    BenchmarkResult,
    normalize_prompts,
)
from inference.core.microgpt_chat import STOP_SEQUENCES

# 2. Desktop UI imports
from inference.ui import (
    ChatMessageWidget,
    ChatInputEdit,
    markdown_to_html,
    format_agent_artifacts_markdown,
    build_chat_tab,
    build_benchmark_tab,
)

# 3. Backwards-compatibility shims
import engine.agent_executor as shim_agent_exec
import engine.microgpt_chat as shim_microgpt_chat
import engine.llama_chat as shim_llama_chat
import engine.generation as shim_generation
import engine.evaluation as shim_evaluation
import interface.chat_widgets as shim_chat_widgets
import interface.markdown_renderer as shim_markdown_renderer


@pytest.fixture(scope="module")
def qapp():
    """Ensure a single QApplication instance for Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInferenceModularization:
    """Test suite for decoupled inference core and desktop UI."""

    def test_inference_core_exports(self) -> None:
        """Verify inference.core exposes all required classes and functions."""
        assert hasattr(inf_core, "MicroGPTChatSession")
        assert hasattr(inf_core, "LlamaChatSession")
        assert hasattr(inf_core, "execute_agent_tool")
        assert hasattr(inf_core, "parse_tool_calls")
        assert hasattr(inf_core, "evaluate_checkpoint")
        assert hasattr(inf_core, "generate_text")
        assert hasattr(inf_core, "load_model_from_checkpoint")

    def test_inference_types_dataclasses(self) -> None:
        """Verify pure Python inference dataclasses."""
        msg = InferenceMessage(role="assistant", content="Hello world", thought="Thinking...")
        assert msg.role == "assistant"
        assert msg.content == "Hello world"
        assert msg.thought == "Thinking..."
        assert msg.tool_calls == []

        opts = GenerationOptions(temperature=0.8, max_new_tokens=128, enable_tools=True)
        assert opts.temperature == 0.8
        assert opts.max_new_tokens == 128
        assert opts.enable_tools is True
        assert opts.max_tool_hops == 3

        chunk = StreamChunk(content="Hi", token_count=1, elapsed_seconds=0.05, tokens_per_second=20.0)
        assert chunk.content == "Hi"
        assert chunk.tokens_per_second == 20.0

    def test_agent_executor_in_core(self) -> None:
        """Verify agent executor tools operate correctly in inference.core."""
        # Python interpreter arithmetic
        result = execute_python_code("print(17 * 23)")
        assert result.strip() == "391"

        # Web search simulation/fallback
        web_res = json.loads(execute_web_search("Python programming"))
        assert "query" in web_res
        assert "results" in web_res

        # Universal tool router
        py_tool = execute_agent_tool("python_interpreter", {"code": "print(2 ** 8)"})
        assert py_tool.strip() == "256"

        # Parser
        tool_json = '<tool_calls>[{"name": "web_search", "arguments": {"query": "deep learning"}}]</tool_calls>'
        parsed = parse_tool_calls(tool_json)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "web_search"
        assert parsed[0]["arguments"] == {"query": "deep learning"}

    def test_evaluation_normalize_prompts(self) -> None:
        """Verify prompt normalization in inference.core.evaluation."""
        raw = "Prompt 1\n\nPrompt 2\n\n\nPrompt 3"
        prompts = normalize_prompts(raw)
        assert prompts == ["Prompt 1", "Prompt 2", "Prompt 3"]

        empty_prompts = normalize_prompts("")
        assert empty_prompts == DEFAULT_BENCHMARK_PROMPTS

    def test_desktop_ui_chat_widgets(self, qapp) -> None:
        """Verify ChatMessageWidget in inference.ui correctly separates reasoning."""
        calls = []
        raw_text = "<thought>Calculating 5! step by step</thought>The answer is 120."
        widget = ChatMessageWidget(
            role="assistant",
            content=raw_text,
            html_renderer=markdown_to_html,
            resend_callback=lambda p: calls.append(p),
        )
        widget.show()

        # Thought toggle must be visible
        assert not widget.thought_toggle.isHidden()
        assert "Thought Process" in widget.thought_toggle.text()

        # Browser displays final reply, not the thought
        assert "120" in widget.browser.toPlainText()
        assert "<thought>" not in widget.browser.toPlainText()

        # Thought browser contains thought text
        assert "Calculating 5!" in widget.thought_browser.toPlainText()

        # Toggle collapse/expand
        initial_visible = not widget.thought_browser.isHidden()
        widget._toggle_thought()
        assert (not widget.thought_browser.isHidden()) != initial_visible

    def test_desktop_ui_markdown_renderer(self) -> None:
        """Verify Markdown-to-HTML rendering with agent styling in inference.ui."""
        snippet = "⚙️ *Calling tool `python_interpreter`...*\n<tool_result id=\"call_1\">Result: 42</tool_result>"
        html = markdown_to_html(snippet)
        assert "tool-call-box" in html
        assert "tool-obs-box" in html
        assert "python_interpreter" in html
        assert "Result: 42" in html

    def test_backwards_compatibility_shims(self) -> None:
        """Verify existing modules in engine and interface re-export correctly."""
        # engine shims
        assert shim_agent_exec.execute_agent_tool is execute_agent_tool
        assert shim_agent_exec.parse_tool_calls is parse_tool_calls
        assert shim_microgpt_chat.STOP_SEQUENCES == STOP_SEQUENCES
        assert hasattr(shim_microgpt_chat, "MicroGPTChatSession")
        assert hasattr(shim_llama_chat, "LlamaChatSession")
        assert hasattr(shim_generation, "load_model_from_checkpoint")
        assert hasattr(shim_evaluation, "evaluate_checkpoint")

        # interface shims
        assert shim_chat_widgets.ChatMessageWidget is ChatMessageWidget
        assert shim_chat_widgets.ChatInputEdit is ChatInputEdit
        assert shim_markdown_renderer.markdown_to_html is markdown_to_html
        assert shim_markdown_renderer.format_agent_artifacts_markdown is format_agent_artifacts_markdown
