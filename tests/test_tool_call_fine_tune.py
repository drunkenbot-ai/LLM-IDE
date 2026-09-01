"""Tests for the Tool-call fine-tune workflow controls."""

from __future__ import annotations

import json
from pathlib import Path

from interface.screens.fine_tuning_screen import FineTuningScreenMixin
from engine.data_core import load_structured_json_documents


class _Combo:
    """Minimal combo-box test double."""

    def __init__(self, text: str) -> None:
        """Store the selected text.

        Args:
            text: Selected combo-box label.
        """

        self._text = text

    def currentText(self) -> str:
        """Return the selected label.

        Returns:
            Selected label.
        """

        return self._text


def test_tool_call_fine_tune_maps_to_fine_tune_stage() -> None:
    """Map the UI label to the engine fine-tuning and dataset stages."""

    window = FineTuningScreenMixin()
    window.training_mode = _Combo("Tool-call fine-tune")

    assert window._training_mode_value() == "fine_tune"
    assert window._training_stage_value() == "tool_call"


def test_tool_call_jsonl_preserves_tools_calls_and_results(tmp_path: Path) -> None:
    source = tmp_path / "tools.jsonl"
    source.write_text(json.dumps({
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "messages": [
            {"role": "user", "content": "Find it"},
            {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "{\"value\": 1}"},
        ],
    }) + "\n", encoding="utf-8")

    documents = load_structured_json_documents(source, "tool_call")

    assert len(documents) == 1
    assert "Tools:" in documents[0].text
    assert "<tool_calls>" in documents[0].text
    assert '<tool_result id="call-1">' in documents[0].text


def test_structured_jsonl_skips_malformed_and_invalid_records(tmp_path: Path) -> None:
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        '{"instruction":"keep","output":"yes"}\n'
        '{"unrelated":"missing training fields"}\n'
        '{"messages":[{"role":"user","content":"valid"}]}\n'
        '{not json}\n',
        encoding="utf-8",
    )
    messages: list[str] = []

    documents = load_structured_json_documents(
        source, "instruction", on_invalid=messages.append
    )

    assert len(documents) == 1
    assert "keep" in documents[0].text
    assert any("line 2" in message for message in messages)
    assert any("line 4" in message and "invalid JSON" in message for message in messages)
