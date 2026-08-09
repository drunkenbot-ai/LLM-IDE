"""Tests for the Tool-call fine-tune workflow controls."""

from __future__ import annotations

from interface.main_window_part13 import MainWindowPart13


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

    window = MainWindowPart13()
    window.training_mode = _Combo("Tool-call fine-tune")

    assert window._training_mode_value() == "fine_tune"
    assert window._training_stage_value() == "tool_call"
