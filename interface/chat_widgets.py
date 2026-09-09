"""Chat message widgets and inputs.

Note: Desktop UI implementation has moved to `inference.ui.chat_widgets`.
This module re-exports all symbols for backwards compatibility.
"""

from __future__ import annotations

from inference.ui.chat_widgets import (
    ChatInputEdit,
    ChatMessageWidget,
)

__all__ = [
    "ChatInputEdit",
    "ChatMessageWidget",
]
