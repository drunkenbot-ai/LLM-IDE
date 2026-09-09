"""Chat tab builder.

Note: Desktop UI implementation has moved to `inference.ui.chat_tab`.
This module re-exports build_chat_tab for backwards compatibility.
"""

from __future__ import annotations

from inference.ui.chat_tab import build_chat_tab

__all__ = ["build_chat_tab"]
