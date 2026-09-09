"""Chat screen mixin for MainWindow.

Note: Desktop UI implementation has moved to `inference.ui.chat_screen`.
This module re-exports ChatScreenMixin for backwards compatibility.
"""

from __future__ import annotations

from inference.ui.chat_screen import ChatScreenMixin

__all__ = ["ChatScreenMixin"]
