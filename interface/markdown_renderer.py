"""Markdown to HTML renderer with tool call, observation, and thought formatting.

Note: Desktop UI implementation has moved to `inference.ui.markdown_renderer`.
This module re-exports all symbols for backwards compatibility.
"""

from __future__ import annotations

from inference.ui.markdown_renderer import (
    _html_theme_colors,
    basic_markdown_html,
    code_block_html,
    colorize_code,
    escape_html,
    format_agent_artifacts_markdown,
    guess_code_language,
    inline_basic_markdown,
    markdown_to_html,
    normalize_code_blocks,
    render_basic_prose,
)

__all__ = [
    "format_agent_artifacts_markdown",
    "markdown_to_html",
    "basic_markdown_html",
    "code_block_html",
    "render_basic_prose",
    "inline_basic_markdown",
    "escape_html",
    "colorize_code",
    "normalize_code_blocks",
    "guess_code_language",
    "_html_theme_colors",
]
