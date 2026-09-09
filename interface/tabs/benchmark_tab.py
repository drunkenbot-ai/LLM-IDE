"""Benchmark tab builder.

Note: Desktop UI implementation has moved to `inference.ui.benchmark_tab`.
This module re-exports build_benchmark_tab for backwards compatibility.
"""

from __future__ import annotations

from inference.ui.benchmark_tab import build_benchmark_tab

__all__ = ["build_benchmark_tab"]
