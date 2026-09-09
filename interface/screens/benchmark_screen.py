"""Benchmark screen mixin for MainWindow.

Note: Desktop UI implementation has moved to `inference.ui.benchmark_screen`.
This module re-exports BenchmarkScreenMixin for backwards compatibility.
"""

from __future__ import annotations

from inference.ui.benchmark_screen import BenchmarkScreenMixin

__all__ = ["BenchmarkScreenMixin"]
