"""Check repository-wide Python source constraints.

This dependency-free check is intentionally small so it can run before the
optional formatting, linting, and type-checking tools are installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


DEFAULT_IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "packaging",
}


def iter_python_files(root: Path) -> list[Path]:
    """Return Python source files below a repository root.

    Args:
        root: Repository directory to scan.

    Returns:
        Python files in deterministic path order.
    """
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in DEFAULT_IGNORED_DIRECTORIES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def count_lines(path: Path) -> int:
    """Count lines in a UTF-8 Python source file.

    Args:
        path: Python file to read.

    Returns:
        Number of lines in the file.
    """
    return len(path.read_text(encoding="utf-8-sig").splitlines())


def find_oversized_files(root: Path, maximum_lines: int) -> list[tuple[Path, int]]:
    """Find Python files exceeding the configured line limit.

    Args:
        root: Repository directory to scan.
        maximum_lines: Maximum permitted number of lines per file.

    Returns:
        Paths and line counts for files over the limit.
    """
    oversized: list[tuple[Path, int]] = []
    for path in iter_python_files(root):
        line_count = count_lines(path)
        if line_count > maximum_lines:
            oversized.append((path, line_count))
    return oversized


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=500,
        help="Maximum permitted lines in a Python file.",
    )
    return parser


def main() -> int:
    """Run the repository source constraint check.

    Returns:
        Zero when all files satisfy the configured constraints; otherwise one.
    """
    args = build_parser().parse_args()
    if args.max_lines < 1:
        raise ValueError("--max-lines must be greater than zero")

    oversized = find_oversized_files(args.root.resolve(), args.max_lines)
    if not oversized:
        print(f"All Python files are at or below {args.max_lines} lines.")
        return 0

    print(f"Python files exceeding {args.max_lines} lines:")
    for path, line_count in oversized:
        print(f"  {path.relative_to(args.root.resolve())}: {line_count} lines")
    return 1


if __name__ == "__main__":
    sys.exit(main())
