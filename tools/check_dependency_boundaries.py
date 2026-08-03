"""Check the dependency direction between the engine and desktop interface."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundaryViolation:
    """A forbidden import found in a package."""

    path: Path
    line: int
    imported: str


def _python_files(package_root: Path) -> list[Path]:
    """Return Python files below a package in deterministic order."""

    return sorted(
        path
        for path in package_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _imports(path: Path) -> list[tuple[int, str]]:
    """Return absolute top-level imports from a Python file."""

    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.lineno, node.module))
    return imports


def find_violations(root: Path) -> list[BoundaryViolation]:
    """Find imports that violate the engine/interface dependency boundary."""

    rules = (("engine", "interface"),)
    violations: list[BoundaryViolation] = []
    for package_name, forbidden_root in rules:
        package_root = root / package_name
        if not package_root.is_dir():
            continue
        for path in _python_files(package_root):
            for line, imported in _imports(path):
                if imported == forbidden_root or imported.startswith(f"{forbidden_root}."):
                    violations.append(
                        BoundaryViolation(path.relative_to(root), line, imported)
                    )
    return violations


def build_parser() -> argparse.ArgumentParser:
    """Build the boundary-check command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan.",
    )
    return parser


def main() -> int:
    """Run the boundary check and return a process status."""

    root = build_parser().parse_args().root.resolve()
    violations = find_violations(root)
    if violations:
        print("Forbidden package-boundary imports found:")
        for violation in violations:
            print(f"  {violation.path}:{violation.line}: {violation.imported}")
        return 1
    print(
        "Dependency boundaries are clean: engine cannot import interface."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
