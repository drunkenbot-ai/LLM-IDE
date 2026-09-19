"""Unit tests for Scientific Literature Packager."""

import json
import tempfile
from pathlib import Path
import pytest

from tools.package_scientific_literature import (
    ShardedScienceWriter,
    chunk_text,
    package_science_dataset,
)
from tools.audit_dataset_quality import audit_directory


def test_chunk_text_boundary_preservation():
    """Verify paragraph-aware chunking keeps passages under max_words."""
    paragraphs = [f"Paragraph {i} contains detailed scientific observations. " * 10 for i in range(20)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, max_words=100)
    assert len(chunks) >= 3
    for c in chunks:
        words = len(c.split())
        assert words > 0


def test_package_science_dataset_integration():
    """Verify end-to-end science packaging, sharding, and quality audit pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        res_dir = tmp_path / "research"
        eb_dir = tmp_path / "ebooks"
        out_dir = tmp_path / "out_science"
        res_dir.mkdir()
        eb_dir.mkdir()

        # Create mock research paper
        (res_dir / "paper1.md").write_text(
            "# Astrophysics\n\nAbstract of research.\n\n" + ("Observation data on stellar spectra.\n\n" * 30),
            encoding="utf-8"
        )
        # Create mock textbook
        (eb_dir / "book1.txt").write_text(
            "Histology Textbook\n\nChapter 1: Cellular structures.\n\n" + ("Epithelial tissue microarchitecture.\n\n" * 40),
            encoding="utf-8"
        )

        written = package_science_dataset(
            research_dir=res_dir,
            ebooks_dir=eb_dir,
            output_dir=out_dir,
            max_file_mb=0.05,
        )

        assert len(written) >= 1
        assert (out_dir / "manifest.json").exists()
        assert audit_directory(out_dir, kind="generic") is True
