"""Unit tests for Multilingual & Cross-Lingual Alignment Pretraining Generator."""

import json
import random
import tempfile
from pathlib import Path
import pytest

from tools.generate_base_multilingual_pretraining import (
    GENERATORS,
    ShardedMultilingualWriter,
    generate_batch,
    write_partitioned_multilingual_dataset,
)
from tools.audit_dataset_quality import audit_directory


def test_multilingual_generators_validity():
    """Verify that all multilingual generators produce authentic text and parallel cross-lingual sections."""
    rng = random.Random(42)

    for gen_fn, subspecialty, lang in GENERATORS:
        text = gen_fn(rng)
        assert isinstance(text, str)
        assert len(text) > 300, f"Generator {gen_fn.__name__} output too short"
        assert "Cross-Lingual Alignment" in text or "English Parallel" in text
        assert "##" in text


def test_sharded_multilingual_writer_ceiling():
    """Ensure ShardedMultilingualWriter properly shards and respects ceiling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        small_ceiling = 20 * 1024  # 20 KB ceiling
        writer = ShardedMultilingualWriter(output_dir=out_dir, prefix="test_multi", max_bytes=small_ceiling)

        sample_doc = "# Multilingual Sample\n" + ("Das ist ein wichtiger wissenschaftlicher Text.\n" * 300)
        for _ in range(6):
            writer.write_record(text=sample_doc, language="de", subspecialty="test_de")
        writer.close()

        written = list(out_dir.glob("test_multi_*.jsonl"))
        assert len(written) >= 2, f"Expected multiple partitions, got {len(written)}"
        for p in written:
            assert p.stat().st_size <= small_ceiling + 1024
            with open(p, "r", encoding="utf-8") as fp:
                for line in fp:
                    data = json.loads(line)
                    assert "text" in data
                    assert "meta" in data
                    assert data["meta"]["domain"] == "multilingual"


def test_write_partitioned_multilingual_dataset_audit():
    """Verify multilingual dataset generation integration and quality audit pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        written = write_partitioned_multilingual_dataset(
            output_dir=out_dir,
            count=30,
            max_file_mb=0.04,
            seed=99,
        )

        assert len(written) >= 2
        assert (out_dir / "manifest.json").exists()
        assert audit_directory(out_dir, kind="generic") is True
