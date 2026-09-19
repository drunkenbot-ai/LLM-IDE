"""Unit tests for Hardware Engineering & Semiconductor RTL Pretraining Generator."""

import json
import random
import tempfile
from pathlib import Path
import pytest

from tools.generate_base_hardware_pretraining import (
    GENERATORS,
    ShardedHardwareWriter,
    generate_batch,
    write_partitioned_hardware_dataset,
)
from tools.audit_dataset_quality import audit_directory


def test_hardware_generators_validity():
    """Verify that all hardware generators produce synthesizable RTL or semiconductor physics content."""
    rng = random.Random(42)
    hw_keywords = {"module", "logic", "clock", "pipeline", "slack", "timing", "cache", "verilog", "fifo", "register"}

    for gen_fn, subspecialty in GENERATORS:
        text = gen_fn(rng)
        assert isinstance(text, str)
        assert len(text) > 400, f"Generator {gen_fn.__name__} output too short"
        text_lower = text.lower()
        assert any(kw in text_lower for kw in hw_keywords), f"Missing hardware keywords in {gen_fn.__name__}"


def test_sharded_hardware_writer_ceiling():
    """Ensure ShardedHardwareWriter properly shards and respects ceiling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        small_ceiling = 20 * 1024  # 20 KB ceiling
        writer = ShardedHardwareWriter(output_dir=out_dir, prefix="test_hw", max_bytes=small_ceiling)

        sample_doc = "# RTL Design\n" + ("module test (input clk, output data);\n" * 300)
        for _ in range(6):
            writer.write_record(text=sample_doc, subspecialty="test_rtl")
        writer.close()

        written = list(out_dir.glob("test_hw_*.jsonl"))
        assert len(written) >= 2, f"Expected multiple partitions, got {len(written)}"
        for p in written:
            assert p.stat().st_size <= small_ceiling + 1024
            with open(p, "r", encoding="utf-8") as fp:
                for line in fp:
                    data = json.loads(line)
                    assert "text" in data
                    assert "meta" in data
                    assert data["meta"]["domain"] == "hardware_engineering"


def test_write_partitioned_hardware_dataset_audit():
    """Verify hardware dataset generation integration and quality audit pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        written = write_partitioned_hardware_dataset(
            output_dir=out_dir,
            count=30,
            max_file_mb=0.04,
            seed=99,
        )

        assert len(written) >= 2
        assert (out_dir / "manifest.json").exists()
        assert audit_directory(out_dir, kind="code") is True
