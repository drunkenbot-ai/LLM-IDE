"""Unit tests for Cybersecurity & Cryptographic Engineering Pretraining Generator."""

import json
import random
import tempfile
from pathlib import Path
import pytest

from tools.generate_base_cybersecurity_pretraining import (
    GENERATORS,
    ShardedCybersecurityWriter,
    generate_batch,
    write_partitioned_cyber_dataset,
)
from tools.audit_dataset_quality import audit_directory


def test_cybersecurity_generators_validity():
    """Verify that all cybersecurity generators produce authentic binary, crypto, or kernel analysis."""
    rng = random.Random(42)
    sec_keywords = {"buffer", "exploit", "canary", "rop", "crypto", "constant-time", "ebpf", "kernel", "syscall", "memory", "mitigation"}

    for gen_fn, subspecialty in GENERATORS:
        text = gen_fn(rng)
        assert isinstance(text, str)
        assert len(text) > 400, f"Generator {gen_fn.__name__} output too short"
        text_lower = text.lower()
        assert any(kw in text_lower for kw in sec_keywords), f"Missing cybersecurity keywords in {gen_fn.__name__}"


def test_sharded_cybersecurity_writer_ceiling():
    """Ensure ShardedCybersecurityWriter properly shards and respects ceiling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        small_ceiling = 20 * 1024  # 20 KB ceiling
        writer = ShardedCybersecurityWriter(output_dir=out_dir, prefix="test_sec", max_bytes=small_ceiling)

        sample_doc = "# Vulnerability Analysis\n" + ("Buffer overflow analysis in memory frame.\n" * 300)
        for _ in range(6):
            writer.write_record(text=sample_doc, subspecialty="test_sec")
        writer.close()

        written = list(out_dir.glob("test_sec_*.jsonl"))
        assert len(written) >= 2, f"Expected multiple partitions, got {len(written)}"
        for p in written:
            assert p.stat().st_size <= small_ceiling + 1024
            with open(p, "r", encoding="utf-8") as fp:
                for line in fp:
                    data = json.loads(line)
                    assert "text" in data
                    assert "meta" in data
                    assert data["meta"]["domain"] == "cybersecurity"


def test_write_partitioned_cyber_dataset_audit():
    """Verify cybersecurity dataset generation integration and quality audit pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        written = write_partitioned_cyber_dataset(
            output_dir=out_dir,
            count=30,
            max_file_mb=0.04,
            seed=99,
        )

        assert len(written) >= 2
        assert (out_dir / "manifest.json").exists()
        assert audit_directory(out_dir, kind="generic") is True
