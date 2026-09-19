"""Tests for Dataset Enrichment Tools (fetch_open_finance and curate_pretraining_mixture)."""

import json
import tempfile
from pathlib import Path
import pytest

from tools.fetch_open_finance import generate_batch, write_partitioned_dataset
from tools.curate_pretraining_mixture import ShardedWriter, curate_mixture
from tools.audit_dataset_quality import audit_directory, audit_file


def test_fetch_open_finance_record_generation():
    """Verify generated financial records have valid schema and keywords."""
    records = list(generate_batch(count=10, seed=42))
    assert len(records) == 10
    for r in records:
        assert "messages" in r
        assert len(r["messages"]) == 2
        assert r["messages"][0]["role"] == "user"
        assert r["messages"][1]["role"] == "assistant"
        asst_content = r["messages"][1]["content"]
        assert any(kw in asst_content.lower() for kw in ["$", "margin", "ebit", "revenue", "cash", "debt", "equity", "growth", "interest", "valuation"])


def test_fetch_open_finance_partitioning_and_audit():
    """Verify finance partition writing respects the size gate and passes quality audit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir)
        written = write_partitioned_dataset(out_path, target_count=50, max_file_mb=0.05, seed=123)
        assert len(written) >= 2  # Partitioned because max_file_mb is small
        for p in written:
            assert p.exists()
            assert p.stat().st_size <= 0.06 * 1024 * 1024
        
        # Verify audit passes
        passed = audit_directory(out_path, kind="finance")
        assert passed is True


def test_curate_pretraining_mixture_streaming_and_manifest():
    """Verify curate_pretraining_mixture shards correctly and produces manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_dir = tmp_path / "src_finance"
        write_partitioned_dataset(src_dir, target_count=30, max_file_mb=28.0, seed=99)

        curated_out = tmp_path / "curated"
        manifest = curate_mixture(
            source_dirs={"finance": src_dir},
            output_dir=curated_out,
            target_tokens=5000,
            ratios={"finance": 1.0},
            max_partition_mb=28.0,
        )

        assert manifest["partitions_count"] >= 1
        assert (curated_out / "curation_manifest.json").exists()
        assert audit_directory(curated_out, kind="generic") is True
