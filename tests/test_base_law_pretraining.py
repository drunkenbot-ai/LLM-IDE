"""Unit tests for Law, Governance & Contracts Base Pretraining Generator."""

import json
import random
import tempfile
from pathlib import Path

import pytest
from tools.generate_base_law_pretraining import (
    LAW_GENERATOR_REGISTRY,
    ShardedLawWriter,
    estimate_tokens,
    generate_base_law_corpus,
)


def test_law_generators_validity():
    """Verify that all law generators produce authentic legal text with correct metadata."""
    rng = random.Random(42)
    legal_keywords = ["section", "article", "agreement", "liability", "regulation", "court", "license", "fiduciary"]

    for gen_fn in LAW_GENERATOR_REGISTRY:
        text, meta = gen_fn(rng)
        assert isinstance(text, str)
        assert len(text) > 300, f"Law generator {gen_fn.__name__} output too short"
        assert isinstance(meta, dict)
        assert meta.get("domain") == "law_governance"
        assert "subdomain" in meta
        assert "topic" in meta
        assert meta.get("tokens", 0) > 50
        text_lower = text.lower()
        assert any(kw in text_lower for kw in legal_keywords), f"Missing legal keywords in {gen_fn.__name__}"


def test_sharded_law_writer_partition_ceiling():
    """Ensure ShardedLawWriter properly rolls over when partition limit is hit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        small_ceiling = 30 * 1024  # 30 KB ceiling
        writer = ShardedLawWriter(output_dir=out_dir, prefix="test_law_shard", max_bytes=small_ceiling)

        sample_doc = "# Contract Agreement\n" + ("Party agrees to terms and conditions.\n" * 500)
        for _ in range(8):
            writer.write_document(text=sample_doc, meta={"domain": "law_governance", "subdomain": "contract"})
        writer.close()

        written = list(out_dir.glob("test_law_shard_*.jsonl"))
        assert len(written) >= 2, f"Expected multiple partitions, got {len(written)}"

        for p in written:
            assert p.stat().st_size > 0
            with open(p, "r", encoding="utf-8") as fp:
                for line in fp:
                    data = json.loads(line)
                    assert "text" in data
                    assert "meta" in data


def test_generate_base_law_corpus_integration():
    """Test full integration generation pipeline for Law pretraining data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        manifest = generate_base_law_corpus(
            output_dir=out_dir,
            count=40,
            seed=1337,
            max_mb=28.0,
            prefix="test_law_base"
        )

        assert manifest["total_records"] == 40
        assert manifest["total_bytes"] > 0
        assert manifest["partitions_count"] >= 1
        assert (out_dir / "manifest.json").exists()

        subdomains = manifest["subdomain_distribution"]
        assert len(subdomains) >= 3, f"Expected diverse subdomains, got {subdomains}"
