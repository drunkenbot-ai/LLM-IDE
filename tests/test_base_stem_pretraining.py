"""Unit tests for STEM & Formal Mathematics Base Pretraining Generator."""

import json
import os
import random
import tempfile
from pathlib import Path

import pytest
from tools.generate_base_stem_pretraining import (
    STEM_GENERATOR_REGISTRY,
    ShardedSTEMWriter,
    calculate_latex_density,
    estimate_tokens,
    generate_base_stem_corpus,
)


def test_stem_generators_validity():
    """Verify that all STEM generators produce rich LaTeX content and complete proofs."""
    rng = random.Random(2026)
    math_symbols = ["$", "\\frac", "\\int", "\\sum", "\\mathbf", "\\mathbb", "\\lim"]

    for gen_fn in STEM_GENERATOR_REGISTRY:
        text, meta = gen_fn(rng)
        assert isinstance(text, str)
        assert len(text) > 300, f"STEM generator {gen_fn.__name__} output too short"
        assert isinstance(meta, dict)
        assert meta.get("domain") == "stem"
        assert "subdomain" in meta
        assert "topic" in meta
        assert meta.get("tokens", 0) > 50

        # Check LaTeX density and math symbols
        density = calculate_latex_density(text)
        assert density > 0.02, f"LaTeX density {density} too low for {gen_fn.__name__}"
        assert any(sym in text for sym in math_symbols), f"Missing mathematical symbols in {gen_fn.__name__}"


def test_sharded_stem_writer_partition_ceiling():
    """Ensure ShardedSTEMWriter properly rolls over when partition limit is hit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        small_ceiling = 40 * 1024  # 40 KB ceiling
        writer = ShardedSTEMWriter(output_dir=out_dir, prefix="test_stem_shard", max_bytes=small_ceiling)

        sample_doc = "# Theorem\nLet $\\int_0^1 f(x) dx = 0$.\n" + ("x " * 5000)
        for _ in range(8):
            writer.write_document(text=sample_doc, meta={"domain": "stem", "subdomain": "analysis"})
        writer.close()

        written = list(out_dir.glob("test_stem_shard_*.jsonl"))
        assert len(written) >= 2, f"Expected multiple partitions, got {len(written)}"

        for p in written:
            assert p.stat().st_size > 0
            with open(p, "r", encoding="utf-8") as fp:
                for line in fp:
                    data = json.loads(line)
                    assert "text" in data
                    assert "meta" in data


def test_generate_base_stem_corpus_integration():
    """Test full integration generation pipeline for STEM pretraining data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        manifest = generate_base_stem_corpus(
            output_dir=out_dir,
            count=50,
            seed=1337,
            max_mb=28.0,
            prefix="test_stem_base"
        )

        assert manifest["total_records"] == 50
        assert manifest["total_bytes"] > 0
        assert manifest["partitions_count"] >= 1
        assert manifest["average_latex_density"] > 0.05
        assert (out_dir / "manifest.json").exists()

        subdomains = manifest["subdomain_distribution"]
        assert len(subdomains) >= 4, f"Expected diverse subdomains, got {subdomains}"
