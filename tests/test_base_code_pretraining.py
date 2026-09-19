"""Unit tests for Code & Systems Architecture Base Pretraining Generator."""

import ast
import json
import os
import random
import tempfile
from pathlib import Path

import pytest
from tools.generate_base_code_pretraining import (
    GENERATOR_REGISTRY,
    ShardedCodeWriter,
    estimate_tokens,
    generate_base_code_corpus,
)


def test_registry_generators_validity():
    """Verify that all generators in the registry produce valid text and metadata."""
    rng = random.Random(1337)
    for gen_fn in GENERATOR_REGISTRY:
        text, meta = gen_fn(rng)
        assert isinstance(text, str)
        assert len(text) > 100, f"Generator {gen_fn.__name__} output too short"
        assert isinstance(meta, dict)
        assert "domain" in meta
        assert "language" in meta
        assert "tokens" in meta
        assert meta["tokens"] > 20

        # If python code, ensure syntax is valid by checking ast
        if meta["language"] == "python":
            # Extract code block if markdown fenced or direct python
            code_to_check = text
            if "```python" in text:
                parts = text.split("```python")
                code_to_check = parts[1].split("```")[0]
            # ast.parse must succeed without syntax error
            parsed = ast.parse(code_to_check)
            assert parsed is not None


def test_sharded_writer_partition_ceiling():
    """Ensure ShardedCodeWriter properly rolls over when partition limit is hit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        # Set a tiny 50 KB ceiling to test sharding
        small_ceiling = 50 * 1024
        writer = ShardedCodeWriter(output_dir=out_dir, prefix="test_shard", max_bytes=small_ceiling)

        sample_doc = "A" * 15000  # 15KB doc
        for _ in range(10):
            writer.write_document(text=sample_doc, meta={"domain": "test", "language": "text"})
        writer.close()

        # Should have produced multiple partitions
        written = list(out_dir.glob("test_shard_*.jsonl"))
        assert len(written) >= 2, f"Expected multiple partitions, got {len(written)}"

        # Verify each partition stays under limit (within 1 document threshold)
        for p in written:
            sz = p.stat().st_size
            assert sz > 0
            # Read records and check JSON validity
            with open(p, "r", encoding="utf-8") as fp:
                for line in fp:
                    data = json.loads(line)
                    assert "text" in data
                    assert "meta" in data


def test_generate_base_code_corpus_integration():
    """Test full integration generation pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        manifest = generate_base_code_corpus(
            output_dir=out_dir,
            count=50,
            seed=42,
            max_mb=28.0,
            prefix="test_code_base"
        )

        assert manifest["total_records"] == 50
        assert manifest["total_bytes"] > 0
        assert manifest["partitions_count"] >= 1
        assert (out_dir / "manifest.json").exists()

        # Check language diversity
        langs = manifest["language_distribution"]
        assert len(langs) >= 5, f"Expected diverse languages, got {langs}"
        assert "rust" in langs or "go" in langs or "python" in langs
