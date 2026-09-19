"""Unit tests for Clinical Medicine, Pharmacology & Molecular Genetics Base Pretraining Generator."""

import json
import random
import tempfile
from pathlib import Path
import pytest

from tools.generate_base_medicine_pretraining import (
    GENERATORS,
    ShardedMedicineWriter,
    generate_batch,
    write_partitioned_dataset,
)
from tools.audit_dataset_quality import audit_directory, audit_file


def test_medicine_generators_validity():
    """Verify that all 8 medical/pharmacological generators produce authentic clinical text."""
    rng = random.Random(42)
    med_keywords = {"patient", "clinical", "diagnosis", "therapy", "mg", "receptor", "serum", "pathology", "arterial", "cellular", "protein"}

    for gen_fn, subspecialty in GENERATORS:
        text = gen_fn(rng)
        assert isinstance(text, str)
        assert len(text) > 400, f"Generator {gen_fn.__name__} output too short ({len(text)} chars)"
        text_lower = text.lower()
        assert any(kw in text_lower for kw in med_keywords), f"Missing biomedical keywords in {gen_fn.__name__}"


def test_sharded_medicine_writer_ceiling():
    """Ensure ShardedMedicineWriter rolls over when partition limit is hit and stays <= max_bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        small_ceiling = 20 * 1024  # 20 KB ceiling
        writer = ShardedMedicineWriter(output_dir=out_dir, prefix="test_med", max_bytes=small_ceiling)

        sample_doc = "# Clinical Case\n" + ("Patient presents with acute symptoms.\n" * 300)
        for _ in range(6):
            writer.write_record(text=sample_doc, subspecialty="test_case")
        writer.close()

        written = list(out_dir.glob("test_med_*.jsonl"))
        assert len(written) >= 2, f"Expected multiple partitions, got {len(written)}"
        for p in written:
            assert p.stat().st_size <= small_ceiling + 1024
            with open(p, "r", encoding="utf-8") as fp:
                for line in fp:
                    data = json.loads(line)
                    assert "text" in data
                    assert "meta" in data
                    assert data["meta"]["domain"] == "medicine"


def test_write_partitioned_medicine_dataset_audit():
    """Verify medicine dataset generation integration and quality audit pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        written = write_partitioned_dataset(
            output_dir=out_dir,
            count=30,
            max_file_mb=0.04,
            seed=99,
        )

        assert len(written) >= 2
        assert (out_dir / "manifest.json").exists()
        assert audit_directory(out_dir, kind="medicine") is True
