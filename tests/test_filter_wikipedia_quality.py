"""Unit tests for Wikipedia Quality Filtering and Sharding Tool."""

import json
import tempfile
from pathlib import Path

import pytest
from tools.filter_wikipedia_quality import (
    ShardedEncyclopedicWriter,
    extract_title_and_text,
    filter_and_shard_wikipedia,
    is_quality_article,
)


def test_is_quality_article_rules():
    """Verify that quality heuristics correctly filter low-information articles."""
    # 1. Stub rejection
    short_text = "This is a short article about a person. Born in 1980."
    passed, reason = is_quality_article("John Doe", short_text, min_words=200)
    assert not passed
    assert reason == "stub"

    # 2. Listicle rejection
    list_title = "List of Roman Emperors"
    long_list_text = "List of Roman Emperors\n" + ("* Emperor " * 300)
    passed, reason = is_quality_article(list_title, long_list_text, min_words=200)
    assert not passed
    assert reason == "list_index"

    # 3. Year event chronology rejection
    year_title = "1974 in sports"
    year_text = "1974 in sports\n" + ("January 1: Match played.\n" * 150)
    passed, reason = is_quality_article(year_title, year_text, min_words=200)
    assert not passed
    assert reason == "year_chronology"

    # 4. Disambiguation rejection
    dis_title = "Mercury (disambiguation)"
    dis_text = "Mercury may refer to: the planet, the chemical element, the car brand.\n" + ("item " * 250)
    passed, reason = is_quality_article(dis_title, dis_text, min_words=200)
    assert not passed
    assert reason == "disambiguation"

    # 5. High-quality substantive article acceptance
    good_title = "General Relativity"
    good_text = (
        "General relativity, also known as the general theory of relativity, is the geometric theory of "
        "gravitation published by Albert Einstein in 1915 and is the current description of gravitation in modern physics. "
        "General relativity generalizes special relativity and refines Newton's law of universal gravitation, providing "
        "a unified description of gravity as a geometric property of space and time or four-dimensional spacetime. "
    ) * 10
    passed, reason = is_quality_article(good_title, good_text, min_words=200)
    assert passed
    assert reason == "pass"


def test_sharded_encyclopedic_writer():
    """Verify ShardedEncyclopedicWriter partitions correctly and stays under byte ceiling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        small_ceiling = 20 * 1024  # 20 KB
        writer = ShardedEncyclopedicWriter(output_dir=out_dir, prefix="test_enc", max_bytes=small_ceiling)

        doc = "A" * 6000  # 6KB doc
        for i in range(8):
            writer.write_article(title=f"Article {i}", text=doc, word_count=len(doc.split()))
        writer.close()

        written = list(out_dir.glob("test_enc_*.jsonl"))
        assert len(written) >= 2

        for p in written:
            assert p.stat().st_size > 0
            with open(p, "r", encoding="utf-8") as fp:
                for line in fp:
                    data = json.loads(line)
                    assert "text" in data
                    assert "meta" in data
                    assert data["meta"]["domain"] == "encyclopedic"


def test_filter_and_shard_wikipedia_integration():
    """Verify end-to-end extraction from simulated raw Wikipedia .md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        source_dir = base_dir / "raw_wiki"
        output_dir = base_dir / "filtered_wiki"
        source_dir.mkdir(parents=True, exist_ok=True)

        # Create mock raw wiki markdown file with multiple split articles
        mock_file = source_dir / "articles_00001.md"
        mock_content = (
            "## ---\n"
            "# Short Stub\n"
            "Just a tiny definition.\n"
            "## ---\n"
            "# List of Countries\n"
            "List of countries in the world.\n" + ("* Country\n" * 300) +
            "## ---\n"
            "# Photosynthesis in Plants\n"
            "Photosynthesis is a biological process used by plants and other organisms to convert light energy into chemical energy. "
            "Light energy is captured and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds. "
            "Photosynthesis is largely responsible for producing and maintaining the oxygen content of the Earth's atmosphere, and supplies "
            "most of the energy necessary for life on Earth. Although photosynthesis is performed differently by different species, the process "
            "always begins when energy from light is absorbed by proteins called reaction centres that contain green chlorophyll pigments. "
        ) * 4

        mock_file.write_text(mock_content, encoding="utf-8")

        manifest = filter_and_shard_wikipedia(
            source_dirs=[source_dir],
            output_dir=output_dir,
            target_mb=10.0,
            min_words=50,  # Lower for test
            max_mb_per_shard=28.0,
            prefix="test_part"
        )

        assert manifest["total_records"] > 0
        assert (output_dir / "manifest.json").exists()
        assert manifest["filtering_stats"]["dropped_stubs"] >= 1
