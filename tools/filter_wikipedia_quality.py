"""Wikipedia Quality & Relevance Filter for Base Pre-Training.

Extracts, filters, and shards high-information encyclopedic articles from raw Wikipedia dumps.

Quality Filters Applied:
1. Stub Elimination: Drops articles with < 200 words.
2. List & Roster Pruning: Drops articles titled "List of ..." or "XXXX in <country/sport>".
3. Disambiguation Stripping: Drops disambiguation pages ("may refer to:").
4. Formatting Density: Drops pages dominated by tables/lists (> 40% list/table lines).
5. Output Partitioning: Strictly shards JSONL output to <= 28.0 MB (29,360,128 bytes).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

MAX_PARTITION_BYTES = 28 * 1024 * 1024  # 28 MB ceiling

# Regex patterns for low-information filler
YEAR_EVENT_PATTERN = re.compile(r"^#\s+\d{4}\s+in\s+", re.IGNORECASE)
LIST_PATTERN = re.compile(r"^#\s+List\s+of\s+", re.IGNORECASE)
DISAMBIGUATION_PATTERN = re.compile(r"\(disambiguation\)", re.IGNORECASE)


class ShardedEncyclopedicWriter:
    """Writes JSONL records into strictly partitioned files under 28 MB."""

    def __init__(self, output_dir: Path, prefix: str = "encyclopedic_part", max_bytes: int = MAX_PARTITION_BYTES) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.part_idx = 1
        self.cur_file = self.output_dir / f"{self.prefix}_{self.part_idx:03d}.jsonl"
        self.cur_fp = open(self.cur_file, "w", encoding="utf-8")
        self.cur_bytes = 0
        self.cur_records = 0
        self.total_records = 0
        self.total_bytes = 0
        self.written_files: List[Path] = [self.cur_file]

    def write_article(self, title: str, text: str, word_count: int) -> None:
        doc = {
            "text": text,
            "meta": {
                "domain": "encyclopedic",
                "title": title,
                "word_count": word_count,
                "tokens": max(1, len(text) // 4)
            }
        }
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        b_len = len(line.encode("utf-8"))

        if self.cur_bytes + b_len > self.max_bytes and self.cur_records > 0:
            self.cur_fp.close()
            mb = self.cur_bytes / (1024 * 1024)
            print(f"  Saved {self.cur_file.name}: {self.cur_records:,} articles ({mb:.2f} MB)")
            self.part_idx += 1
            self.cur_file = self.output_dir / f"{self.prefix}_{self.part_idx:03d}.jsonl"
            self.cur_fp = open(self.cur_file, "w", encoding="utf-8")
            self.written_files.append(self.cur_file)
            self.cur_bytes = 0
            self.cur_records = 0

        self.cur_fp.write(line)
        self.cur_bytes += b_len
        self.cur_records += 1
        self.total_records += 1
        self.total_bytes += b_len

    def close(self) -> None:
        if not self.cur_fp.closed:
            self.cur_fp.close()
            if self.cur_records > 0:
                mb = self.cur_bytes / (1024 * 1024)
                print(f"  Saved {self.cur_file.name}: {self.cur_records:,} articles ({mb:.2f} MB)")
            elif len(self.written_files) > 1 and self.cur_file.exists() and self.cur_file.stat().st_size == 0:
                self.cur_file.unlink()
                self.written_files.remove(self.cur_file)


def extract_title_and_text(raw_block: str) -> Tuple[str, str]:
    """Extract clean title and content body from an article block."""
    lines = raw_block.strip().split("\n")
    title = ""
    start_idx = 0

    for idx, line in enumerate(lines):
        l = line.strip()
        if l.startswith("# "):
            title = l.lstrip("# ").strip()
            start_idx = idx
            break
        elif l.startswith("## "):
            title = l.lstrip("# ").strip()
            start_idx = idx
            break

    body = "\n".join(lines[start_idx:]).strip()
    return title, body


def is_quality_article(title: str, text: str, min_words: int = 200, max_table_ratio: float = 0.40) -> Tuple[bool, str]:
    """Assess whether a Wikipedia article passes frontier quality standards."""
    if not title or not text:
        return False, "empty"

    # 1. Disambiguation filter
    if DISAMBIGUATION_PATTERN.search(title) or "may refer to:" in text[:400].lower():
        return False, "disambiguation"

    # 2. Listicle & chronology filter
    if LIST_PATTERN.search(f"# {title}"):
        return False, "list_index"
    if YEAR_EVENT_PATTERN.search(f"# {title}"):
        return False, "year_chronology"

    # 3. Word count check
    words = text.split()
    word_count = len(words)
    if word_count < min_words:
        return False, "stub"

    # 4. Table and list density check
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False, "empty_lines"

    table_list_lines = sum(1 for l in lines if l.startswith("* ") or l.startswith("- ") or l.startswith("|") or l.startswith("+"))
    ratio = table_list_lines / len(lines)
    if ratio > max_table_ratio and word_count < 1000:
        return False, "table_list_heavy"

    return True, "pass"


def filter_and_shard_wikipedia(
    source_dirs: List[Path],
    output_dir: Path,
    target_mb: Optional[float] = 2500.0,  # ~2.5 GB target
    min_words: int = 200,
    max_mb_per_shard: float = 28.0,
    prefix: str = "encyclopedic_part"
) -> Dict[str, Any]:
    """Filter raw Wikipedia directories and write balanced, sharded JSONL files."""
    writer = ShardedEncyclopedicWriter(output_dir=output_dir, prefix=prefix, max_bytes=int(max_mb_per_shard * 1024 * 1024))
    
    stats = {
        "articles_processed": 0,
        "articles_passed": 0,
        "dropped_stubs": 0,
        "dropped_lists": 0,
        "dropped_chronologies": 0,
        "dropped_disambiguations": 0,
        "dropped_tables": 0,
    }

    target_bytes = int(target_mb * 1024 * 1024) if target_mb else None
    print(f"Starting Wikipedia Quality Filtering:")
    print(f"  Source Dirs: {[d.name for d in source_dirs]}")
    print(f"  Target Output: {output_dir} (Target Cap: {target_mb:.1f} MB)")
    print(f"  Min Words: {min_words} | Max Partition: {max_mb_per_shard} MB\n")

    try:
        for s_dir in source_dirs:
            if not s_dir.exists():
                print(f"  Warning: Directory not found: {s_dir}")
                continue

            md_files = sorted(s_dir.glob("*.md"))
            print(f"Processing folder {s_dir.name} ({len(md_files)} files)...")

            for md_file in md_files:
                if target_bytes and writer.total_bytes >= target_bytes:
                    print(f"Target size budget reached ({writer.total_bytes / (1024*1024):.2f} MB). Stopping.")
                    break

                try:
                    with open(md_file, "r", encoding="utf-8", errors="replace") as fp:
                        content = fp.read()

                    # Raw wiki files may use '## ---\n# Title' or '\n# Title'
                    split_pattern = re.compile(r'(?:\n## ---\n+|\n+)(?=# [^\n]+)')
                    blocks = split_pattern.split(content)
                    for block in blocks:
                        b_str = block.strip()
                        if not b_str:
                            continue
                        if b_str.startswith("## ---"):
                            b_str = b_str[6:].strip()

                        stats["articles_processed"] += 1
                        title, body = extract_title_and_text(b_str)

                        passed, reason = is_quality_article(title=title, text=body, min_words=min_words)
                        if passed:
                            writer.write_article(title=title, text=body, word_count=len(body.split()))
                            stats["articles_passed"] += 1
                            if target_bytes and writer.total_bytes >= target_bytes:
                                break
                        else:
                            if reason == "stub":
                                stats["dropped_stubs"] += 1
                            elif reason == "list_index":
                                stats["dropped_lists"] += 1
                            elif reason == "year_chronology":
                                stats["dropped_chronologies"] += 1
                            elif reason == "disambiguation":
                                stats["dropped_disambiguations"] += 1
                            elif reason == "table_list_heavy":
                                stats["dropped_tables"] += 1

                except Exception as err:
                    print(f"  Error reading {md_file.name}: {err}")

            if target_bytes and writer.total_bytes >= target_bytes:
                break
    finally:
        writer.close()

    manifest = {
        "version": "encyclopedic-curated-v1.0",
        "total_records": writer.total_records,
        "total_bytes": writer.total_bytes,
        "total_mb": writer.total_bytes / (1024 * 1024),
        "estimated_tokens": writer.total_bytes // 4,
        "partitions_count": len(writer.written_files),
        "partition_files": [f.name for f in writer.written_files],
        "filtering_stats": stats,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\nFiltering Complete!")
    print(f"  Articles Evaluated: {stats['articles_processed']:,}")
    print(f"  Articles Retained: {stats['articles_passed']:,} (Pass Rate: {stats['articles_passed']/max(1, stats['articles_processed'])*100:.1f}%)")
    print(f"  Dropped: Stubs={stats['dropped_stubs']:,}, Lists={stats['dropped_lists']:,}, Chronologies={stats['dropped_chronologies']:,}, Disambiguations={stats['dropped_disambiguations']:,}")
    print(f"  Final Size: {manifest['total_mb']:.2f} MB across {len(writer.written_files)} partitions")
    print(f"  Manifest written to: {manifest_path}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and shard raw Wikipedia dumps into quality encyclopedic corpus")
    parser.add_argument("--base-dataset", type=str, default=r"E:\AI_Projects\dataset", help="Base dataset root")
    parser.add_argument("--output-dir", type=str, default=r"E:\AI_Projects\dataset\encyclopedic", help="Output directory")
    parser.add_argument("--folders", nargs="+", default=["en_wiki_2026_01", "en_wiki_2026_02", "en_wiki_2026_03"], help="Wiki folders to scan")
    parser.add_argument("--target-mb", type=float, default=2500.0, help="Target MB cap for curated subset")
    parser.add_argument("--min-words", type=int, default=200, help="Minimum word count per article")
    parser.add_argument("--max-mb", type=float, default=28.0, help="Max MB per partition")
    args = parser.parse_args()

    base = Path(args.base_dataset)
    source_dirs = [base / f for f in args.folders]

    filter_and_shard_wikipedia(
        source_dirs=source_dirs,
        output_dir=Path(args.output_dir),
        target_mb=args.target_mb,
        min_words=args.min_words,
        max_mb_per_shard=args.max_mb,
    )


if __name__ == "__main__":
    main()
