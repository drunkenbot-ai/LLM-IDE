"""Package Peer-Reviewed Research Papers and Scientific Ebooks into Sharded Base Pretraining Data.

Reads research papers (astrophysics, genomics, neuroscience) and scientific textbooks
(histology, medicine, physiology) from local dataset directories and converts them into
clean, sharded JSONL pretraining records strictly under 28 MB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

MAX_PARTITION_BYTES = 28 * 1024 * 1024  # 28 MB ceiling


class ShardedScienceWriter:
    """Writes JSONL base pretraining records into partitioned files under 28 MB."""

    def __init__(self, output_dir: Path, prefix: str = "science_part", max_bytes: int = MAX_PARTITION_BYTES) -> None:
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

    def write_record(self, text: str, source_type: str, title: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        doc = {
            "text": text,
            "meta": {
                "domain": "science",
                "source_type": source_type,
                "title": title,
                "word_count": len(text.split()),
                "tokens": max(1, len(text) // 4),
                **(metadata or {})
            }
        }
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        b_len = len(line.encode("utf-8"))

        if self.cur_bytes + b_len > self.max_bytes and self.cur_records > 0:
            self.cur_fp.close()
            mb = self.cur_bytes / (1024 * 1024)
            print(f"  Saved {self.cur_file.name}: {self.cur_records:,} records ({mb:.2f} MB)")
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
                print(f"  Saved {self.cur_file.name}: {self.cur_records:,} records ({mb:.2f} MB)")
            elif len(self.written_files) > 1 and self.cur_file.exists() and self.cur_file.stat().st_size == 0:
                self.cur_file.unlink()
                self.written_files.remove(self.cur_file)


def chunk_text(text: str, max_words: int = 1500) -> List[str]:
    """Split long scientific chapters into coherent topical passages."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: List[str] = []
    cur_chunk: List[str] = []
    cur_words = 0

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        p_words = len(p_clean.split())
        if cur_words + p_words > max_words and cur_chunk:
            chunks.append("\n\n".join(cur_chunk))
            cur_chunk = [p_clean]
            cur_words = p_words
        else:
            cur_chunk.append(p_clean)
            cur_words += p_words

    if cur_chunk:
        chunks.append("\n\n".join(cur_chunk))
    return chunks


def package_science_dataset(
    research_dir: Path,
    ebooks_dir: Path,
    output_dir: Path,
    max_file_mb: float = 28.0,
) -> List[Path]:
    """Package scientific research papers and textbooks into sharded JSONL files."""
    output_dir = Path(output_dir)
    max_bytes = int(max_file_mb * 1024 * 1024)
    writer = ShardedScienceWriter(output_dir, prefix="science_part", max_bytes=max_bytes)

    print(f"Packaging scientific literature from {research_dir} and {ebooks_dir}...")

    # 1. Process Research Papers
    papers_processed = 0
    if research_dir.exists():
        for p_file in sorted(research_dir.glob("*.md")):
            try:
                content = p_file.read_text(encoding="utf-8", errors="replace").strip()
                if len(content.split()) < 100:
                    continue  # Skip stubs
                title = p_file.stem
                if content.startswith("# "):
                    title = content.split("\n", 1)[0].replace("# ", "").strip()
                
                chunks = chunk_text(content, max_words=1500)
                for c in chunks:
                    if len(c.split()) >= 80:
                        writer.write_record(c, source_type="peer_reviewed_paper", title=title)
                papers_processed += 1
            except Exception as err:
                print(f"Warning: Failed to read {p_file.name}: {err}")

    print(f"  Processed {papers_processed} peer-reviewed research papers.")

    # 2. Process Scientific Ebooks
    books_processed = 0
    if ebooks_dir.exists():
        for b_file in sorted(ebooks_dir.glob("*.txt")):
            try:
                content = b_file.read_text(encoding="utf-8", errors="replace").strip()
                if len(content.split()) < 200:
                    continue
                title = b_file.stem
                chunks = chunk_text(content, max_words=1500)
                for c in chunks:
                    if len(c.split()) >= 100:
                        writer.write_record(c, source_type="scientific_textbook", title=title)
                books_processed += 1
            except Exception as err:
                print(f"Warning: Failed to read {b_file.name}: {err}")

    print(f"  Processed {books_processed} scientific textbook corpora.")
    writer.close()

    manifest = {
        "version": "v1.0-science-pretraining",
        "domain": "scientific_literature",
        "total_records": writer.total_records,
        "total_bytes": writer.total_bytes,
        "total_tokens_est": writer.total_bytes // 4,
        "partitions_count": len(writer.written_files),
        "partitions": [f.name for f in writer.written_files],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done! Created {len(writer.written_files)} partitions in {output_dir}")
    return writer.written_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Scientific Literature into Pretraining Shards")
    parser.add_argument("--research-dir", type=str, default=r"E:\AI_Projects\dataset\research_papers", help="Research papers directory")
    parser.add_argument("--ebooks-dir", type=str, default=r"E:\AI_Projects\dataset\ebooks", help="Scientific ebooks directory")
    parser.add_argument("--output-dir", type=str, default=r"E:\AI_Projects\dataset\science_pretraining", help="Output directory")
    parser.add_argument("--max-partition-mb", type=float, default=28.0, help="Max MB per partition")
    args = parser.parse_args()

    package_science_dataset(
        research_dir=Path(args.research_dir),
        ebooks_dir=Path(args.ebooks_dir),
        output_dir=Path(args.output_dir),
        max_file_mb=args.max_partition_mb,
    )


if __name__ == "__main__":
    main()
