#!/usr/bin/env python3
"""CLI utility to scrub repetitive system prompt boilerplate and near-duplicates from dataset folders."""

import argparse
import json
import sys
from pathlib import Path

# Ensure LLM-IDE repository root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.dataset_scrubber import scrub_jsonl_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrub repetitive boilerplate and near-duplicates from JSONL shards.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="E:/AI_Projects/dataset/curated_2b_base",
        help="Source directory containing .jsonl shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="E:/AI_Projects/dataset/curated_2b_base_clean",
        help="Destination directory for clean 28MB shards.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="curated_clean",
        help="Prefix for output shards.",
    )
    parser.add_argument(
        "--max-prefix-repeats",
        type=int,
        default=50,
        help="Maximum allowed occurrences of identical 48-char prefix boilerplate.",
    )

    args = parser.parse_args()

    print(f"[*] Starting Corpus Scrubber:")
    print(f"  Source: {args.source_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Max prefix repeats: {args.max_prefix_repeats}")

    def report_progress(scanned: int, retained: int) -> None:
        print(f"  -> Processed {scanned:,} records (retained: {retained:,})...")

    report = scrub_jsonl_directory(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
        max_prefix_repeats=args.max_prefix_repeats,
        progress_callback=report_progress,
    )

    print("\n[+] Scrub Complete!")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
