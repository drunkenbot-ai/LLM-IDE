"""Curate, Balance, and Shard a 2-Billion Token Base Pretraining Mixture for Frontier Models.

Enforces target distribution:
- 35% Encyclopedic / High-Quality Web (~700M tokens)
- 25% Code & Technical Documentation (~500M tokens)
- 15% Finance, Markets & Corporate (~300M tokens)
- 15% Reasoning, Math & STEM (~300M tokens)
- 10% Dialogue, Chat & Structured Analysis (~200M tokens)

Features:
- Enforces strict chunking: Every output partition is <= 28.0 MB (29,360,128 bytes).
- Fast streaming and deduplication based on content hashes.
- Generates manifest with domain breakdowns, partition counts, and token estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

MAX_PARTITION_BYTES = 28 * 1024 * 1024  # 28 MB ceiling


TARGET_RATIOS = {
    "encyclopedic": 0.35,
    "code": 0.25,
    "finance": 0.15,
    "reasoning": 0.15,
    "dialogue": 0.10,
}


def estimate_tokens_from_text(text: str) -> int:
    """Rough heuristic token estimator (~4 chars per token for English prose, ~3.2 for code)."""
    return max(1, len(text) // 4)


class ShardedWriter:
    """Writes JSONL or plain text records into partitioned files under 28 MB."""

    def __init__(self, output_dir: Path, prefix: str = "curated_part", max_bytes: int = MAX_PARTITION_BYTES) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.part_idx = 1
        self.cur_file = self.output_dir / f"{self.prefix}_{self.part_idx:04d}.jsonl"
        self.cur_fp = open(self.cur_file, "w", encoding="utf-8")
        self.cur_bytes = 0
        self.cur_records = 0
        self.total_records = 0
        self.total_bytes = 0
        self.written_files: List[Path] = [self.cur_file]

    def write_record(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        b_len = len(line.encode("utf-8"))

        if self.cur_bytes + b_len > self.max_bytes and self.cur_records > 0:
            self.cur_fp.close()
            print(f"  [Partition] Finalized {self.cur_file.name}: {self.cur_records:,} items ({self.cur_bytes / (1024*1024):.2f} MB)")
            self.part_idx += 1
            self.cur_file = self.output_dir / f"{self.prefix}_{self.part_idx:04d}.jsonl"
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
                print(f"  [Partition] Finalized {self.cur_file.name}: {self.cur_records:,} items ({self.cur_bytes / (1024*1024):.2f} MB)")
            elif len(self.written_files) > 1 and self.cur_file.exists() and self.cur_file.stat().st_size == 0:
                self.cur_file.unlink()
                self.written_files.remove(self.cur_file)


def stream_jsonl_records(paths: List[Path], max_items: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    seen_hashes = set()
    count = 0
    for p in paths:
        if not p.exists():
            continue
        files = [p] if p.is_file() else sorted(p.glob("*.jsonl"))
        for f in files:
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fp:
                    for line in fp:
                        l = line.strip()
                        if not l:
                            continue
                        # Exact hash deduplication
                        h = hashlib.md5(l.encode("utf-8")).hexdigest()
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                        try:
                            data = json.loads(l)
                            yield data
                            count += 1
                            if max_items and count >= max_items:
                                return
                        except Exception:
                            continue
            except Exception as err:
                print(f"Warning: Could not read {f}: {err}")


def curate_mixture(
    source_dirs: Dict[str, Path],
    output_dir: Path,
    target_tokens: int = 2_000_000_000,
    ratios: Dict[str, float] = TARGET_RATIOS,
    max_partition_mb: float = 28.0,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Mix and shard domains into balanced output files."""
    rng = random.Random(seed)
    max_bytes = int(max_partition_mb * 1024 * 1024)
    writer = ShardedWriter(output_dir, prefix="pretrain_mixture", max_bytes=max_bytes)

    domain_targets = {domain: int(target_tokens * ratio) for domain, ratio in ratios.items()}
    print(f"\nTarget Pretraining Token Budget: {target_tokens:,} tokens across {len(ratios)} domains:")
    for d, tok in domain_targets.items():
        print(f"  - {d.capitalize():<15}: {tok:>12,} tokens ({ratios[d]*100:.1f}%)")

    stats = {
        "target_tokens": target_tokens,
        "domains": {},
        "total_records": 0,
        "estimated_tokens": 0,
        "partitions": 0,
    }

    # Interleave sources according to ratios
    domain_iterators = {}
    for domain, s_path in source_dirs.items():
        if s_path.exists():
            domain_iterators[domain] = stream_jsonl_records([s_path])
        else:
            print(f"Notice: Source path for {domain} ({s_path}) not yet present; will use synthetic generator fallback if available.")

    accumulated_tokens = {d: 0 for d in ratios}
    total_tokens_written = 0

    active_domains = list(ratios.keys())
    while active_domains and total_tokens_written < target_tokens:
        chosen_domain = rng.choices(
            active_domains,
            weights=[ratios[d] for d in active_domains],
            k=1,
        )[0]

        iterator = domain_iterators.get(chosen_domain)
        record = None
        if iterator:
            try:
                record = next(iterator)
            except StopIteration:
                domain_iterators.pop(chosen_domain, None)
                active_domains.remove(chosen_domain)
                continue

        if record:
            # Measure record tokens
            rec_text = json.dumps(record)
            tok_count = estimate_tokens_from_text(rec_text)
            writer.write_record(record)
            accumulated_tokens[chosen_domain] += tok_count
            total_tokens_written += tok_count

            if accumulated_tokens[chosen_domain] >= domain_targets[chosen_domain] and chosen_domain in active_domains:
                active_domains.remove(chosen_domain)

        if not domain_iterators:
            break

    writer.close()

    manifest = {
        "version": "v2.0-curated-2b",
        "target_tokens": target_tokens,
        "tokens_written": total_tokens_written,
        "domain_breakdown": accumulated_tokens,
        "partitions_count": len(writer.written_files),
        "partition_files": [f.name for f in writer.written_files],
    }

    manifest_path = output_dir / "curation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nCuration complete! Manifest written to {manifest_path}")
    print(f"Total partitions: {len(writer.written_files)}, total tokens estimated: {total_tokens_written:,}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate balanced pretraining mixture for 2B tokens")
    parser.add_argument("--output-dir", type=str, default="dataset/curated_2b_base", help="Target output directory")
    parser.add_argument("--target-tokens", type=int, default=2_000_000_000, help="Target pretraining token count (default 2B)")
    parser.add_argument("--max-partition-mb", type=float, default=28.0, help="Max MB per partition (default 28MB)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    base_dataset = Path(r"E:\AI_Projects\dataset")
    source_dirs = {
        "encyclopedic": base_dataset / "encyclopedic",
        "code": base_dataset / "clean_code",
        "finance": base_dataset / "finance",
        "reasoning": base_dataset / "fine_tune_thinking",
        "dialogue": base_dataset / "fine_tune_conversation",
    }

    curate_mixture(
        source_dirs=source_dirs,
        output_dir=Path(args.output_dir),
        target_tokens=args.target_tokens,
        max_partition_mb=args.max_partition_mb,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
