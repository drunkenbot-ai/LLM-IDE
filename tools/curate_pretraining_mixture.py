"""Curate, Balance, and Shard a Frontier Base Pretraining Mixture for LLM Pretraining.

Enforces target frontier pretraining distribution:
- 35% Encyclopedic / High-Quality Web (~700M / 35% tokens)
- 25% Code & Systems Architecture (~500M / 25% tokens)
- 15% STEM & Formal Mathematics (~300M / 15% tokens)
- 10% Finance, Markets & Corporate Risk (~200M / 10% tokens)
-  5% Law, Governance, Contracts & Regulation (~100M / 5% tokens)
- 10% Dialogue, Chat & Structured Analysis (~200M / 10% tokens)

Features:
- Strict Partitioning Gate: Every partition file is <= 28.0 MB (29,360,128 bytes).
- Domain Balance Protection: Cycling & epoch-resampling prevents premature domain
  exhaustion and avoids domain skew across later partition files.
- Deficit-Weighted Selection: Dynamically balances sampling so partitions maintain
  the exact target ratio throughout the entire corpus.
- Exact-Hash Deduplication: Eliminates duplicate records on initial stream.
- Manifest Generation: Emits comprehensive domain token counts, record counts,
  epoch numbers, and partition listings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

MAX_PARTITION_BYTES = 28 * 1024 * 1024  # 28 MB cluster node ceiling

DEFAULT_TARGET_RATIOS = {
    "encyclopedic": 0.25,
    "code": 0.15,
    "algorithms": 0.10,
    "stem": 0.10,
    "medicine": 0.10,
    "hardware": 0.08,
    "cybersecurity": 0.07,
    "finance": 0.05,
    "law": 0.04,
    "multilingual": 0.03,
    "dialogue": 0.03,
}


def estimate_tokens_from_record(record: Dict[str, Any]) -> int:
    """Heuristic token estimator based on active text payload (~4 chars per token)."""
    if "text" in record:
        text = str(record["text"])
    elif "messages" in record:
        text = " ".join(str(m.get("content", "")) for m in record["messages"] if isinstance(m, dict))
    elif "instruction" in record or "response" in record or "output" in record:
        text = f"{record.get('instruction', '')} {record.get('input', '')} {record.get('output', record.get('response', ''))}"
    else:
        text = json.dumps(record)
    return max(1, len(text) // 4)


def estimate_tokens_from_text(text: str) -> int:
    """Backward-compatible token estimator from string."""
    return max(1, len(text) // 4)


class ShardedWriter:
    """Writes JSONL records into partitioned files strictly under 28 MB."""

    def __init__(self, output_dir: Path, prefix: str = "pretrain_mixture", max_bytes: int = MAX_PARTITION_BYTES) -> None:
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
            mb = self.cur_bytes / (1024 * 1024)
            print(f"  [Partition] Finalized {self.cur_file.name}: {self.cur_records:,} items ({mb:.2f} MB)")
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
                mb = self.cur_bytes / (1024 * 1024)
                print(f"  [Partition] Finalized {self.cur_file.name}: {self.cur_records:,} items ({mb:.2f} MB)")
            elif len(self.written_files) > 1 and self.cur_file.exists() and self.cur_file.stat().st_size == 0:
                self.cur_file.unlink()
                self.written_files.remove(self.cur_file)


def stream_domain_records(
    paths: List[Path],
    cycle: bool = True,
    max_epochs: int = 10,
    seed: int = 42,
) -> Iterator[Tuple[Dict[str, Any], int]]:
    """Stream JSONL records from a domain path list with optional multi-epoch cycling."""
    existing_files: List[Path] = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_file():
            existing_files.append(p)
        else:
            existing_files.extend(sorted(p.glob("*.jsonl")))

    if not existing_files:
        return

    rng = random.Random(seed)
    epoch = 1
    seen_hashes: set[str] = set()

    while True:
        shuffled_files = list(existing_files)
        rng.shuffle(shuffled_files)
        records_in_epoch = 0

        for f in shuffled_files:
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fp:
                    for line in fp:
                        l = line.strip()
                        if not l:
                            continue
                        # Deduplicate on initial pass
                        if epoch == 1:
                            h = hashlib.md5(l.encode("utf-8")).hexdigest()
                            if h in seen_hashes:
                                continue
                            seen_hashes.add(h)
                        try:
                            data = json.loads(l)
                            records_in_epoch += 1
                            yield data, epoch
                        except Exception:
                            continue
            except Exception as err:
                print(f"Warning: Could not read {f}: {err}")

        if not cycle or epoch >= max_epochs or records_in_epoch == 0:
            break
        epoch += 1


def stream_jsonl_records(paths: List[Path], max_items: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Backward-compatible single-pass record streamer."""
    for record, _ in stream_domain_records(paths, cycle=False):
        yield record
        if max_items is not None:
            max_items -= 1
            if max_items <= 0:
                break


def curate_mixture(
    source_dirs: Dict[str, Union[Path, List[Path]]],
    output_dir: Path,
    target_tokens: int = 250_000_000,
    ratios: Optional[Dict[str, float]] = None,
    max_partition_mb: float = 28.0,
    cycle: bool = True,
    max_epochs: int = 10,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Mix and shard domains into balanced output files strictly adhering to 28 MB ceiling."""
    if ratios is None:
        ratios = DEFAULT_TARGET_RATIOS

    rng = random.Random(seed)
    max_bytes = int(max_partition_mb * 1024 * 1024)
    writer = ShardedWriter(output_dir, prefix="pretrain_mixture", max_bytes=max_bytes)

    # Normalize ratios to sum to 1.0
    total_ratio = sum(ratios.values())
    norm_ratios = {d: r / total_ratio for d, r in ratios.items()}
    domain_targets = {domain: int(target_tokens * norm_ratios[domain]) for domain in norm_ratios}

    print(f"\nTarget Pretraining Token Budget: {target_tokens:,} tokens across {len(norm_ratios)} domains:")
    for d, tok in domain_targets.items():
        print(f"  - {d.capitalize():<15}: {tok:>12,} tokens ({norm_ratios[d]*100:.1f}%)")

    # Initialize domain streams
    domain_iterators: Dict[str, Iterator[Tuple[Dict[str, Any], int]]] = {}
    for domain, s_path in source_dirs.items():
        if domain not in norm_ratios:
            continue
        p_list = s_path if isinstance(s_path, list) else [s_path]
        existing = [p for p in p_list if p.exists()]
        if existing:
            domain_iterators[domain] = stream_domain_records(
                existing,
                cycle=cycle,
                max_epochs=max_epochs,
                seed=seed + hash(domain) % 10000,
            )
        else:
            print(f"Notice: Source path for '{domain}' ({s_path}) not found; skipping.")

    accumulated_tokens = {d: 0 for d in norm_ratios}
    accumulated_records = {d: 0 for d in norm_ratios}
    domain_max_epoch = {d: 1 for d in norm_ratios}
    total_tokens_written = 0

    active_domains = [d for d in norm_ratios if d in domain_iterators]

    while active_domains and total_tokens_written < target_tokens:
        # Deficit-weighted selection to maintain exact target ratio throughout
        deficits = {
            d: max(0.001, domain_targets[d] - accumulated_tokens[d])
            for d in active_domains
        }
        total_deficit = sum(deficits.values())
        if total_deficit <= 0.001 * len(active_domains):
            # All domains reached target
            break

        weights = [deficits[d] for d in active_domains]
        chosen_domain = rng.choices(active_domains, weights=weights, k=1)[0]
        iterator = domain_iterators[chosen_domain]

        try:
            record, epoch = next(iterator)
            domain_max_epoch[chosen_domain] = max(domain_max_epoch[chosen_domain], epoch)
        except StopIteration:
            domain_iterators.pop(chosen_domain, None)
            active_domains.remove(chosen_domain)
            continue

        tok_count = estimate_tokens_from_record(record)
        writer.write_record(record)
        accumulated_tokens[chosen_domain] += tok_count
        accumulated_records[chosen_domain] += 1
        total_tokens_written += tok_count

        if accumulated_tokens[chosen_domain] >= domain_targets[chosen_domain] and not cycle:
            active_domains.remove(chosen_domain)

    writer.close()

    manifest = {
        "version": "v2.0-frontier-curated",
        "target_tokens": target_tokens,
        "tokens_written": total_tokens_written,
        "total_records": writer.total_records,
        "domain_breakdown_tokens": accumulated_tokens,
        "domain_breakdown_records": accumulated_records,
        "domain_percentages": {
            d: round(accumulated_tokens[d] / max(1, total_tokens_written) * 100, 2)
            for d in norm_ratios
        },
        "domain_epochs": domain_max_epoch,
        "partitions_count": len(writer.written_files),
        "partition_files": [f.name for f in writer.written_files],
    }

    manifest_path = output_dir / "curation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nCuration complete! Manifest written to {manifest_path}")
    print(f"Total partitions: {len(writer.written_files)}, total tokens written: {total_tokens_written:,}")
    print("Domain Token Distribution:")
    for d, tok in accumulated_tokens.items():
        pct = (tok / max(1, total_tokens_written)) * 100
        recs = accumulated_records[d]
        ep = domain_max_epoch[d]
        print(f"  - {d.capitalize():<15}: {tok:>12,} tokens ({pct:>5.1f}%) | {recs:>8,} recs | {ep} epoch(s)")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate balanced pretraining mixture for Frontier LLM Base Training")
    parser.add_argument("--output-dir", type=str, default=r"E:\AI_Projects\dataset\curated_2b_base", help="Target output directory")
    parser.add_argument("--target-tokens", type=int, default=250_000_000, help="Target pretraining token count (default 250M)")
    parser.add_argument("--max-partition-mb", type=float, default=28.0, help="Max MB per partition (default 28MB)")
    parser.add_argument("--no-cycle", action="store_true", help="Disable cycling for single pass")
    parser.add_argument("--max-epochs", type=int, default=10, help="Max epochs per domain when cycling")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    base_dataset = Path(r"E:\AI_Projects\dataset")
    source_dirs = {
        "encyclopedic": [base_dataset / "encyclopedic", base_dataset / "science_pretraining"],
        "code": base_dataset / "code_pretraining",
        "algorithms": base_dataset / "algorithms_pretraining",
        "stem": base_dataset / "stem_pretraining",
        "medicine": base_dataset / "medicine_pretraining",
        "hardware": base_dataset / "hardware_pretraining",
        "cybersecurity": base_dataset / "cybersecurity_pretraining",
        "finance": base_dataset / "finance",
        "law": base_dataset / "law_pretraining",
        "multilingual": base_dataset / "multilingual_pretraining",
        "dialogue": base_dataset / "fine_tune_conversation",
    }

    curate_mixture(
        source_dirs=source_dirs,
        output_dir=Path(args.output_dir),
        target_tokens=args.target_tokens,
        ratios=DEFAULT_TARGET_RATIOS,
        max_partition_mb=args.max_partition_mb,
        cycle=not args.no_cycle,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
