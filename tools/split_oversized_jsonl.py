"""Utility to split JSONL files exceeding 30MB into clean <= 28MB parts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def split_jsonl_file(
    file_path: Path,
    max_bytes_per_file: int = 28 * 1024 * 1024, # 28MB
    remove_original: bool = True,
) -> list[Path]:
    """Split an oversized JSONL file into clean parts under max_bytes_per_file."""

    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    if file_size <= max_bytes_per_file:
        print(f"Skipping {file_path.name}: {file_size / (1024*1024):.2f} MB is already <= 28 MB")
        return [file_path]

    print(f"Splitting {file_path.name} ({file_size / (1024*1024):.2f} MB)...")
    
    stem = file_path.stem
    parent = file_path.parent
    
    part_idx = 1
    current_part_path = parent / f"{stem}_subpart_{part_idx:03d}.jsonl"
    current_handle = open(current_part_path, "w", encoding="utf-8", newline="\n")
    current_bytes = 0
    
    created_parts: list[Path] = [current_part_path]
    total_lines = 0
    part_lines = 0
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as in_f:
        for line in in_f:
            if not line.strip():
                continue
            line_bytes = len(line.encode("utf-8"))
            
            if current_bytes + line_bytes > max_bytes_per_file and part_lines > 0:
                current_handle.close()
                part_size_mb = current_part_path.stat().st_size / (1024 * 1024)
                print(f"  Created {current_part_path.name}: {part_lines:,} rows, {part_size_mb:.2f} MB")
                
                part_idx += 1
                current_part_path = parent / f"{stem}_subpart_{part_idx:03d}.jsonl"
                current_handle = open(current_part_path, "w", encoding="utf-8", newline="\n")
                current_bytes = 0
                part_lines = 0
                created_parts.append(current_part_path)
                
            current_handle.write(line)
            current_bytes += line_bytes
            part_lines += 1
            total_lines += 1
            
    current_handle.close()
    part_size_mb = current_part_path.stat().st_size / (1024 * 1024)
    print(f"  Created {current_part_path.name}: {part_lines:,} rows, {part_size_mb:.2f} MB")
    
    # Verify row count
    split_total = sum(sum(1 for _ in open(p, "r", encoding="utf-8")) for p in created_parts)
    assert split_total == total_lines, f"Row mismatch! Original: {total_lines}, Split: {split_total}"
    print(f"Verified {file_path.name}: all {total_lines:,} rows preserved across {len(created_parts)} parts.")
    
    if remove_original:
        file_path.unlink()
        print(f"Removed original oversized file: {file_path.name}\n")
        
    return created_parts


def split_directory(
    dir_path: Path,
    max_mb: float = 28.0,
    remove_original: bool = True,
) -> None:
    """Scan directory and split any JSONL exceeding max_mb."""

    dir_path = Path(dir_path)
    max_bytes = int(max_mb * 1024 * 1024)
    
    files = sorted(dir_path.glob("*.jsonl"))
    oversized = [f for f in files if f.stat().st_size > max_bytes and "_subpart_" not in f.name]
    
    print(f"Found {len(oversized)} oversized JSONL files (> {max_mb} MB) in {dir_path.name}:")
    for f in oversized:
        print(f"  {f.name}: {f.stat().st_size / (1024*1024):.2f} MB")
        
    for f in oversized:
        split_jsonl_file(f, max_bytes_per_file=max_bytes, remove_original=remove_original)
        
    print(f"Completed splitting for {dir_path.name}!\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split oversized JSONL files to <= 28MB.")
    parser.add_argument("paths", nargs="+", help="Files or directories to process.")
    parser.add_argument("--max-mb", type=float, default=28.0, help="Maximum MB per file.")
    parser.add_argument("--keep-original", action="store_true", help="Do not delete original.")
    args = parser.parse_args()
    
    for p_str in args.paths:
        p = Path(p_str)
        if p.is_dir():
            split_directory(p, max_mb=args.max_mb, remove_original=not args.keep_original)
        elif p.is_file():
            split_jsonl_file(p, max_bytes_per_file=int(args.max_mb * 1024 * 1024), remove_original=not args.keep_original)


if __name__ == "__main__":
    main()
