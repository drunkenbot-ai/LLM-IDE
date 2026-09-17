#!/usr/bin/env python3
"""Convert instruction dataset files between OpenAI Chat messages format and App instruction format.

Features:
- Stream-based line-by-line processing (low RAM usage, handles 100s of GBs).
- Converts OpenAI messages format to app-compatible instruction/response format.
- Supports multi-turn dialogue unfolding (preserves dialogue context).
- Supports single file or recursive directory batch conversion.
- Safe atomic in-place updates or output to a separate directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Generator, Optional


def _clean_text(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val).strip()


def convert_record_to_app_instruction(
    rec: dict[str, Any],
    multiturn_mode: str = "unfold",
) -> list[dict[str, str]]:
    """Convert any record (OpenAI messages, ShareGPT, prompt/completion) to App instruction format."""
    # If already in app instruction format, preserve it directly
    if "instruction" in rec and any(k in rec for k in ("response", "output", "answer", "completion")):
        instr = _clean_text(rec.get("instruction"))
        user_in = _clean_text(rec.get("input"))
        resp = _clean_text(
            rec.get("response")
            or rec.get("output")
            or rec.get("answer")
            or rec.get("completion")
        )
        if user_in:
            instr = f"{instr}\n\nInput:\n{user_in}" if instr else user_in
        if instr and resp:
            return [{"instruction": instr, "response": resp}]

    # Handle prompt / completion
    if "prompt" in rec and any(k in rec for k in ("completion", "response", "output")):
        prompt = _clean_text(rec.get("prompt"))
        resp = _clean_text(rec.get("completion") or rec.get("response") or rec.get("output"))
        if prompt and resp:
            return [{"instruction": prompt, "response": resp}]

    # Handle question / answer
    if "question" in rec and any(k in rec for k in ("answer", "response", "output")):
        q = _clean_text(rec.get("question"))
        resp = _clean_text(rec.get("answer") or rec.get("response") or rec.get("output"))
        if q and resp:
            return [{"instruction": q, "response": resp}]

    # Handle OpenAI / ShareGPT messages format
    messages = rec.get("messages") or rec.get("conversations") or rec.get("turns") or rec.get("dialogue")
    if not isinstance(messages, list) or not messages:
        return []

    system_prompt = ""
    turns: list[tuple[str, str]] = []

    for m in messages:
        if isinstance(m, str):
            continue
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or m.get("from") or m.get("speaker") or "").strip().lower()
        content = _clean_text(m.get("content") or m.get("value") or m.get("text"))
        if not content:
            continue

        if role == "system":
            system_prompt = f"{system_prompt}\n{content}".strip() if system_prompt else content
        elif role in ("user", "human"):
            turns.append(("user", content))
        elif role in ("assistant", "gpt", "bot", "model"):
            turns.append(("assistant", content))

    if not turns:
        return []

    results: list[dict[str, str]] = []
    history: list[tuple[str, str]] = []

    for role, content in turns:
        if role == "assistant":
            if not history:
                continue

            prev_user = history[-1][1]
            prior_history = history[:-1]

            if multiturn_mode == "last":
                # Only use the immediate user turn + system prompt
                instr_parts = []
                if system_prompt:
                    instr_parts.append(f"System: {system_prompt}")
                instr_parts.append(f"User: {prev_user}")
                instruction_text = "\n\n".join(instr_parts)
                return [{"instruction": instruction_text, "response": content}]

            # Unfold mode: include cumulative prior dialogue history as context
            instr_parts = []
            if system_prompt:
                instr_parts.append(f"System: {system_prompt}")
            for hist_role, hist_content in prior_history:
                prefix = "User" if hist_role == "user" else "Assistant"
                instr_parts.append(f"{prefix}: {hist_content}")
            instr_parts.append(f"User: {prev_user}")

            if len(instr_parts) == 1 and not system_prompt:
                instruction_text = prev_user
            else:
                instruction_text = "\n\n".join(instr_parts)

            results.append({"instruction": instruction_text, "response": content})
            history.append((role, content))
        else:
            history.append((role, content))

    return results


def convert_record_to_openai(rec: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert Alpaca instruction format to OpenAI messages format."""
    # Already in OpenAI messages format
    if "messages" in rec and isinstance(rec["messages"], list):
        return rec

    instruction = _clean_text(rec.get("instruction") or rec.get("prompt") or rec.get("question"))
    user_in = _clean_text(rec.get("input"))
    response = _clean_text(
        rec.get("response") or rec.get("output") or rec.get("answer") or rec.get("completion")
    )

    if not (instruction or user_in) or not response:
        return None

    user_content = f"{instruction}\n\nInput:\n{user_in}" if (instruction and user_in) else (instruction or user_in)
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }


def convert_file(
    input_path: Path,
    output_path: Path,
    target: str = "app",
    multiturn_mode: str = "unfold",
) -> tuple[int, int, int]:
    """Stream-convert a single JSONL file.

    Returns:
        tuple of (total_read, valid_written, invalid_lines)
    """
    total_read = 0
    valid_written = 0
    invalid_lines = 0

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp_conv")
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with (
            open(input_path, "r", encoding="utf-8", errors="replace") as fin,
            open(temp_path, "w", encoding="utf-8") as fout,
        ):
            for line_idx, line in enumerate(fin, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue
                total_read += 1

                try:
                    record = json.loads(raw_line)
                except Exception:
                    invalid_lines += 1
                    continue

                if not isinstance(record, dict):
                    invalid_lines += 1
                    continue

                if target == "app":
                    converted_records = convert_record_to_app_instruction(record, multiturn_mode=multiturn_mode)
                    if converted_records:
                        for cr in converted_records:
                            fout.write(json.dumps(cr, ensure_ascii=False) + "\n")
                            valid_written += 1
                    else:
                        invalid_lines += 1
                else:  # target == "openai"
                    converted = convert_record_to_openai(record)
                    if converted:
                        fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        valid_written += 1
                    else:
                        invalid_lines += 1

        # Atomically replace
        if temp_path.exists():
            temp_path.replace(output_path)

    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise

    return total_read, valid_written, invalid_lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert dataset files to be compatible with LLM-IDE Instruction Fine-Tune or OpenAI format."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a JSONL file or directory containing JSONL files.",
    )
    parser.add_argument(
        "--target",
        choices=["app", "openai"],
        default="app",
        help="Conversion target: 'app' (instruction/response for LLM-IDE, default) or 'openai' (messages array).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory to save converted files. If not specified, --in-place or --suffix is required.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input files in-place with converted records (atomic replace).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a .bak copy before replacing in-place.",
    )
    parser.add_argument(
        "--multiturn",
        choices=["unfold", "last"],
        default="unfold",
        help="How to handle multi-turn conversations: 'unfold' (create samples preserving dialogue history, default) or 'last' (only final turn).",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    if not args.output_dir and not args.in_place:
        print(
            "[ERROR] You must specify either --output-dir <DIR> or --in-place to proceed safely.",
            file=sys.stderr,
        )
        return 1

    files_to_process: list[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() == ".jsonl":
            files_to_process.append(input_path)
        else:
            print(f"[ERROR] Input file is not a .jsonl file: {input_path}", file=sys.stderr)
            return 1
    else:
        files_to_process = sorted(p for p in input_path.rglob("*.jsonl") if not p.name.endswith(".tmp_conv"))

    if not files_to_process:
        print(f"[WARNING] No .jsonl files found in {input_path}")
        return 0

    out_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Dataset Converter -> Target: {args.target.upper()}")
    print(f"Total files found: {len(files_to_process)}")
    if out_dir:
        print(f"Output directory:  {out_dir}")
    else:
        print("Mode:              IN-PLACE replacement (atomic)")
        if args.backup:
            print("Backup:            Enabled (.bak files will be created)")
    print(f"Multi-turn mode:   {args.multiturn}")
    print("=" * 70)

    start_time = time.time()
    grand_total_read = 0
    grand_total_written = 0
    grand_total_invalid = 0

    for idx, fpath in enumerate(files_to_process, start=1):
        rel_path = fpath.relative_to(input_path) if input_path.is_dir() else fpath.name
        if out_dir:
            target_out = out_dir / rel_path
        else:
            target_out = fpath

        if args.in_place and args.backup:
            bak_path = fpath.with_suffix(fpath.suffix + ".bak")
            if not bak_path.exists():
                shutil.copyfile(fpath, bak_path)

        f_start = time.time()
        file_size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"[{idx}/{len(files_to_process)}] Processing {rel_path} ({file_size_mb:.1f} MB)...", flush=True)

        try:
            total_read, valid_written, invalid_lines = convert_file(
                fpath,
                target_out,
                target=args.target,
                multiturn_mode=args.multiturn,
            )
            elapsed = max(0.001, time.time() - f_start)
            rate = valid_written / elapsed
            print(
                f"    -> Converted {valid_written:,} sample(s) "
                f"({total_read:,} read, {invalid_lines:,} skipped) "
                f"in {elapsed:.2f}s ({rate:,.0f} samples/s)",
                flush=True,
            )
            grand_total_read += total_read
            grand_total_written += valid_written
            grand_total_invalid += invalid_lines
        except Exception as exc:
            print(f"    [ERROR] Failed to process {rel_path}: {exc}", file=sys.stderr)

    total_time = max(0.001, time.time() - start_time)
    print("=" * 70)
    print("CONVERSION COMPLETE!")
    print(f"Files processed:        {len(files_to_process)}")
    print(f"Total raw lines read:   {grand_total_read:,}")
    print(f"Total valid written:    {grand_total_written:,}")
    print(f"Invalid / skipped lines:{grand_total_invalid:,}")
    print(f"Total elapsed time:     {total_time:.1f}s ({grand_total_written / total_time:,.0f} samples/s)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
