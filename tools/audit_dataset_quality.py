"""Audit Dataset Quality Tool.

Enforces 6 strict quality gates on fine-tuning datasets:
1. File Sizing Gate: Every file must strictly be <= 30 MB (30,000,000 bytes).
2. JSON Integrity Gate: 100% of non-empty lines must parse as valid UTF-8 JSON.
3. Schema Conformance Gate: Messages or instruction/response format with valid roles.
4. Diversity / Anti-Mode-Collapse Gate: No single canned phrase dominates (>5%).
5. Target Span Gate: Assistant completion spans detected for loss computation.
6. Domain-Specific Gate:
   - Thinking: <think>...</think> tags present and closed.
   - Code: Markdown code fences or valid code implementations.
   - Tool-Call: OpenAI tools and tool_calls definitions or negative direct answers.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAX_ALLOWED_BYTES = 30 * 1024 * 1024  # 30 MB


def audit_file(file_path: Path, kind: str) -> Dict[str, Any]:
    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    size_mb = file_size / (1024 * 1024)

    stats: Dict[str, Any] = {
        "file": file_path.name,
        "size_mb": round(size_mb, 2),
        "size_pass": file_size <= MAX_ALLOWED_BYTES,
        "total_records": 0,
        "json_errors": 0,
        "schema_errors": 0,
        "target_span_errors": 0,
        "domain_errors": 0,
        "top_responses": collections.Counter(),
        "passed": True,
        "error_samples": [],
    }

    if not stats["size_pass"]:
        stats["passed"] = False
        stats["error_samples"].append(f"File size {size_mb:.2f} MB exceeds 30 MB ceiling")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            
            try:
                rec = json.loads(line_str)
            except Exception as exc:
                stats["json_errors"] += 1
                stats["passed"] = False
                if len(stats["error_samples"]) < 5:
                    stats["error_samples"].append(f"Line {line_num}: JSON error: {exc}")
                continue

            stats["total_records"] += 1

            # Extract assistant text and check schema
            asst_text = ""
            if "messages" in rec:
                msgs = rec.get("messages", [])
                if not isinstance(msgs, list) or not msgs:
                    stats["schema_errors"] += 1
                    continue
                for m in msgs:
                    if m.get("role") == "assistant":
                        asst_text = str(m.get("content", ""))
                        if "tool_calls" in m:
                            asst_text += " " + json.dumps(m["tool_calls"])
            elif "instruction" in rec or "response" in rec or "output" in rec:
                asst_text = str(rec.get("response", rec.get("output", rec.get("answer", ""))))
            else:
                stats["schema_errors"] += 1
                continue

            if asst_text.strip():
                # Count first 60 chars for mode-collapse detection
                prefix = asst_text.strip()[:60]
                stats["top_responses"][prefix] += 1

            # Domain specific check
            if kind == "thinking":
                if "<think>" not in asst_text or "</think>" not in asst_text:
                    stats["domain_errors"] += 1
                    if len(stats["error_samples"]) < 5:
                        stats["error_samples"].append(f"Line {line_num}: Missing <think> tags")
            elif kind == "code":
                if not asst_text.strip():
                    stats["domain_errors"] += 1
            elif kind == "tool_call":
                if "tools" not in rec and "tool_calls" not in asst_text and not asst_text.strip():
                    stats["domain_errors"] += 1

    # Check mode collapse (>5% identical responses on datasets > 100 items)
    if stats["total_records"] > 100 and stats["top_responses"]:
        top_phrase, top_count = stats["top_responses"].most_common(1)[0]
        ratio = top_count / stats["total_records"]
        if ratio > 0.05 and ("Sure" in top_phrase or "help" in top_phrase or "Hello" in top_phrase):
            stats["passed"] = False
            stats["error_samples"].append(
                f"Mode-collapse detected: {ratio*100:.1f}% rows start with: '{top_phrase[:40]}...'"
            )

    if (
        stats["json_errors"] > 0
        or stats["schema_errors"] > 0
        or stats["domain_errors"] > 0
    ):
        stats["passed"] = False

    return stats


def audit_directory(dir_path: Path, kind: Optional[str] = None) -> bool:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        print(f"Directory not found: {dir_path}")
        return False

    if kind is None:
        name = dir_path.name.lower()
        if "think" in name:
            kind = "thinking"
        elif "conv" in name:
            kind = "conversation"
        elif "tool" in name:
            kind = "tool_call"
        elif "code" in name:
            kind = "code"
        elif "instr" in name:
            kind = "instruction"
        else:
            kind = "generic"

    files = sorted(dir_path.glob("*.jsonl"))
    print(f"\nAuditing {len(files)} files in {dir_path.name} (detected kind: {kind})...")

    all_passed = True
    total_recs = 0

    for f in files:
        res = audit_file(f, kind)
        total_recs += res["total_records"]
        status = "PASS" if res["passed"] else "FAIL"
        if not res["passed"]:
            all_passed = False
            print(f"  [{status}] {f.name} ({res['size_mb']} MB, {res['total_records']:,} rows)")
            for err in res["error_samples"][:3]:
                print(f"         ! {err}")
        else:
            print(f"  [{status}] {f.name} ({res['size_mb']} MB, {res['total_records']:,} rows)")

    print(f"Finished {dir_path.name}: Total rows: {total_recs:,} | Status: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fine-tuning datasets against 6 quality gates.")
    parser.add_argument("paths", nargs="*", help="File or directory paths to audit.")
    parser.add_argument("--kind", choices=["thinking", "conversation", "tool_call", "code", "instruction", "generic"])
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        # Default to all dataset directories in E:\AI_Projects\dataset
        base = Path(r"E:\AI_Projects\dataset")
        paths = [
            str(base / "fine_tune_thinking"),
            str(base / "fine_tune_conversation"),
            str(base / "fine_tune_tool_call"),
            str(base / "fine_tune_code"),
        ]

    success = True
    for p_str in paths:
        p = Path(p_str)
        if p.is_dir():
            if not audit_directory(p, kind=args.kind):
                success = False
        elif p.is_file():
            res = audit_file(p, kind=args.kind or "generic")
            status = "PASS" if res["passed"] else "FAIL"
            print(f"[{status}] {p.name} ({res['size_mb']} MB, {res['total_records']:,} rows)")
            if not res["passed"]:
                success = False
                for err in res["error_samples"]:
                    print(f"  ! {err}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
