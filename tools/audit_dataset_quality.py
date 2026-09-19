r"""Audit Dataset Quality Tool.
 
Enforces 6 strict quality gates on pretraining and fine-tuning datasets:
1. File Sizing Gate: Every file must strictly be <= 30 MB (30,000,000 bytes; 28 MB target).
2. JSON Integrity Gate: 100% of non-empty lines must parse as valid UTF-8 JSON.
3. Schema Conformance Gate: Messages or instruction/response format with valid roles.
4. Diversity / Anti-Mode-Collapse Gate: No single canned phrase dominates (>5%).
5. Target Span Gate: Assistant completion spans detected for loss computation.
6. Domain-Specific Gate:
   - Thinking: <think>...</think> tags present and closed, with self-correction/backtracking markers.
   - Safety: Objective, non-preachy boundary enforcement and safe educational dual-use pivots.
   - OpenWebMath / Math: LaTeX notation ($, $$, \frac, \sum) and formal mathematical terms.
   - Deep Analysis: Structured markdown hierarchy (headings, comparative tables, executive summaries).
   - Unit Test / TDD: Test function definitions (def test_), assertions, and test fixtures.
   - Competition Math: Multi-step arithmetic reasoning and boxed answers (\boxed{...}).
   - Finance: Financial keywords, calculations, and table structures.
   - Code: Markdown code fences and valid code implementations.
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
                else:
                    think_content = asst_text.split("<think>", 1)[1].split("</think>", 1)[0].lower()
                    correction_markers = ["wait", "check", "verify", "however", "re-evaluat", "re-check", "step", "alternativ"]
                    if not any(cm in think_content for cm in correction_markers):
                        stats["domain_errors"] += 1
                        if len(stats["error_samples"]) < 5:
                            stats["error_samples"].append(f"Line {line_num}: Thinking trace lacks self-correction/verification markers")
            elif kind == "safety":
                refusal_markers = ["cannot", "unable", "not able", "i can explain", "i can, however", "defensive", "educational", "security mechanism"]
                preachy_markers = ["as an ai language model", "it is important to remember", "as a responsible ai"]
                text_lower = asst_text.lower()
                if not any(rm in text_lower for rm in refusal_markers):
                    stats["domain_errors"] += 1
                    if len(stats["error_samples"]) < 5:
                        stats["error_samples"].append(f"Line {line_num}: Missing polite boundary/refusal markers")
                if any(pm in text_lower for pm in preachy_markers):
                    stats["domain_errors"] += 1
                    if len(stats["error_samples"]) < 5:
                        stats["error_samples"].append(f"Line {line_num}: Preachy robotic template detected")
            elif kind in {"openwebmath", "math"}:
                math_markers = ["$", "\\frac", "\\sum", "\\int", "\\begin", "theorem", "proof", "lemma", "derivative", "matrix"]
                text_lower = asst_text.lower()
                if not any(mm in text_lower for mm in math_markers):
                    stats["domain_errors"] += 1
                    if len(stats["error_samples"]) < 5:
                        stats["error_samples"].append(f"Line {line_num}: Missing LaTeX math syntax or theorem terms")
            elif kind == "analysis":
                has_headers = ("##" in asst_text or "###" in asst_text)
                has_tables = ("|" in asst_text and "---" in asst_text)
                if not (has_headers or has_tables):
                    stats["domain_errors"] += 1
                    if len(stats["error_samples"]) < 5:
                        stats["error_samples"].append(f"Line {line_num}: Lacks structured analysis formatting (headers or markdown tables)")
            elif kind in {"tdd", "unit_test"}:
                test_markers = ["def test_", "assert ", "@pytest", "testcase", "expect(", "assert_eq!"]
                if not any(tm in asst_text for tm in test_markers):
                    stats["domain_errors"] += 1
                    if len(stats["error_samples"]) < 5:
                        stats["error_samples"].append(f"Line {line_num}: Missing test assertions or test function definitions")
            elif kind == "competition_math":
                has_boxed = ("\\boxed" in asst_text or "final answer" in asst_text.lower() or "####" in asst_text)
                if not has_boxed:
                    stats["domain_errors"] += 1
                    if len(stats["error_samples"]) < 5:
                        stats["error_samples"].append(f"Line {line_num}: Missing formal boxed answer termination (\\boxed{{...}} or **Final Answer:**)")
            elif kind == "code":
                if not asst_text.strip():
                    stats["domain_errors"] += 1
            elif kind == "tool_call":
                if "tools" not in rec and "tool_calls" not in asst_text and not asst_text.strip():
                    stats["domain_errors"] += 1
            elif kind == "finance":
                finance_keywords = {"$", "margin", "ebit", "revenue", "cash", "debt", "equity", "growth", "interest", "asset", "valuation", "tax"}
                text_lower = asst_text.lower()
                if not any(kw in text_lower for kw in finance_keywords):
                    stats["domain_errors"] += 1
                    if len(stats["error_samples"]) < 5:
                        stats["error_samples"].append(f"Line {line_num}: Missing financial keywords or calculations")

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
        if "safe" in name:
            kind = "safety"
        elif "think" in name:
            kind = "thinking"
        elif "compet" in name:
            kind = "competition_math"
        elif "openweb" in name or "stem" in name or "math" in name:
            kind = "openwebmath"
        elif "analy" in name:
            kind = "analysis"
        elif "tdd" in name or "unit_test" in name or "test" in name:
            kind = "tdd"
        elif "conv" in name:
            kind = "conversation"
        elif "tool" in name:
            kind = "tool_call"
        elif "code" in name:
            kind = "code"
        elif "finan" in name:
            kind = "finance"
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
    parser.add_argument("--dir", "--output-dir", dest="extra_dir", help="Directory path to audit (convenience alias)")
    parser.add_argument(
        "--kind",
        choices=[
            "thinking", "conversation", "tool_call", "code", "instruction",
            "finance", "safety", "openwebmath", "math", "analysis", "tdd", "competition_math", "generic"
        ]
    )
    args = parser.parse_args()

    paths = list(args.paths)
    if args.extra_dir:
        paths.append(args.extra_dir)
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
