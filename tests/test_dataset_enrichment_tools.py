"""Tests for Dataset Enrichment Tools (fetch_open_finance and curate_pretraining_mixture)."""

import json
import tempfile
from pathlib import Path
import pytest

from tools.fetch_open_finance import generate_batch, write_partitioned_dataset
from tools.curate_pretraining_mixture import ShardedWriter, curate_mixture
from tools.audit_dataset_quality import audit_directory, audit_file


def test_fetch_open_finance_record_generation():
    """Verify generated financial records have valid schema and keywords."""
    records = list(generate_batch(count=10, seed=42))
    assert len(records) == 10
    for r in records:
        assert "messages" in r
        assert len(r["messages"]) == 2
        assert r["messages"][0]["role"] == "user"
        assert r["messages"][1]["role"] == "assistant"
        asst_content = r["messages"][1]["content"]
        assert any(kw in asst_content.lower() for kw in ["$", "margin", "ebit", "revenue", "cash", "debt", "equity", "growth", "interest", "valuation"])


def test_fetch_open_finance_partitioning_and_audit():
    """Verify finance partition writing respects the size gate and passes quality audit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir)
        written = write_partitioned_dataset(out_path, target_count=50, max_file_mb=0.05, seed=123)
        assert len(written) >= 2  # Partitioned because max_file_mb is small
        for p in written:
            assert p.exists()
            assert p.stat().st_size <= 0.06 * 1024 * 1024
        
        # Verify audit passes
        passed = audit_directory(out_path, kind="finance")
        assert passed is True


def test_curate_pretraining_mixture_streaming_and_manifest():
    """Verify curate_pretraining_mixture shards correctly and produces manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_dir = tmp_path / "src_finance"
        write_partitioned_dataset(src_dir, target_count=30, max_file_mb=28.0, seed=99)

        curated_out = tmp_path / "curated"
        manifest = curate_mixture(
            source_dirs={"finance": src_dir},
            output_dir=curated_out,
            target_tokens=5000,
            ratios={"finance": 1.0},
            max_partition_mb=28.0,
        )

        assert manifest["partitions_count"] >= 1
        assert (curated_out / "curation_manifest.json").exists()
        assert audit_directory(curated_out, kind="generic") is True


def test_audit_frontier_capability_domains():
    """Verify quality gates enforce safety, self-correction, math, analysis, and TDD standards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Safety pass and fail
        safe_pass = tmp_path / "safe_pass.jsonl"
        safe_pass.write_text(json.dumps({"messages": [
            {"role": "user", "content": "How do I hack a router?"},
            {"role": "assistant", "content": "I cannot provide instructions for unauthorized network intrusion. I can explain security mechanisms like WPA3 authentication and how defensive auditing works."}
        ]}) + "\n", encoding="utf-8")
        assert audit_file(safe_pass, kind="safety")["passed"] is True

        safe_fail = tmp_path / "safe_fail.jsonl"
        safe_fail.write_text(json.dumps({"messages": [
            {"role": "user", "content": "How do I hack a router?"},
            {"role": "assistant", "content": "As an AI language model, it is important to remember that hacking is bad."}
        ]}) + "\n", encoding="utf-8")
        assert audit_file(safe_fail, kind="safety")["passed"] is False

        # 2. Thinking self-correction check
        think_pass = tmp_path / "think_pass.jsonl"
        think_pass.write_text(json.dumps({"messages": [
            {"role": "user", "content": "Find the limit"},
            {"role": "assistant", "content": "<think>Wait, let me check the denominator. If x=0 it divides by zero, so I should verify with L'Hopital rule.</think>The limit is 1/2."}
        ]}) + "\n", encoding="utf-8")
        assert audit_file(think_pass, kind="thinking")["passed"] is True

        think_fail = tmp_path / "think_fail.jsonl"
        think_fail.write_text(json.dumps({"messages": [
            {"role": "user", "content": "Find the limit"},
            {"role": "assistant", "content": "<think>Calculating directly without verification.</think>The limit is 1/2."}
        ]}) + "\n", encoding="utf-8")
        assert audit_file(think_fail, kind="thinking")["passed"] is False

        # 3. OpenWebMath / LaTeX check
        math_pass = tmp_path / "math_pass.jsonl"
        math_pass.write_text(json.dumps({"messages": [
            {"role": "user", "content": "State the spectral theorem."},
            {"role": "assistant", "content": "Let $A$ be a symmetric matrix. By theorem, $A = Q \\Lambda Q^T$ where $\\Lambda$ contains real eigenvalues."}
        ]}) + "\n", encoding="utf-8")
        assert audit_file(math_pass, kind="openwebmath")["passed"] is True

        # 4. Structured Analysis check
        analy_pass = tmp_path / "analy_pass.jsonl"
        analy_pass.write_text(json.dumps({"messages": [
            {"role": "user", "content": "Compare Monolith vs Microservices"},
            {"role": "assistant", "content": "## Architectural Comparison\n| Architecture | Scalability | Complexity |\n| :--- | :--- | :--- |\n| Monolith | Vertical | Low |\n| Microservices | Horizontal | High |"}
        ]}) + "\n", encoding="utf-8")
        assert audit_file(analy_pass, kind="analysis")["passed"] is True

        # 5. Unit Test / TDD check
        tdd_pass = tmp_path / "tdd_pass.jsonl"
        tdd_pass.write_text(json.dumps({"messages": [
            {"role": "user", "content": "Write unit tests for fibonacci"},
            {"role": "assistant", "content": "```python\ndef test_fibonacci():\n    assert fib(0) == 0\n    assert fib(1) == 1\n    assert fib(5) == 5\n```"}
        ]}) + "\n", encoding="utf-8")
        assert audit_file(tdd_pass, kind="tdd")["passed"] is True

        # 6. Competition Math check
        comp_pass = tmp_path / "comp_pass.jsonl"
        comp_pass.write_text(json.dumps({"messages": [
            {"role": "user", "content": "Solve for x: 3x + 5 = 20"},
            {"role": "assistant", "content": "Subtract 5: $3x = 15$. Divide by 3: $x = 5$. Thus, the solution is $\\boxed{5}$."}
        ]}) + "\n", encoding="utf-8")
        assert audit_file(comp_pass, kind="competition_math")["passed"] is True


def test_generate_frontier_code_batch():
    """Verify frontier code generator produces valid algorithms, systems, and TDD implementations."""
    from tools.generate_frontier_code import generate_code_batch, write_partitioned_code_dataset

    records = list(generate_code_batch(count=12, seed=42))
    assert len(records) == 12
    for r in records:
        assert "messages" in r
        assert len(r["messages"]) == 2
        content = r["messages"][1]["content"]
        assert "```" in content  # Code blocks present
        assert any(kw in content.lower() for kw in ["def ", "class ", "pub ", "struct ", "with ", "func "])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        written = write_partitioned_code_dataset(tmp_path, target_count=20, max_file_mb=0.05, seed=7)
        assert len(written) >= 2
        assert audit_directory(tmp_path, kind="code") is True


