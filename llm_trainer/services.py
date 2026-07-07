from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import numpy as np
import PyPDF2

from .config import DatasetConfig, ModelConfig, TrainingConfig, dataclass_to_jsonable
from .conversation_datasets import CONVERSATION_DATASET_PRESETS, dataset_ids_for_stage, load_conversation_documents
from .data import (
    Document,
    SUPPORTED_CODE_SUFFIXES,
    SUPPORTED_TEXT_SUFFIXES,
    document_from_dict,
    document_to_dict,
    expand_code_documents,
    file_sha256,
    load_structured_json_documents,
    read_supported_document,
    supported_source_paths,
    write_training_corpus,
)
from .lineage import read_json, record_dataset_version, stable_json_hash, utc_timestamp, write_json
from .tokenizer import (
    MAX_TOKENIZER_TRAINING_CHARS,
    PAD_TOKEN,
    encode_file_to_bin,
    load_token_memmap,
    load_tokenizer,
    token_dtype_for_vocab,
    token_id,
    train_tokenizer,
    validate_training_tokenizer,
)
from .training import TrainingResult, latest_checkpoint, write_split_token_bins, train_model


LOGGER = logging.getLogger(__name__)


def _missing_dataset_artifacts(dataset_dir: Path, require_summary: bool = False) -> list[str]:
    """Return names of required prepared-dataset artifacts that are missing.

    Accepts either the current memory-mapped ``.bin`` token format or the
    legacy ``.json`` format (from datasets prepared before that switch), so
    older prepared projects don't suddenly look "unprepared" right after an
    upgrade.

    Args:
        dataset_dir: Dataset folder.
        require_summary: Whether ``dataset_summary.json`` also counts as a
            required artifact.

    Returns:
        Names of missing required files/formats. Empty if everything needed
        is present.
    """

    missing: list[str] = []
    if not (dataset_dir / "tokenizer.json").exists():
        missing.append("tokenizer.json")
    if require_summary and not (dataset_dir / "dataset_summary.json").exists():
        missing.append("dataset_summary.json")
    for stem in ("train_tokens", "val_tokens"):
        if not (dataset_dir / f"{stem}.bin").exists() and not (dataset_dir / f"{stem}.json").exists():
            missing.append(f"{stem}.bin")
    return missing


def _local_structured_dataset_paths(config: DatasetConfig) -> list[tuple[Path, str, str]]:
    """Return configured local structured dataset paths.

    Args:
        config: Dataset configuration.

    Returns:
        Tuples of path, document kind, and progress label.
    """

    items: list[tuple[Path, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in [config.conversation_dataset_path, *config.conversation_dataset_paths]:
        if path is None or not str(path).strip():
            continue
        key = ("conversation", str(Path(path)))
        if key not in seen:
            seen.add(key)
            items.append((Path(path), "conversation", "local conversation"))
    for path in [config.instruction_dataset_path, *config.instruction_dataset_paths]:
        if path is None or not str(path).strip():
            continue
        key = ("instruction", str(Path(path)))
        if key not in seen:
            seen.add(key)
            items.append((Path(path), "instruction", "local instruction"))
    return items


@dataclass
class DatasetBuildResult:
    """Result returned after dataset preparation.

    Attributes:
        output_dir: Prepared dataset folder.
        tokenizer_path: Path to tokenizer JSON.
        document_count: Number of loaded samples.
        token_count: Total encoded tokens.
        train_window_count: Number of sliding training windows.
        val_window_count: Number of sliding validation windows.
        sequence_token_stats: Approximate min/avg/median/max source token lengths.
        vocab_size: Final tokenizer vocabulary size.
        character_count: Total corpus characters.
        suggested_vocab_size: Automatically estimated vocabulary size.
        warning: Optional dataset quality warning.
        code_sample_count: Number of code samples.
        prose_sample_count: Number of prose samples.
        conversation_sample_count: Number of conversation/instruction samples.
        cached_file_count: Number of unchanged source files reused from cache.
        processed_file_count: Number of source files extracted this run.
        skipped_file_count: Number of files with no readable text.
        failed_file_count: Number of files that failed extraction.
        dataset_version_id: Unique dataset version identifier.
        dataset_version_number: One-based dataset version number.
        mixture_report: Per-source family sampling report.
        quality_score: Dataset quality score from 0 to 100.
        quality_stars: Dataset quality rating from 0 to 5.
        quality_label: Human-readable dataset quality label.
        quality_reasons: Short reasons behind the quality score.
        duplicate_block_count: Number of repeated blocks in the written corpus.
        unique_block_count: Number of unique blocks in the written corpus.
        corpus_block_count: Number of non-empty blocks inspected in the written corpus.
        duplicate_block_ratio: Fraction of repeated text blocks in the written corpus.
        unique_block_ratio: Fraction of unique text blocks in the written corpus.
    """

    output_dir: Path
    tokenizer_path: Path
    document_count: int
    token_count: int
    vocab_size: int
    character_count: int
    suggested_vocab_size: int
    train_window_count: int = 0
    val_window_count: int = 0
    sequence_token_stats: dict[str, float] = field(default_factory=dict)
    warning: Optional[str] = None
    code_sample_count: int = 0
    prose_sample_count: int = 0
    conversation_sample_count: int = 0
    cached_file_count: int = 0
    processed_file_count: int = 0
    skipped_file_count: int = 0
    failed_file_count: int = 0
    dataset_version_id: str = ""
    dataset_version_number: int = 0
    mixture_report: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    quality_stars: float = 0.0
    quality_label: str = "Not rated"
    quality_reasons: list[str] = field(default_factory=list)
    duplicate_block_count: int = 0
    unique_block_count: int = 0
    corpus_block_count: int = 0
    duplicate_block_ratio: float = 0.0
    unique_block_ratio: float = 1.0


@dataclass
class ProjectHealthResult:
    """Result returned by the project health checker.

    Attributes:
        status: Overall status: ok, warning, or error.
        checks: Individual check dictionaries.
        summary: Human-readable summary.
    """

    status: str
    checks: list[dict[str, str]]
    summary: str


@dataclass
class DatasetPreviewResult:
    """Result returned by the dataset preview scanner.

    Attributes:
        source_file_count: Number of supported source files.
        prepared: Whether prepared dataset artifacts exist.
        total_bytes: Total bytes across supported source files.
        suffix_counts: Supported file counts by suffix.
        sample_previews: Small text/code samples for inspection.
        issues: Quality issues or recommendations.
        summary: Existing dataset summary when available.
        duplicate_count: Number of files involved in likely duplicate groups.
        duplicate_groups: Duplicate group summaries.
        bad_extraction_count: Number of files with suspicious extraction quality.
        bad_extraction_files: Suspicious extraction summaries.
        code_preview_count: Code-like samples seen during preview.
        prose_preview_count: Prose samples seen during preview.
        balance_label: Human-readable code/prose balance.
        readiness_score: Training readiness score from 0 to 100.
        readiness_label: Human-readable readiness label.
        readiness_reasons: Reasons behind the readiness score.
    """

    source_file_count: int
    prepared: bool
    total_bytes: int
    suffix_counts: dict[str, int]
    sample_previews: list[dict[str, str]]
    issues: list[str]
    summary: dict[str, Any]
    duplicate_count: int = 0
    duplicate_groups: list[dict[str, Any]] = field(default_factory=list)
    bad_extraction_count: int = 0
    bad_extraction_files: list[dict[str, str]] = field(default_factory=list)
    code_preview_count: int = 0
    prose_preview_count: int = 0
    balance_label: str = "Unknown"
    readiness_score: int = 0
    readiness_label: str = "Unknown"
    readiness_reasons: list[str] = field(default_factory=list)


def _emit(progress: Optional[Callable[[Any], None]], message: str, percent: Optional[int] = None) -> None:
    """Emit a progress event if a callback is available.

    Args:
        progress: Optional callback for progress dictionaries.
        message: Human-readable progress message.
        percent: Optional progress percentage.
    """

    LOGGER.info(message)
    if progress:
        progress({"message": message, "percent": percent})


def _health_check(name: str, status: str, detail: str) -> dict[str, str]:
    """Create a project health check row.

    Args:
        name: Check label.
        status: ok, warning, or error.
        detail: Human-readable detail.

    Returns:
        Check dictionary.
    """

    return {"name": name, "status": status, "detail": detail}


def check_project_health(
    input_dir: Path,
    dataset_dir: Path,
    model_dir: Path,
    export_dir: Path,
    gguf_path: Optional[Path],
    llama_cpp_dir: Optional[Path],
    training_device: str,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> ProjectHealthResult:
    """Validate paths, artifacts, and hardware readiness for the current project.

    Args:
        input_dir: Source data folder.
        dataset_dir: Prepared dataset folder.
        model_dir: Training output folder.
        export_dir: Export output folder.
        gguf_path: Optional GGUF model selected for chat.
        llama_cpp_dir: Optional llama.cpp checkout folder.
        training_device: Selected training device.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when the user requested stop.

    Returns:
        Project health result.
    """

    checks: list[dict[str, str]] = []
    _emit(progress, "Checking project paths...", 10)
    if should_stop and should_stop():
        raise RuntimeError("Project health check stopped by user.")

    if input_dir.exists() and input_dir.is_dir():
        try:
            source_count = len(supported_source_paths(input_dir, code_training_mode=True, include_source_code=True))
            if source_count:
                checks.append(_health_check("Source vault", "ok", f"{source_count:,} supported file(s) found."))
            else:
                checks.append(_health_check("Source vault", "warning", "Folder exists, but no supported files were found."))
        except Exception as exc:
            checks.append(_health_check("Source vault", "error", str(exc)))
    else:
        checks.append(_health_check("Source vault", "warning", f"Folder not found: {input_dir}"))

    _emit(progress, "Checking prepared dataset artifacts...", 30)
    missing_dataset = _missing_dataset_artifacts(dataset_dir, require_summary=True)
    if not dataset_dir.exists():
        checks.append(_health_check("Dataset core", "warning", f"Dataset folder not found yet: {dataset_dir}"))
    elif missing_dataset:
        checks.append(_health_check("Dataset core", "warning", f"Missing prepared artifact(s): {', '.join(missing_dataset)}"))
    else:
        summary = read_json(dataset_dir / "dataset_summary.json", default={}) or {}
        tokens = int(summary.get("token_count", 0) or 0)
        vocab = int(summary.get("tokenizer_vocab_size", 0) or 0)
        train_windows = int(summary.get("train_window_count", 0) or 0)
        val_windows = int(summary.get("val_window_count", 0) or 0)
        window_text = f", {train_windows:,}/{val_windows:,} train/val window(s)" if train_windows or val_windows else ""
        checks.append(_health_check("Dataset core", "ok", f"Prepared with {tokens:,} token(s), vocab {vocab:,}{window_text}."))

    _emit(progress, "Checking model artifacts...", 50)
    if not model_dir.exists():
        checks.append(_health_check("Model output", "warning", f"Model folder not found yet: {model_dir}"))
    elif (model_dir / "final_model.pt").exists():
        checks.append(_health_check("Model output", "ok", "final_model.pt found."))
    elif (model_dir / "checkpoints").exists() and any((model_dir / "checkpoints").glob("*.pt")):
        checks.append(_health_check("Model output", "warning", "Checkpoints found, but final_model.pt is not present yet."))
    else:
        checks.append(_health_check("Model output", "warning", "No trained model or checkpoint found yet."))

    _emit(progress, "Checking export and GGUF paths...", 65)
    if export_dir.exists():
        artifact_count = sum(1 for item in export_dir.iterdir())
        checks.append(_health_check("Export bay", "ok" if artifact_count else "warning", f"{artifact_count:,} item(s) in export folder."))
    else:
        checks.append(_health_check("Export bay", "warning", f"Export folder not found yet: {export_dir}"))

    if gguf_path and str(gguf_path).strip():
        if gguf_path.exists():
            checks.append(_health_check("GGUF chat model", "ok", f"Found {gguf_path.name}."))
        else:
            checks.append(_health_check("GGUF chat model", "warning", f"GGUF file not found: {gguf_path}"))
    else:
        checks.append(_health_check("GGUF chat model", "warning", "No GGUF model selected for chat."))

    if llama_cpp_dir and str(llama_cpp_dir).strip():
        converter = llama_cpp_dir / "convert_hf_to_gguf.py"
        checks.append(
            _health_check(
                "llama.cpp",
                "ok" if converter.exists() else "warning",
                "convert_hf_to_gguf.py found." if converter.exists() else f"Converter not found in {llama_cpp_dir}.",
            )
        )

    _emit(progress, "Checking hardware/runtime...", 85)
    if training_device == "cuda":
        if torch.cuda.is_available():
            checks.append(_health_check("Training device", "ok", f"CUDA ready: {torch.cuda.get_device_name(0)}."))
        else:
            checks.append(_health_check("Training device", "error", "CUDA selected but PyTorch cannot use CUDA."))
    else:
        checks.append(_health_check("Training device", "ok", "CPU selected. Training will be slower but compatible."))

    statuses = [check["status"] for check in checks]
    if "error" in statuses:
        status = "error"
    elif "warning" in statuses:
        status = "warning"
    else:
        status = "ok"
    summary = f"{sum(1 for item in statuses if item == 'ok')} ok, {sum(1 for item in statuses if item == 'warning')} warning, {sum(1 for item in statuses if item == 'error')} error."
    _emit(progress, f"Project health check complete: {summary}", 100)
    return ProjectHealthResult(status=status, checks=checks, summary=summary)


def _supported_source_paths_cancellable(
    input_dir: Path,
    code_training_mode: bool,
    include_source_code: bool,
    should_stop: Optional[Callable[[], bool]],
) -> list[Path]:
    """Return supported source paths while honoring cancellation.

    Args:
        input_dir: Folder to scan.
        code_training_mode: Whether source-code files are supported.
        include_source_code: Whether to include source-code files.
        should_stop: Optional cancellation callback.

    Returns:
        Sorted supported source paths.
    """

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    allowed = set(SUPPORTED_TEXT_SUFFIXES) | {".pdf", ".jsonl"}
    if code_training_mode and include_source_code:
        allowed |= set(SUPPORTED_CODE_SUFFIXES)
    paths: list[Path] = []
    for root, dirs, files in os.walk(input_dir):
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        dirs[:] = [
            name
            for name in dirs
            if name not in {".git", "__pycache__", ".venv", "venv", "node_modules"}
        ]
        for filename in files:
            if should_stop and should_stop():
                raise RuntimeError("Dataset preview stopped by user.")
            path = Path(root) / filename
            if path.suffix.lower() in allowed:
                paths.append(path)
    return sorted(paths)


def _preview_supported_document(
    path: Path,
    config: DatasetConfig,
    should_stop: Optional[Callable[[], bool]],
    max_chars: int = 1200,
) -> Optional[dict[str, str]]:
    """Read a small preview from one source document.

    Args:
        path: Source path.
        config: Dataset options.
        should_stop: Optional cancellation callback.
        max_chars: Maximum preview characters.

    Returns:
        Preview dictionary, or ``None`` when the file has no readable text.
    """

    if should_stop and should_stop():
        raise RuntimeError("Dataset preview stopped by user.")
    suffix = path.suffix.lower()
    kind = "prose"
    language = ""
    text = ""
    if config.code_training_mode and suffix in SUPPORTED_CODE_SUFFIXES:
        kind = "code"
        language = SUPPORTED_CODE_SUFFIXES[suffix]
        with path.open("rb") as file:
            text = file.read(128 * 1024).decode("utf-8", errors="ignore")
        text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    elif suffix in SUPPORTED_TEXT_SUFFIXES:
        with path.open("rb") as file:
            text = file.read(128 * 1024).decode("utf-8", errors="ignore")
        text = re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()
    elif suffix == ".jsonl":
        chunks: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for index, line in enumerate(file):
                if should_stop and should_stop():
                    raise RuntimeError("Dataset preview stopped by user.")
                if index >= 40 or sum(len(chunk) for chunk in chunks) >= max_chars:
                    break
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, str):
                    chunks.append(value)
                elif isinstance(value, dict):
                    for key in ("text", "content", "prompt", "completion"):
                        if value.get(key):
                            chunks.append(str(value[key]))
                            break
        text = "\n".join(chunks).strip()
    elif suffix == ".pdf":
        chunks: list[str] = []
        with path.open("rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages[:3]:
                if should_stop and should_stop():
                    raise RuntimeError("Dataset preview stopped by user.")
                chunks.append(page.extract_text() or "")
                if sum(len(chunk) for chunk in chunks) >= max_chars:
                    break
        text = re.sub(r"\s+", " ", "\n".join(chunks).replace("\x00", " ")).strip()
    if config.lowercase:
        text = text.lower()
    if not text:
        return None
    return {
        "path": str(path),
        "kind": kind,
        "language": language,
        "characters": str(len(text)),
        "preview": text[:max_chars],
    }


def _file_sha256_cancellable(path: Path, should_stop: Optional[Callable[[], bool]]) -> str:
    """Calculate a file digest while honoring cancellation between chunks.

    Args:
        path: File path.
        should_stop: Optional cancellation callback.

    Returns:
        File SHA-256 hex digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            if should_stop and should_stop():
                raise RuntimeError("Dataset preview stopped by user.")
            digest.update(chunk)
    return digest.hexdigest()


def _preview_fingerprint(text: str) -> str:
    """Return a normalized content fingerprint for duplicate detection.

    Args:
        text: Preview text.

    Returns:
        SHA-1 fingerprint.
    """

    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    normalized = re.sub(r"\d+", "0", normalized)
    return hashlib.sha1(normalized[:4000].encode("utf-8", errors="ignore")).hexdigest()


def _bad_extraction_reasons(path: Path, preview: Optional[dict[str, str]], size: int) -> list[str]:
    """Return quality reasons when extracted preview text looks suspicious.

    Args:
        path: Source path.
        preview: Preview dictionary, or ``None``.
        size: Source file size in bytes.

    Returns:
        List of quality concerns.
    """

    suffix = path.suffix.lower()
    if preview is None:
        return ["no readable preview text"]
    text = str(preview.get("preview", ""))
    if not text.strip():
        return ["empty preview text"]
    reasons: list[str] = []
    visible = [char for char in text if not char.isspace()]
    if len(text) < 80 and suffix == ".pdf":
        reasons.append("very little text extracted from PDF preview")
    if size > 250_000 and suffix == ".pdf" and len(text) < 200:
        reasons.append("large PDF produced very little readable text")
    if visible:
        alpha_ratio = sum(char.isalpha() for char in visible) / len(visible)
        symbol_ratio = sum(not char.isalnum() for char in visible) / len(visible)
        if alpha_ratio < 0.25 and str(preview.get("kind")) != "code":
            reasons.append("low alphabetic text ratio")
        if symbol_ratio > 0.45:
            reasons.append("high symbol/noise ratio")
    if re.search(r"(.)\1{18,}", text):
        reasons.append("long repeated character run")
    if text.count("\ufffd") >= 3 or "Ã" in text[:500]:
        reasons.append("encoding artifacts detected")
    words = re.findall(r"[A-Za-z]{2,}", text)
    if suffix in {".pdf", ".txt", ".md", ".text"} and len(set(words)) < 8 and len(text) > 200:
        reasons.append("very low word variety")
    return reasons


def _balance_label(code_count: int, prose_count: int, code_training_mode: bool) -> str:
    """Return a readable code/prose balance label.

    Args:
        code_count: Number of code samples.
        prose_count: Number of prose samples.
        code_training_mode: Whether code-oriented preparation is enabled.

    Returns:
        Balance label.
    """

    total = code_count + prose_count
    if total <= 0:
        return "Unknown"
    code_ratio = code_count / total
    if not code_training_mode:
        return "Prose focused"
    if code_ratio < 0.2:
        return "Prose heavy"
    if code_ratio > 0.8:
        return "Code heavy"
    return "Balanced code/prose"


def _readiness_report(
    source_file_count: int,
    total_bytes: int,
    prepared: bool,
    summary: dict[str, Any],
    duplicate_count: int,
    bad_extraction_count: int,
    code_count: int,
    prose_count: int,
    code_training_mode: bool,
) -> tuple[int, str, list[str]]:
    """Calculate an explainable training readiness score.

    Args:
        source_file_count: Number of supported source files.
        total_bytes: Total supported source bytes.
        prepared: Whether token artifacts exist.
        summary: Prepared dataset summary when available.
        duplicate_count: Files involved in likely duplicates.
        bad_extraction_count: Files with suspicious extraction.
        code_count: Code sample count.
        prose_count: Prose sample count.
        code_training_mode: Whether user is preparing a code-oriented dataset.

    Returns:
        Score, label, and explanatory reasons.
    """

    score = 100
    reasons: list[str] = []
    token_count = int(summary.get("token_count", 0) or 0)
    if prepared and token_count:
        if token_count < 50_000:
            score -= 35
            reasons.append("Prepared token count is very small.")
        elif token_count < 250_000:
            score -= 18
            reasons.append("Prepared token count is modest.")
        else:
            reasons.append("Prepared token count looks usable.")
    else:
        score -= 15
        reasons.append("Dataset is not prepared yet; score is based on source preview.")
        if total_bytes < 250_000:
            score -= 25
            reasons.append("Source size is small for meaningful training.")
        elif total_bytes < 2_000_000:
            score -= 10
            reasons.append("Source size is modest.")

    if source_file_count == 0:
        score -= 60
        reasons.append("No supported source files were found.")
    duplicate_ratio = duplicate_count / max(source_file_count, 1)
    if duplicate_ratio >= 0.25:
        score -= 25
        reasons.append("Duplicate ratio is high.")
    elif duplicate_ratio >= 0.1:
        score -= 12
        reasons.append("Some likely duplicate files were found.")

    bad_ratio = bad_extraction_count / max(source_file_count, 1)
    if bad_ratio >= 0.2:
        score -= 25
        reasons.append("Many files have suspicious extraction quality.")
    elif bad_ratio > 0:
        score -= 10
        reasons.append("Some files have suspicious extraction quality.")

    total_samples = code_count + prose_count
    if code_training_mode and total_samples:
        code_ratio = code_count / total_samples
        if code_ratio < 0.1:
            score -= 12
            reasons.append("Code training mode is enabled but very little code was detected.")
        elif code_ratio > 0.95 and prose_count == 0:
            score -= 5
            reasons.append("Dataset is almost entirely code; explanations may be weak.")

    score = max(0, min(100, score))
    if score >= 80:
        label = "Ready"
    elif score >= 60:
        label = "Usable with warnings"
    elif score >= 35:
        label = "Needs cleanup"
    else:
        label = "Not ready"
    return score, label, reasons


def scan_dataset_preview(
    config: DatasetConfig,
    sample_limit: int = 8,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> DatasetPreviewResult:
    """Scan source/prepared dataset quality without writing training artifacts.

    Args:
        config: Dataset preparation settings.
        sample_limit: Maximum readable source samples to preview.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when the user requested stop.

    Returns:
        Dataset preview result.
    """

    _emit(progress, "Scanning supported source files...", 5)
    local_structured_paths = _local_structured_dataset_paths(config)
    if config.input_dir.exists():
        paths = _supported_source_paths_cancellable(
            config.input_dir,
            config.code_training_mode,
            config.include_source_code,
            should_stop,
        )
    elif config.conversation_datasets or local_structured_paths:
        paths = []
    else:
        paths = _supported_source_paths_cancellable(
            config.input_dir,
            config.code_training_mode,
            config.include_source_code,
            should_stop,
        )
    suffix_counts: dict[str, int] = {}
    total_bytes = 0
    for path in paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        suffix_counts[path.suffix.lower() or "<none>"] = suffix_counts.get(path.suffix.lower() or "<none>", 0) + 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
    for path, _, _ in local_structured_paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        path = Path(path)
        if path.exists() and path.is_file():
            suffix_counts[path.suffix.lower() or "<none>"] = suffix_counts.get(path.suffix.lower() or "<none>", 0) + 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass

    summary = read_json(config.output_dir / "dataset_summary.json", default={}) or {}
    prepared = not _missing_dataset_artifacts(config.output_dir)
    issues: list[str] = []
    if not paths and not config.conversation_datasets and not local_structured_paths:
        issues.append("No supported source files found.")
    if config.conversation_dataset_path:
        issues.append(f"Local conversation JSON selected: {config.conversation_dataset_path}.")
    if config.instruction_dataset_path:
        issues.append(f"Local instruction JSON selected: {config.instruction_dataset_path}.")
    if config.conversation_datasets:
        labels = [
            CONVERSATION_DATASET_PRESETS[item].label
            for item in config.conversation_datasets
            if item in CONVERSATION_DATASET_PRESETS
        ]
        issues.append(f"Conversation datasets selected: {', '.join(labels)}.")
    if total_bytes < 100_000:
        issues.append("Source content appears small for meaningful LLM training.")
    if prepared and summary:
        token_count = int(summary.get("token_count", 0) or 0)
        if token_count < 50_000:
            issues.append("Prepared token count is low; expect smoke-test quality only.")
        if summary.get("warning"):
            issues.append(str(summary["warning"]))
    elif config.output_dir.exists():
        issues.append("Dataset folder exists but does not contain a complete prepared dataset.")

    _emit(progress, "Reading a few preview samples...", 30)
    sample_previews: list[dict[str, str]] = []
    all_previews: list[dict[str, str]] = []
    bad_extraction_files: list[dict[str, str]] = []
    content_fingerprints: dict[str, list[str]] = {}
    readable = 0
    scan_limit = min(len(paths), max(sample_limit * 8, 80))
    for index, path in enumerate(paths[:scan_limit], start=1):
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        try:
            preview = _preview_supported_document(path, config, should_stop)
        except RuntimeError:
            raise
        except Exception as exc:
            issues.append(f"Could not preview {path.name}: {exc}")
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        reasons = _bad_extraction_reasons(path, preview, size)
        if reasons:
            bad_extraction_files.append(
                {
                    "path": str(path),
                    "reasons": "; ".join(reasons),
                    "size": str(size),
                }
            )
        if preview is None:
            issues.append(f"{path.name} has no readable text.")
            continue
        readable += 1
        all_previews.append(preview)
        preview_text = preview.get("preview", "")
        if len(preview_text) >= 120:
            content_fingerprints.setdefault(_preview_fingerprint(preview_text), []).append(str(path))
        if len(sample_previews) < sample_limit:
            sample_previews.append(preview)
            percent = 30 + int(45 * len(sample_previews) / max(sample_limit, 1))
            _emit(progress, f"Previewed {path.name}.", percent)

    for local_path, kind, _ in local_structured_paths:
        if len(sample_previews) >= sample_limit:
            continue
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        try:
            local_documents = load_structured_json_documents(Path(local_path), kind=kind, lowercase=config.lowercase)
        except Exception as exc:
            issues.append(f"Could not preview {kind} JSON dataset: {exc}")
            continue
        for document in local_documents[: max(1, sample_limit - len(sample_previews))]:
            preview = {
                "path": str(document.path),
                "kind": document.kind,
                "language": document.language or "",
                "characters": str(len(document.text)),
                "preview": document.text[:1200],
            }
            all_previews.append(preview)
            sample_previews.append(preview)
            _emit(progress, f"Previewed {kind} JSON sample.", 75)
            if len(sample_previews) >= sample_limit:
                break

    if readable == 0 and paths:
        issues.append("Supported files were found, but none produced readable preview text.")
    _emit(progress, "Checking duplicate files and repeated extracted text...", 85)
    size_groups: dict[int, list[Path]] = {}
    for path in paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        try:
            size_groups.setdefault(path.stat().st_size, []).append(path)
        except OSError:
            continue
    exact_hashes: dict[str, list[str]] = {}
    for same_size_paths in size_groups.values():
        if len(same_size_paths) < 2:
            continue
        for path in same_size_paths:
            digest = _file_sha256_cancellable(path, should_stop)
            exact_hashes.setdefault(digest, []).append(str(path))
    duplicate_groups: list[dict[str, Any]] = []
    for group in exact_hashes.values():
        if len(group) > 1:
            duplicate_groups.append({"type": "exact file", "count": len(group), "files": group[:8]})
    for group in content_fingerprints.values():
        unique_group = sorted(set(group))
        if len(unique_group) > 1:
            duplicate_groups.append({"type": "similar extracted text", "count": len(unique_group), "files": unique_group[:8]})
    duplicate_files = {
        file_path
        for group in duplicate_groups
        for file_path in group.get("files", [])
    }
    duplicate_count = len(duplicate_files)
    if duplicate_groups:
        issues.append(f"Found {len(duplicate_groups)} likely duplicate group(s) involving {duplicate_count} file entries.")
    if bad_extraction_files:
        issues.append(f"Found {len(bad_extraction_files)} file(s) with suspicious extraction quality.")
    summary_code_count = int(summary.get("code_sample_count", 0) or 0)
    summary_prose_count = int(summary.get("prose_sample_count", 0) or 0)
    code_preview_count = summary_code_count or sum(1 for preview in all_previews if preview.get("kind") == "code")
    prose_preview_count = summary_prose_count or sum(1 for preview in all_previews if preview.get("kind") != "code")
    balance = _balance_label(code_preview_count, prose_preview_count, config.code_training_mode)
    effective_source_count = len(paths) + len(config.conversation_datasets) + len(local_structured_paths)
    readiness_score, readiness_label, readiness_reasons = _readiness_report(
        source_file_count=effective_source_count,
        total_bytes=total_bytes,
        prepared=prepared,
        summary=summary,
        duplicate_count=duplicate_count,
        bad_extraction_count=len(bad_extraction_files),
        code_count=code_preview_count,
        prose_count=prose_preview_count,
        code_training_mode=config.code_training_mode,
    )
    _emit(progress, "Dataset preview complete.", 100)
    return DatasetPreviewResult(
        source_file_count=effective_source_count,
        prepared=prepared,
        total_bytes=total_bytes,
        suffix_counts=dict(sorted(suffix_counts.items())),
        sample_previews=sample_previews,
        issues=issues,
        summary=summary,
        duplicate_count=duplicate_count,
        duplicate_groups=duplicate_groups,
        bad_extraction_count=len(bad_extraction_files),
        bad_extraction_files=bad_extraction_files[:30],
        code_preview_count=code_preview_count,
        prose_preview_count=prose_preview_count,
        balance_label=balance,
        readiness_score=readiness_score,
        readiness_label=readiness_label,
        readiness_reasons=readiness_reasons,
    )


def estimate_vocab_size(character_count: int, unique_word_count: int) -> int:
    """Estimate a reasonable tokenizer vocabulary size.

    Args:
        character_count: Number of corpus characters.
        unique_word_count: Approximate number of unique whitespace words.

    Returns:
        Suggested vocabulary size.
    """

    if character_count < 20_000:
        ceiling = 1_000
    elif character_count < 100_000:
        ceiling = 4_000
    elif character_count < 500_000:
        ceiling = 8_000
    elif character_count < 2_000_000:
        ceiling = 16_000
    else:
        ceiling = 32_000

    desired = max(512, int(unique_word_count * 1.7), int(character_count / 45))
    return max(256, min(ceiling, desired))


def content_warning(character_count: int) -> Optional[str]:
    """Return a corpus-size warning when the dataset is small.

    Args:
        character_count: Number of corpus characters.

    Returns:
        Warning text, or ``None`` when the corpus is large enough.
    """

    if character_count < 10_000:
        return "The corpus is very small. Training can run, but the model will only be useful for smoke tests."
    if character_count < 100_000:
        return "The corpus is modest. Use more text for better generations and reasoning behavior."
    return None


def _resolve_tokenizer_strategy(config: DatasetConfig, tokenizer_path: Path) -> tuple[str, bool]:
    """Resolve tokenizer strategy into an executable mode.

    Args:
        config: Dataset configuration.
        tokenizer_path: Dataset tokenizer output path.

    Returns:
        Strategy name and whether the dataset tokenizer should be reused.
    """

    strategy = config.tokenizer_strategy or "auto"
    if strategy == "auto":
        return strategy, config.prepare_mode == "incremental" and tokenizer_path.exists()
    if strategy == "reuse_dataset":
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Cannot reuse dataset tokenizer because tokenizer.json was not found in {config.output_dir}."
            )
        return strategy, True
    if strategy in {"train_new", "import_tokenizer"}:
        return strategy, False
    raise ValueError(f"Unsupported tokenizer strategy: {strategy}")


def _load_or_create_tokenizer(
    config: DatasetConfig,
    corpus_path: Path,
    tokenizer_path: Path,
    selected_vocab_size: int,
    progress: Optional[Callable[[Any], None]],
    should_stop: Optional[Callable[[], bool]],
) -> tuple[Any, bool, bool, Optional[str]]:
    """Load, import, or train a tokenizer for the prepared corpus.

    Args:
        config: Dataset configuration.
        corpus_path: Normalized training corpus path.
        tokenizer_path: Dataset tokenizer output path.
        selected_vocab_size: Vocabulary size used when training a new tokenizer.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Tokenizer, reused flag, imported flag, and optional source path.
    """

    strategy, reuse_tokenizer = _resolve_tokenizer_strategy(config, tokenizer_path)
    imported = False
    source_path: Optional[str] = None

    if reuse_tokenizer:
        _emit(progress, "Reusing existing dataset tokenizer.json...", 62)
        return load_tokenizer(tokenizer_path), True, imported, source_path

    if strategy == "import_tokenizer":
        if config.tokenizer_path is None:
            raise ValueError("Choose a tokenizer.json file when tokenizer strategy is Import tokenizer.json.")
        import_path = Path(config.tokenizer_path)
        if not import_path.exists():
            raise FileNotFoundError(f"Tokenizer import file not found: {import_path}")
        _emit(progress, f"Importing tokenizer from {import_path}...", 62)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        if import_path.resolve() != tokenizer_path.resolve():
            shutil.copy2(import_path, tokenizer_path)
        return load_tokenizer(tokenizer_path), False, True, str(import_path)

    training_mb = min(corpus_path.stat().st_size, MAX_TOKENIZER_TRAINING_CHARS) / (1024 * 1024)
    _emit(
        progress,
        f"Training tokenizer on a bounded {training_mb:.1f} MB corpus sample to keep memory stable...",
        62,
    )
    tokenizer = train_tokenizer(
        corpus_path,
        tokenizer_path,
        vocab_size=selected_vocab_size,
        min_frequency=config.min_frequency,
        should_stop=should_stop,
    )
    return tokenizer, False, imported, source_path


def _cache_key(config: DatasetConfig) -> str:
    """Return a cache key for extraction-affecting options.

    Args:
        config: Dataset configuration.

    Returns:
        Cache key string.
    """

    return json.dumps(
        {
            "lowercase": config.lowercase,
            "code_training_mode": config.code_training_mode,
            "include_prose": config.include_prose,
            "include_source_code": config.include_source_code,
            "extract_code_blocks": config.extract_code_blocks,
            "preserve_indentation": config.preserve_indentation,
            "generate_instruction_samples": config.generate_instruction_samples,
            "reasoning_sample_mode": config.reasoning_sample_mode,
            "dataset_stage": config.dataset_stage,
            "conversation_datasets": config.conversation_datasets,
            "conversation_sample_limit": config.conversation_sample_limit,
            "conversation_dataset_path": str(config.conversation_dataset_path or ""),
            "instruction_dataset_path": str(config.instruction_dataset_path or ""),
            "conversation_dataset_paths": [str(path) for path in config.conversation_dataset_paths],
            "instruction_dataset_paths": [str(path) for path in config.instruction_dataset_paths],
            "default_data_paths": [str(path) for path in config.default_data_paths],
        },
        sort_keys=True,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read a dataset manifest.

    Args:
        path: Manifest path.

    Returns:
        Manifest dictionary.
    """

    if not path.exists():
        return {"version": 1, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_documents_with_cache(
    config: DatasetConfig,
    progress: Optional[Callable[[Any], None]],
    should_stop: Optional[Callable[[], bool]],
) -> tuple[list[Any], dict[str, Any], int, int, int, int]:
    """Load documents using an extraction cache.

    Args:
        config: Dataset configuration.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Documents, manifest, cached, processed, skipped, and failed file counts.
    """

    manifest_path = config.output_dir / "dataset_manifest.json"
    cache_dir = config.output_dir / "cache" / "documents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(manifest_path)
    manifest.setdefault("files", {})
    key = _cache_key(config)
    force_reprocess = config.prepare_mode == "force_reprocess"

    local_structured_paths = _local_structured_dataset_paths(config)
    selected_default_files = [
        Path(path)
        for path in config.default_data_paths
        if Path(path).exists() and Path(path).is_file()
    ]
    input_dir_resolved = config.input_dir.resolve() if config.input_dir.exists() else None
    default_files_under_input = bool(selected_default_files) and input_dir_resolved is not None and all(
        input_dir_resolved in candidate.resolve().parents or candidate.resolve() == input_dir_resolved
        for candidate in selected_default_files
    )
    if config.input_dir.exists() and not default_files_under_input:
        source_paths = supported_source_paths(
            config.input_dir,
            code_training_mode=config.code_training_mode,
            include_source_code=config.include_source_code,
        )
    elif config.conversation_datasets or local_structured_paths or config.default_data_paths:
        source_paths = []
    else:
        source_paths = supported_source_paths(
            config.input_dir,
            code_training_mode=config.code_training_mode,
            include_source_code=config.include_source_code,
        )
    default_paths = []
    seen_source_paths = {path.resolve() for path in source_paths if path.exists()}
    for candidate in selected_default_files:
        if not candidate.exists() or not candidate.is_file():
            _emit(progress, f"Skipped bundled data file: {candidate}")
            continue
        suffix = candidate.suffix.lower()
        if suffix not in SUPPORTED_TEXT_SUFFIXES and suffix not in SUPPORTED_CODE_SUFFIXES and suffix not in {".pdf", ".json", ".jsonl"}:
            _emit(progress, f"Skipped unsupported bundled data file: {candidate.name}")
            continue
        resolved = candidate.resolve()
        if resolved in seen_source_paths:
            continue
        seen_source_paths.add(resolved)
        default_paths.append(candidate)
    if default_paths:
        source_paths.extend(default_paths)
        source_paths = sorted(source_paths)
        _emit(progress, f"Bundled starter data enabled: {len(default_paths)} file(s).", 8)
    _emit(progress, f"Found {len(source_paths)} supported files in {config.input_dir}.", 8)
    documents: list[Any] = []
    cached_count = 0
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    new_files: dict[str, Any] = {}

    for index, path in enumerate(source_paths, start=1):
        if should_stop and should_stop():
            raise RuntimeError("Dataset preparation stopped by user.")
        percent = 10 + int(32 * index / max(len(source_paths), 1))
        stat = path.stat()
        digest = file_sha256(path)
        cache_path = cache_dir / f"{digest}.json"
        manifest_key = str(path.resolve())
        previous = manifest.get("files", {}).get(manifest_key, {})
        can_use_cache = (
            not force_reprocess
            and previous.get("sha256") == digest
            and previous.get("cache_key") == key
            and cache_path.exists()
        )
        if can_use_cache:
            cached_documents = [
                document_from_dict(item)
                for item in json.loads(cache_path.read_text(encoding="utf-8"))
            ]
            cached_extraction_reasons = []
            if path.suffix.lower() == ".pdf":
                cached_text = "\n".join(document.text for document in cached_documents)
                cached_extraction_reasons = _bad_extraction_reasons(
                    path,
                    {
                        "path": str(path),
                        "kind": cached_documents[0].kind if cached_documents else "prose",
                        "language": cached_documents[0].language if cached_documents else "",
                        "characters": str(len(cached_text)),
                        "preview": cached_text[:1200],
                    },
                    stat.st_size,
                )
            if cached_extraction_reasons:
                skipped_count += 1
                reason_text = "; ".join(cached_extraction_reasons)
                _emit(progress, f"Skipped cached {path.name}: suspicious PDF extraction ({reason_text}).", percent)
                new_files[manifest_key] = {
                    "path": str(path),
                    "sha256": digest,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "cache_key": key,
                    "status": "skipped_bad_extraction",
                    "reasons": cached_extraction_reasons,
                }
                continue
            documents.extend(cached_documents)
            cached_count += 1
            _emit(progress, f"Reused {path.name} from cache ({len(cached_documents)} sample(s)).", percent)
        else:
            try:
                source_doc = read_supported_document(
                    path,
                    lowercase=config.lowercase,
                    code_training_mode=config.code_training_mode,
                    preserve_indentation=config.preserve_indentation,
                )
            except Exception as exc:
                failed_count += 1
                _emit(progress, f"Failed {path.name}: {exc}", percent)
                new_files[manifest_key] = {
                    "path": str(path),
                    "sha256": digest,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "cache_key": key,
                    "status": "failed",
                    "error": str(exc),
                }
                continue
            if source_doc is None:
                skipped_count += 1
                _emit(progress, f"Skipped {path.name}: no readable text found.", percent)
                new_files[manifest_key] = {
                    "path": str(path),
                    "sha256": digest,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "cache_key": key,
                    "status": "skipped_empty",
                }
                continue
            extraction_reasons = _bad_extraction_reasons(
                path,
                {
                    "path": str(path),
                    "kind": source_doc.kind,
                    "language": source_doc.language or "",
                    "characters": str(len(source_doc.text)),
                    "preview": source_doc.text[:1200],
                },
                stat.st_size,
            )
            if path.suffix.lower() == ".pdf" and extraction_reasons:
                skipped_count += 1
                reason_text = "; ".join(extraction_reasons)
                _emit(progress, f"Skipped {path.name}: suspicious PDF extraction ({reason_text}).", percent)
                new_files[manifest_key] = {
                    "path": str(path),
                    "sha256": digest,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "cache_key": key,
                    "status": "skipped_bad_extraction",
                    "reasons": extraction_reasons,
                }
                continue
            source_documents = [source_doc]
            if config.code_training_mode:
                source_documents = expand_code_documents(
                    source_documents,
                    include_prose=config.include_prose,
                    extract_code_blocks=config.extract_code_blocks,
                    preserve_indentation=config.preserve_indentation,
                    should_stop=should_stop,
                )
            cache_path.write_text(
                json.dumps([document_to_dict(doc) for doc in source_documents], ensure_ascii=False),
                encoding="utf-8",
            )
            documents.extend(source_documents)
            processed_count += 1
            _emit(progress, f"Processed {path.name}: {len(source_documents)} sample(s).", percent)

        new_files[manifest_key] = {
            "path": str(path),
            "sha256": digest,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "cache_key": key,
            "cache_file": str(cache_path.relative_to(config.output_dir)),
            "status": "cached" if can_use_cache else "processed",
        }

    for local_path, kind, label in local_structured_paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preparation stopped by user.")
        local_path = Path(local_path)
        _emit(progress, f"Loading {label} JSON/JSONL dataset: {local_path}", 42)
        local_documents = load_structured_json_documents(local_path, kind=kind, lowercase=config.lowercase)
        documents.extend(local_documents)
        processed_count += 1
        manifest_key = f"local-{kind}://{local_path.resolve()}"
        new_files[manifest_key] = {
            "path": str(local_path),
            "kind": kind,
            "sample_count": len(local_documents),
            "cache_key": key,
            "status": "processed",
        }
        _emit(progress, f"Loaded {len(local_documents):,} {kind} sample(s) from {local_path.name}.", 43)

    if config.conversation_datasets:
        allowed_dataset_ids = set(dataset_ids_for_stage(config.dataset_stage))
        skipped_stage_ids = [dataset_id for dataset_id in config.conversation_datasets if dataset_id not in allowed_dataset_ids]
        selected_dataset_ids = [dataset_id for dataset_id in config.conversation_datasets if dataset_id in allowed_dataset_ids]
        if skipped_stage_ids:
            skipped_labels = [
                CONVERSATION_DATASET_PRESETS[item].label
                for item in skipped_stage_ids
                if item in CONVERSATION_DATASET_PRESETS
            ]
            _emit(progress, f"Skipping dataset(s) not recommended for {config.dataset_stage}: {', '.join(skipped_labels)}.")
        if not selected_dataset_ids:
            _emit(progress, f"No online datasets selected for {config.dataset_stage}; continuing with local sources only.")
            config.conversation_datasets = []
            manifest["files"] = new_files
            manifest["dataset_config"] = dataclass_to_jsonable(config)
            manifest["cache_key"] = key
            return (
                sorted(documents, key=lambda document: (str(document.path), document.kind, document.language or "")),
                manifest,
                cached_count,
                processed_count,
                skipped_count,
                failed_count,
            )
        hf_cache_dir = config.output_dir / "cache" / "huggingface"
        labels = [
            CONVERSATION_DATASET_PRESETS[item].label
            for item in selected_dataset_ids
            if item in CONVERSATION_DATASET_PRESETS
        ]
        _emit(progress, f"Online training datasets enabled: {', '.join(labels)}.", 8)
        _emit(progress, f"Online training datasets will be cached in: {hf_cache_dir}", 8)
        hf_documents = load_conversation_documents(
            selected_dataset_ids,
            config.conversation_sample_limit,
            hf_cache_dir,
            lowercase=config.lowercase,
            progress=progress,
            should_stop=should_stop,
        )
        documents.extend(hf_documents)
        config.conversation_datasets = selected_dataset_ids
        for dataset_id in selected_dataset_ids:
            preset = CONVERSATION_DATASET_PRESETS.get(dataset_id)
            new_files[f"hf://{dataset_id}"] = {
                "path": f"hf://{dataset_id}",
                "dataset": preset.hf_path if preset else dataset_id,
                "config_name": preset.config_name if preset else "",
                "split": preset.split if preset else "",
                "sample_limit": config.conversation_sample_limit,
                "cache_key": key,
                "status": "processed",
            }
        processed_count += len(selected_dataset_ids)

    manifest["files"] = new_files
    manifest["dataset_config"] = dataclass_to_jsonable(config)
    manifest["cache_key"] = key
    return (
        sorted(documents, key=lambda document: (str(document.path), document.kind, document.language or "")),
        manifest,
        cached_count,
        processed_count,
        skipped_count,
        failed_count,
    )


MIXTURE_LABELS = {
    "stories": "Stories",
    "reasoning": "Reasoning",
    "social_emotional": "Social / emotions",
    "factual_knowledge": "Facts / knowledge",
    "mathematics": "Mathematics",
    "code_technical": "Code / technical",
    "language_basics": "Language basics",
    "structured_qa": "Structured Q&A",
    "safety_uncertainty": "Safety / uncertainty",
    "general_prose": "General prose",
    "local_prose": "Local prose",
    "source_code": "Source code",
    "online_base": "Online base",
    "instruction": "Instruction",
    "conversation": "Conversation",
}

DOMAIN_MIXTURE_FAMILIES = {
    "stories",
    "reasoning",
    "social_emotional",
    "factual_knowledge",
    "mathematics",
    "code_technical",
    "language_basics",
    "structured_qa",
    "safety_uncertainty",
    "general_prose",
}

AGGREGATE_MIXTURE_FAMILIES = {"local_prose", "source_code", "online_base", "instruction", "conversation"}
MIXTURE_CHUNK_CHARS = 25_000


# Keep generated_curriculum as a legacy wrapper so older project-local copies
# made before the default_data flattening still classify correctly.
GENERIC_DEFAULT_DATA_FOLDERS = {"base_training", "code_training", "generated_curriculum"}

DEFAULT_STAGE_CATEGORY_FOLDERS = {
    "fine_tune_instruction": "instruction",
    "fine_tune_conversation": "conversation",
    "fine_tune_code": "code_technical",
}

CATEGORY_ALIASES = {
    "story": "stories",
    "stories": "stories",
    "reason": "reasoning",
    "reasoning": "reasoning",
    "why": "reasoning",
    "emotion": "social_emotional",
    "emotions": "social_emotional",
    "social": "social_emotional",
    "conversation": "social_emotional",
    "dialog": "social_emotional",
    "dialogue": "social_emotional",
    "geography": "factual_knowledge",
    "science": "factual_knowledge",
    "biology": "factual_knowledge",
    "physics": "factual_knowledge",
    "chemistry": "factual_knowledge",
    "astronomy": "factual_knowledge",
    "weather": "factual_knowledge",
    "earth": "factual_knowledge",
    "history": "factual_knowledge",
    "facts": "factual_knowledge",
    "knowledge": "factual_knowledge",
    "math": "mathematics",
    "mathematics": "mathematics",
    "code": "code_technical",
    "coding": "code_technical",
    "computer": "code_technical",
    "computers": "code_technical",
    "cs": "code_technical",
    "programming": "code_technical",
    "technical": "code_technical",
    "language": "language_basics",
    "grammar": "language_basics",
    "qa": "structured_qa",
    "question": "structured_qa",
    "answers": "structured_qa",
    "safety": "safety_uncertainty",
    "ethics": "safety_uncertainty",
    "honesty": "safety_uncertainty",
    "fairness": "safety_uncertainty",
    "uncertainty": "safety_uncertainty",
    "everyday": "general_prose",
    "health": "general_prose",
    "finance": "general_prose",
    "jobs": "general_prose",
    "prose": "general_prose",
}


def _slugify_category(value: str) -> str:
    """Convert text into a stable dataset category key.

    Args:
        value: Folder or file text.

    Returns:
        Lowercase underscore category key.
    """

    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "general_prose"


def _mixture_label(category: str) -> str:
    """Return a human-readable mixture label.

    Args:
        category: Mixture category key.

    Returns:
        Display label.
    """

    return MIXTURE_LABELS.get(category, category.replace("_", " ").title())


def _category_from_text(value: str) -> Optional[str]:
    """Infer a known dataset category from text.

    Args:
        value: Folder name or file stem.

    Returns:
        Canonical category key when a known token is present.
    """

    tokens = [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]
    for token in tokens:
        if token in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[token]
    return None


def _default_data_category(path: Path) -> Optional[str]:
    """Infer the Dataset Blueprint category for a bundled data file.

    Args:
        path: Source file path.

    Returns:
        Dataset category key when the path is recognized as bundled starter data.
    """

    parts_lower = [part.lower() for part in path.parts]
    data_roots = ("default_data", "training_data")
    root_indices = [parts_lower.index(root) for root in data_roots if root in parts_lower]
    if not root_indices:
        return None
    default_index = max(root_indices)
    relative_parts = path.parts[default_index + 1 :]
    relative_lower = [part.lower() for part in relative_parts]
    for folder, category in DEFAULT_STAGE_CATEGORY_FOLDERS.items():
        if folder in relative_lower[:-1]:
            return category
    if path.suffix.lower() in SUPPORTED_CODE_SUFFIXES or "code_training" in parts_lower:
        return "code_technical"
    for parent in reversed(relative_parts[:-1]):
        category = _category_from_text(parent)
        if category:
            return category
    stem_category = _category_from_text(path.stem)
    if stem_category:
        return stem_category
    for parent in reversed(relative_parts[:-1]):
        slug = _slugify_category(parent)
        if slug and slug not in GENERIC_DEFAULT_DATA_FOLDERS:
            return slug
    return "general_prose"


def _deduplicate_documents(documents: list[Document]) -> tuple[list[Document], dict[str, Any]]:
    """Remove exact duplicate extracted documents while preserving order.

    Args:
        documents: Loaded documents.

    Returns:
        Deduplicated documents and a small report.
    """

    unique_documents: list[Document] = []
    seen: dict[str, Document] = {}
    duplicates: list[dict[str, str]] = []
    for document in documents:
        canonical_text = _canonical_corpus_block(document.text)
        if not canonical_text:
            unique_documents.append(document)
            continue
        digest = hashlib.sha256(
            f"{document.kind}\n{document.language or ''}\n{canonical_text}".encode("utf-8")
        ).hexdigest()
        original = seen.get(digest)
        if original is not None:
            duplicates.append(
                {
                    "path": str(document.path),
                    "duplicate_of": str(original.path),
                    "kind": document.kind,
                }
            )
            continue
        seen[digest] = document
        unique_documents.append(document)
    return unique_documents, {
        "removed_documents": len(duplicates),
        "duplicates": duplicates[:50],
    }


def _document_mixture_family(document: Document) -> str:
    """Classify a document into a dataset mixture source family.

    Args:
        document: Loaded source document.

    Returns:
        Mixture source family identifier.
    """

    default_category = _default_data_category(document.path)
    if default_category:
        return default_category
    if document.kind == "code":
        return "source_code"
    if document.kind == "instruction":
        return "instruction"
    if document.kind == "conversation":
        return "conversation"
    dataset_id = str(document.language or "")
    preset = CONVERSATION_DATASET_PRESETS.get(dataset_id)
    if preset and preset.stage == "base":
        return "online_base"
    if "__hf_datasets__" in document.path.parts:
        for part in document.path.parts:
            preset = CONVERSATION_DATASET_PRESETS.get(part)
            if preset and preset.stage == "base":
                return "online_base"
    return "local_prose"


def _stable_document_sort_key(document: Document) -> str:
    """Return a stable pseudo-random sort key for sampling.

    Args:
        document: Loaded source document.

    Returns:
        Hex digest used to order documents deterministically.
    """

    text_digest = hashlib.sha256(document.text[:4096].encode("utf-8", errors="ignore")).hexdigest()
    key = f"{document.path}|{document.kind}|{document.language or ''}|{len(document.text)}|{text_digest}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _chunk_document_for_mixture(document: Document, chunk_chars: int = MIXTURE_CHUNK_CHARS) -> list[Document]:
    """Split a large document into sampler-sized pieces.

    Mixture percentages are meant to control corpus weight, not whole-file
    presence. Generated code files can be many megabytes each, so selecting
    whole documents makes a 5% code request become a 70%+ code corpus. Chunking
    lets the sampler take only the amount needed from large categories.

    Args:
        document: Source document.
        chunk_chars: Approximate maximum characters per chunk.

    Returns:
        One or more documents suitable for weighted sampling.
    """

    text = document.text
    if len(text) <= chunk_chars:
        return [document]
    chunks: list[Document] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary <= start + int(chunk_chars * 0.5):
                boundary = text.rfind("\n", start, end)
            if boundary > start + int(chunk_chars * 0.5):
                end = boundary
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Document(
                    path=document.path,
                    text=chunk_text,
                    kind=document.kind,
                    language=document.language,
                )
            )
        start = max(end, start + 1)
    return chunks or [document]


def _chunk_documents_for_mixture(documents: list[Document]) -> list[Document]:
    """Split large documents before applying mixture percentages.

    Args:
        documents: Loaded documents.

    Returns:
        Documents and chunks ready for weighted sampling.
    """

    chunks: list[Document] = []
    for document in documents:
        chunks.extend(_chunk_document_for_mixture(document))
    return chunks


def _empty_mixture_report(weights: dict[str, float], documents: list[Document], applied: bool, reason: str = "") -> dict[str, Any]:
    """Build a mixture report without changing document selection.

    Args:
        weights: Requested mixture weights.
        documents: Available documents.
        applied: Whether weighted sampling was applied.
        reason: Optional reason when sampling was not applied.

    Returns:
        Mixture report dictionary.
    """

    document_families = {_document_mixture_family(document) for document in documents}
    families_to_report = sorted({*MIXTURE_LABELS, *weights, *document_families})
    by_family: dict[str, list[Document]] = {key: [] for key in families_to_report}
    for document in documents:
        by_family.setdefault(_document_mixture_family(document), []).append(document)
    total_chars = sum(len(document.text) for document in documents)
    families = {}
    for family in families_to_report:
        available = by_family.get(family, [])
        available_chars = sum(len(document.text) for document in available)
        families[family] = {
            "label": _mixture_label(family),
            "requested_weight": float(weights.get(family, 0.0) or 0.0),
            "available_documents": len(available),
            "available_characters": available_chars,
            "selected_documents": len(available) if not applied else 0,
            "selected_characters": available_chars if not applied else 0,
            "actual_percent": (available_chars * 100.0 / total_chars) if total_chars else 0.0,
            "dropped_documents": 0,
            "dropped_characters": 0,
        }
    return {
        "applied": applied,
        "reason": reason,
        "total_available_documents": len(documents),
        "total_selected_documents": len(documents) if not applied else 0,
        "total_available_characters": total_chars,
        "total_selected_characters": total_chars if not applied else 0,
        "families": families,
    }


def _apply_dataset_mixture(
    documents: list[Document],
    weights: dict[str, float],
    progress: Optional[Callable[[Any], None]],
) -> tuple[list[Document], dict[str, Any]]:
    """Apply weighted sampling by source family.

    Args:
        documents: Loaded documents before mixture sampling.
        weights: Requested mixture percentages.
        progress: Optional progress callback.

    Returns:
        Selected documents and mixture report.
    """

    original_document_count = len(documents)
    documents = _chunk_documents_for_mixture(documents)
    if len(documents) != original_document_count:
        _emit(
            progress,
            f"Dataset mixture: split {original_document_count:,} source file(s) into "
            f"{len(documents):,} sampling chunk(s) for accurate percentages.",
            49,
        )
    document_families = {_document_mixture_family(document) for document in documents}
    families_to_sample = sorted({*MIXTURE_LABELS, *weights, *document_families})
    clean_weights: dict[str, float] = {}
    for family in families_to_sample:
        try:
            clean_weights[family] = max(0.0, float(weights.get(family, 0.0) or 0.0))
        except (TypeError, ValueError):
            clean_weights[family] = 0.0
    domain_weight_total = sum(clean_weights.get(family, 0.0) for family in DOMAIN_MIXTURE_FAMILIES)
    aggregate_weight_total = sum(clean_weights.get(family, 0.0) for family in AGGREGATE_MIXTURE_FAMILIES)
    has_domain_documents = bool(document_families & DOMAIN_MIXTURE_FAMILIES)
    if has_domain_documents and domain_weight_total >= 99.0 and aggregate_weight_total > 0.0:
        for family in AGGREGATE_MIXTURE_FAMILIES:
            clean_weights[family] = 0.0
        _emit(progress, "Dataset mixture: ignored legacy aggregate weights because a full domain recipe is active.", 49)
    requested_total = sum(clean_weights.values())
    if requested_total <= 0.0:
        return documents, _empty_mixture_report(clean_weights, documents, applied=False, reason="No positive mixture weights.")

    grouped: dict[str, list[Document]] = {key: [] for key in families_to_sample}
    for document in documents:
        grouped.setdefault(_document_mixture_family(document), []).append(document)
    available_families = {
        family: items
        for family, items in grouped.items()
        if items and clean_weights.get(family, 0.0) > 0.0
    }
    if not available_families:
        return documents, _empty_mixture_report(
            clean_weights,
            documents,
            applied=False,
            reason="No documents matched positive mixture weights.",
        )

    total_available_chars = sum(len(document.text) for document in documents)
    active_weight_total = sum(clean_weights[family] for family in available_families)
    available_chars_by_family = {
        family: sum(len(document.text) for document in items)
        for family, items in available_families.items()
    }
    normalized_shares = {
        family: clean_weights[family] / active_weight_total
        for family in available_families
        if clean_weights[family] > 0.0
    }
    limiting_family = min(
        normalized_shares,
        key=lambda family: available_chars_by_family[family] / normalized_shares[family],
    )
    strict_total_budget = int(
        available_chars_by_family[limiting_family] / normalized_shares[limiting_family]
    )
    family_targets = {
        family: max(1, int(strict_total_budget * share))
        for family, share in normalized_shares.items()
    }
    if strict_total_budget < total_available_chars:
        _emit(
            progress,
            (
                "Dataset mixture: strict recipe limited by "
                f"{_mixture_label(limiting_family)} availability "
                f"({available_chars_by_family[limiting_family]:,} characters). "
                "Add more data for that category or lower its percentage to use more of the corpus."
            ),
            49,
        )
    sorted_groups = {
        family: sorted(items, key=_stable_document_sort_key)
        for family, items in available_families.items()
    }
    selected_by_family: dict[str, list[Document]] = {family: [] for family in families_to_sample}
    selected_ids: set[int] = set()

    for family, items in sorted_groups.items():
        target_chars = family_targets.get(family, 0)
        selected_chars = 0
        for document in items:
            if selected_chars >= target_chars and selected_by_family[family]:
                break
            selected_by_family[family].append(document)
            selected_ids.add(id(document))
            selected_chars += len(document.text)

    selected_documents = [
        document
        for family in families_to_sample
        for document in selected_by_family.get(family, [])
    ]
    selected_documents = sorted(selected_documents, key=lambda document: (str(document.path), document.kind, document.language or ""))
    selected_total_chars = sum(len(document.text) for document in selected_documents)
    report = {
        "applied": True,
        "reason": "",
        "total_available_documents": original_document_count,
        "total_available_chunks": len(documents),
        "total_selected_documents": len(selected_documents),
        "total_available_characters": total_available_chars,
        "total_selected_characters": selected_total_chars,
        "families": {},
    }
    for family in families_to_sample:
        available = grouped.get(family, [])
        selected = selected_by_family.get(family, [])
        available_chars = sum(len(document.text) for document in available)
        selected_chars = sum(len(document.text) for document in selected)
        report["families"][family] = {
            "label": _mixture_label(family),
            "requested_weight": clean_weights.get(family, 0.0),
            "effective_requested_percent": (
                clean_weights.get(family, 0.0) * 100.0 / active_weight_total
                if family in available_families and active_weight_total > 0.0
                else 0.0
            ),
            "available_documents": len(available),
            "available_characters": available_chars,
            "selected_documents": len(selected),
            "selected_characters": selected_chars,
            "target_characters": family_targets.get(family, 0),
            "actual_percent": (selected_chars * 100.0 / selected_total_chars) if selected_total_chars else 0.0,
            "dropped_documents": max(0, len(available) - len(selected)),
            "dropped_characters": max(0, available_chars - selected_chars),
        }
    if len(selected_documents) != len(documents):
        _emit(
            progress,
            f"Weighted sampler selected {len(selected_documents):,}/{len(documents):,} sampling chunks "
            f"({selected_total_chars:,}/{total_available_chars:,} characters).",
            49,
        )
    return selected_documents or documents, report


def build_dataset(
    config: DatasetConfig,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> DatasetBuildResult:
    """Build a tokenizer-ready dataset project.

    Args:
        config: Dataset preparation settings.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when the user requested stop.

    Returns:
        Dataset build summary.

    Raises:
        ValueError: If no supported documents are found.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, "Scanning source folder...", 3)
    (
        documents,
        manifest,
        cached_file_count,
        processed_file_count,
        skipped_file_count,
        failed_file_count,
    ) = _load_documents_with_cache(config, progress, should_stop)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    if not documents:
        raise ValueError("No supported text, PDF, JSONL, or structured JSON documents were found.")
    documents, exact_dedup_report = _deduplicate_documents(documents)
    if exact_dedup_report["removed_documents"]:
        _emit(
            progress,
            f"Removed {exact_dedup_report['removed_documents']:,} exact duplicate extracted document(s).",
            44,
        )
    documents, mixture_report = _apply_dataset_mixture(documents, config.mixture_weights, progress)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    if not documents:
        raise ValueError("Dataset mixture selected no documents. Adjust mixture weights and try again.")

    all_text = "\n".join(doc.text for doc in documents)
    character_count = len(all_text)
    unique_words = len({word.lower() for word in all_text.split()})
    suggested_vocab_size = estimate_vocab_size(character_count, unique_words)
    selected_vocab_size = config.vocab_size or suggested_vocab_size
    warning = content_warning(character_count)
    code_sample_count = sum(1 for doc in documents if doc.kind == "code")
    conversation_sample_count = sum(1 for doc in documents if doc.kind in {"conversation", "instruction"})
    prose_sample_count = sum(1 for doc in documents if doc.kind not in {"code", "conversation", "instruction"})
    _emit(progress, f"Content size: {character_count:,} characters across {len(documents)} files.", 45)
    if config.code_training_mode:
        _emit(progress, f"Code mode: {code_sample_count:,} code samples, {prose_sample_count:,} prose samples.", 46)
    if conversation_sample_count:
        _emit(progress, f"Conversation data: {conversation_sample_count:,} dialogue/instruction samples.", 46)
    if cached_file_count or processed_file_count:
        _emit(progress, f"Cache: reused {cached_file_count:,} file(s), processed {processed_file_count:,} file(s).", 47)
    if skipped_file_count or failed_file_count:
        _emit(progress, f"Quality: skipped {skipped_file_count:,} empty file(s), failed {failed_file_count:,} file(s).", 48)
    if config.mixture_weights:
        mixture_parts = []
        for key, value in config.mixture_weights.items():
            try:
                numeric_value = float(value or 0.0)
            except (TypeError, ValueError):
                numeric_value = 0.0
            if numeric_value > 0.0:
                mixture_parts.append(f"{key.replace('_', ' ')} {numeric_value:.1f}%")
        mixture_text = ", ".join(mixture_parts)
        if mixture_text:
            _emit(progress, f"Dataset mixture plan: {mixture_text}.", 49)
    if mixture_report.get("applied"):
        for family, row in mixture_report.get("families", {}).items():
            if int(row.get("selected_documents", 0) or 0) > 0 or float(row.get("requested_weight", 0.0) or 0.0) > 0.0:
                _emit(
                    progress,
                    (
                        f"Mixture {row.get('label', family)}: requested "
                        f"{float(row.get('requested_weight', 0.0) or 0.0):.1f}% "
                        f"(effective {float(row.get('effective_requested_percent', 0.0) or 0.0):.1f}%), "
                        f"selected {int(row.get('selected_documents', 0) or 0):,}/"
                        f"{int(row.get('available_documents', 0) or 0):,} chunk(s), "
                        f"actual {float(row.get('actual_percent', 0.0) or 0.0):.1f}%."
                    ),
                    49,
                )
    _emit(progress, f"Unique word estimate: {unique_words:,}.", 48)
    _emit(progress, f"Auto vocabulary size: {selected_vocab_size:,}.", 50)
    if warning:
        _emit(progress, f"Warning: {warning}")

    corpus_path = config.output_dir / "corpus.txt"
    _emit(progress, "Writing normalized corpus...", 56)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    write_training_corpus(
        documents,
        corpus_path,
        code_training_mode=config.code_training_mode,
        generate_instruction_samples=config.generate_instruction_samples,
        reasoning_sample_mode=config.reasoning_sample_mode,
    )
    corpus_text = corpus_path.read_text(encoding="utf-8")
    duplicate_report = _text_block_duplicate_report(corpus_text)
    _emit(
        progress,
        (
            "Corpus diversity: "
            f"{duplicate_report['unique_block_count']:,}/{duplicate_report['block_count']:,} unique blocks, "
            f"{duplicate_report['duplicate_block_ratio'] * 100:.1f}% repeated."
        ),
        74,
    )
    tokenizer_path = config.output_dir / "tokenizer.json"
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    tokenizer, reuse_tokenizer, tokenizer_imported, tokenizer_source_path = _load_or_create_tokenizer(
        config,
        corpus_path,
        tokenizer_path,
        selected_vocab_size,
        progress,
        should_stop,
    )
    validate_training_tokenizer(tokenizer)

    _emit(progress, "Encoding corpus into token IDs...", 78)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    # Token IDs are streamed straight to a flat binary file instead of being
    # accumulated as one giant Python list -- see encode_file_to_bin. The
    # dtype is picked from the vocab size (uint16 covers every vocab size
    # this app uses in practice), so storage is ~4x smaller than the old
    # JSON-array format and loading later is a near-instant memmap open
    # instead of a full json.loads() over the whole corpus.
    token_dtype = token_dtype_for_vocab(tokenizer.get_vocab_size())
    full_tokens_path = config.output_dir / "tokens_full.bin"
    token_count = encode_file_to_bin(tokenizer, corpus_path, full_tokens_path, token_dtype, should_stop=should_stop)
    _emit(progress, f"Encoded {token_count:,} tokens.", 86)
    token_density = (token_count / max(len(corpus_text), 1)) if corpus_text else 0.0
    document_token_lengths = [max(1, int(round(len(doc.text) * token_density))) for doc in documents if doc.text]
    if document_token_lengths:
        sequence_stats = {
            "min": min(document_token_lengths),
            "average": sum(document_token_lengths) / len(document_token_lengths),
            "median": statistics.median(document_token_lengths),
            "max": max(document_token_lengths),
        }
    else:
        sequence_stats = {"min": 0, "average": 0.0, "median": 0.0, "max": 0}
    _emit(
        progress,
        (
            "Token distribution: "
            f"min {int(sequence_stats['min']):,}, "
            f"avg {float(sequence_stats['average']):,.0f}, "
            f"median {float(sequence_stats['median']):,.0f}, "
            f"max {int(sequence_stats['max']):,}."
        ),
        88,
    )
    train_tokens_path = config.output_dir / "train_tokens.bin"
    val_tokens_path = config.output_dir / "val_tokens.bin"
    train_token_count, val_token_count = write_split_token_bins(
        full_tokens_path,
        token_dtype,
        config.validation_split,
        train_tokens_path,
        val_tokens_path,
    )
    full_tokens_path.unlink(missing_ok=True)
    train_window_count = max(0, train_token_count - config.context_length)
    val_window_count = max(0, val_token_count - config.context_length)
    _emit(progress, f"Training tokens: {train_token_count:,}; validation tokens: {val_token_count:,}.", 92)
    _emit(progress, f"Training windows: {train_window_count:,}; validation windows: {val_window_count:,}.", 92)
    quality_report = _dataset_quality_report(
        document_count=len(documents),
        token_count=token_count,
        vocab_size=tokenizer.get_vocab_size(),
        unique_words=unique_words,
        train_window_count=train_window_count,
        val_window_count=val_window_count,
        code_sample_count=code_sample_count,
        prose_sample_count=prose_sample_count,
        conversation_sample_count=conversation_sample_count,
        skipped_file_count=skipped_file_count,
        failed_file_count=failed_file_count,
        warning=warning,
        sequence_stats=sequence_stats,
        duplicate_report=duplicate_report,
    )
    _emit(
        progress,
        f"Dataset rating: {quality_report['stars']:.1f}/5 stars ({quality_report['label']}, score {quality_report['score']:.1f}/100).",
        94,
    )

    summary = {
        "dataset_config": dataclass_to_jsonable(config),
        "document_count": len(documents),
        "character_count": character_count,
        "token_count": token_count,
        "train_token_count": train_token_count,
        "val_token_count": val_token_count,
        "token_dtype": str(token_dtype),
        "train_window_count": train_window_count,
        "val_window_count": val_window_count,
        "context_length": config.context_length,
        "sequence_token_stats": sequence_stats,
        "code_sample_count": code_sample_count,
        "prose_sample_count": prose_sample_count,
        "conversation_sample_count": conversation_sample_count,
        "dataset_stage": config.dataset_stage,
        "conversation_datasets": config.conversation_datasets,
        "conversation_sample_limit": config.conversation_sample_limit,
        "conversation_dataset_path": str(config.conversation_dataset_path or ""),
        "instruction_dataset_path": str(config.instruction_dataset_path or ""),
        "conversation_dataset_paths": [str(path) for path in config.conversation_dataset_paths],
        "instruction_dataset_paths": [str(path) for path in config.instruction_dataset_paths],
        "default_data_paths": [str(path) for path in config.default_data_paths],
        "mixture_weights": config.mixture_weights,
        "mixture_report": mixture_report,
        "exact_duplicate_documents_removed": exact_dedup_report["removed_documents"],
        "exact_duplicate_document_examples": exact_dedup_report["duplicates"],
        "suggested_vocab_size": suggested_vocab_size,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "warning": warning,
        "source_files": [str(doc.path) for doc in documents],
        "cached_file_count": cached_file_count,
        "processed_file_count": processed_file_count,
        "skipped_file_count": skipped_file_count,
        "failed_file_count": failed_file_count,
        "source_file_count": len(manifest.get("files", {})),
        "prepare_mode": config.prepare_mode,
        "tokenizer_strategy": config.tokenizer_strategy,
        "reasoning_sample_mode": config.reasoning_sample_mode,
        "tokenizer_reused": reuse_tokenizer,
        "tokenizer_imported": tokenizer_imported,
        "tokenizer_source_path": tokenizer_source_path,
        "quality_score": quality_report["score"],
        "quality_stars": quality_report["stars"],
        "quality_label": quality_report["label"],
        "quality_reasons": quality_report["reasons"],
        "quality_components": quality_report["components"],
        "duplicate_block_count": duplicate_report["duplicate_block_count"],
        "unique_block_count": duplicate_report["unique_block_count"],
        "corpus_block_count": duplicate_report["block_count"],
        "duplicate_block_ratio": duplicate_report["duplicate_block_ratio"],
        "unique_block_ratio": duplicate_report["unique_block_ratio"],
        "most_repeated_block_count": duplicate_report["most_repeated_block_count"],
        "top_repeated_blocks": duplicate_report["top_repeated_blocks"],
    }
    dataset_version = record_dataset_version(config.output_dir, summary, manifest)
    write_json(config.output_dir / "dataset_summary.json", summary)
    write_json(config.output_dir / "dataset_manifest.json", manifest)
    _emit(progress, f"Dataset version recorded: {dataset_version['version_id']}.", 98)
    _emit(progress, f"Dataset ready: {config.output_dir}", 100)
    return DatasetBuildResult(
        config.output_dir,
        tokenizer_path,
        len(documents),
        token_count,
        tokenizer.get_vocab_size(),
        character_count,
        suggested_vocab_size,
        train_window_count,
        val_window_count,
        sequence_stats,
        warning,
        code_sample_count,
        prose_sample_count,
        conversation_sample_count,
        cached_file_count,
        processed_file_count,
        skipped_file_count,
        failed_file_count,
        str(dataset_version["version_id"]),
        int(dataset_version["version_number"]),
        mixture_report,
        float(quality_report["score"]),
        float(quality_report["stars"]),
        str(quality_report["label"]),
        list(quality_report["reasons"]),
        int(duplicate_report["duplicate_block_count"]),
        int(duplicate_report["unique_block_count"]),
        int(duplicate_report["block_count"]),
        float(duplicate_report["duplicate_block_ratio"]),
        float(duplicate_report["unique_block_ratio"]),
    )


def _bounded_ratio(value: float, target: float) -> float:
    """Return value/target clamped between 0 and 1.

    Args:
        value: Actual metric value.
        target: Metric value that should receive full credit.

    Returns:
        Clamped ratio.
    """

    if target <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(target)))


def _canonical_corpus_block(text: str) -> str:
    """Normalize a corpus block for repeated-content checks.

    Args:
        text: Raw block text.

    Returns:
        Whitespace-normalized lowercase text.
    """

    return re.sub(r"\s+", " ", text).strip().lower()


def _text_block_duplicate_report(corpus_text: str, max_blocks: int = 500_000) -> dict[str, Any]:
    """Measure exact repeated blocks in the prepared corpus text.

    Args:
        corpus_text: Fully written corpus text.
        max_blocks: Maximum number of non-empty blocks to inspect.

    Returns:
        Dictionary with block counts, ratios, and top repeated examples.
    """

    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    total_blocks = 0
    ignored_blocks = 0
    start = 0
    for match in re.finditer(r"\n\s*\n+", corpus_text):
        if total_blocks >= max_blocks:
            break
        raw_block = corpus_text[start : match.start()]
        start = match.end()
        canonical = _canonical_corpus_block(raw_block)
        if len(canonical) < 12:
            ignored_blocks += 1
            continue
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        counts[digest] += 1
        examples.setdefault(digest, canonical[:240])
        total_blocks += 1
    if total_blocks < max_blocks:
        canonical = _canonical_corpus_block(corpus_text[start:])
        if len(canonical) >= 12:
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            counts[digest] += 1
            examples.setdefault(digest, canonical[:240])
            total_blocks += 1
        elif canonical:
            ignored_blocks += 1

    unique_blocks = len(counts)
    duplicate_blocks = sum(count - 1 for count in counts.values() if count > 1)
    duplicate_ratio = duplicate_blocks / max(total_blocks, 1)
    unique_ratio = unique_blocks / max(total_blocks, 1)
    repeated = [
        {"count": count, "sample": examples[digest]}
        for digest, count in counts.most_common(8)
        if count > 1
    ]
    return {
        "block_count": total_blocks,
        "unique_block_count": unique_blocks,
        "duplicate_block_count": duplicate_blocks,
        "duplicate_block_ratio": duplicate_ratio,
        "unique_block_ratio": unique_ratio,
        "ignored_block_count": ignored_blocks,
        "truncated": total_blocks >= max_blocks,
        "most_repeated_block_count": repeated[0]["count"] if repeated else 1,
        "top_repeated_blocks": repeated,
    }


def _dataset_quality_report(
    *,
    document_count: int,
    token_count: int,
    vocab_size: int,
    unique_words: int,
    train_window_count: int,
    val_window_count: int,
    code_sample_count: int,
    prose_sample_count: int,
    conversation_sample_count: int,
    skipped_file_count: int,
    failed_file_count: int,
    warning: Optional[str],
    sequence_stats: dict[str, float],
    duplicate_report: dict[str, Any],
) -> dict[str, Any]:
    """Rate a prepared dataset for small-LLM training readiness.

    Args:
        document_count: Prepared document/sample count.
        token_count: Total token count.
        vocab_size: Final tokenizer vocabulary size.
        unique_words: Estimated unique words in the corpus.
        train_window_count: Number of trainable context windows.
        val_window_count: Number of validation context windows.
        code_sample_count: Prepared code sample count.
        prose_sample_count: Prepared prose sample count.
        conversation_sample_count: Prepared conversation/instruction sample count.
        skipped_file_count: Empty or unreadable source files skipped.
        failed_file_count: Source files that failed extraction.
        warning: Size/content warning string.
        sequence_stats: Approximate per-document token distribution.
        duplicate_report: Repeated text-block report for the written corpus.

    Returns:
        Dataset quality dictionary with score, stars, label, and reasons.
    """

    reasons: list[str] = []
    token_score = 30.0 * _bounded_ratio(token_count, 1_000_000)
    window_score = 20.0 * _bounded_ratio(train_window_count, 50_000)
    vocab_target = max(4_000.0, min(32_000.0, unique_words * 0.8))
    vocab_score = 18.0 * _bounded_ratio(vocab_size, vocab_target)
    document_score = 12.0 * _bounded_ratio(document_count, 1_000)
    validation_score = 8.0 * _bounded_ratio(val_window_count, 2_000)
    families = sum(1 for count in (code_sample_count, prose_sample_count, conversation_sample_count) if count > 0)
    diversity_score = 7.0 * _bounded_ratio(families, 3)
    average_sequence = float(sequence_stats.get("average", 0.0) or 0.0)
    sequence_score = 5.0 * _bounded_ratio(average_sequence, 256)
    penalty = min(20.0, failed_file_count * 3.0 + skipped_file_count * 0.5)
    duplicate_ratio = float(duplicate_report.get("duplicate_block_ratio", 0.0) or 0.0)
    duplicate_penalty = min(35.0, duplicate_ratio * 70.0)
    penalty += duplicate_penalty
    if warning and warning != "none":
        penalty += 5.0
    score = max(
        0.0,
        min(
            100.0,
            token_score
            + window_score
            + vocab_score
            + document_score
            + validation_score
            + diversity_score
            + sequence_score
            - penalty,
        ),
    )
    stars = round(score / 20.0 * 2.0) / 2.0
    if score >= 85:
        label = "Excellent"
    elif score >= 70:
        label = "Good"
    elif score >= 50:
        label = "Usable"
    elif score >= 30:
        label = "Weak"
    else:
        label = "Very weak"
    if token_count < 250_000:
        reasons.append("Token count is low for robust training.")
    else:
        reasons.append("Token count is sufficient for a small experiment.")
    if train_window_count < 5_000:
        reasons.append("Few training windows; model may memorize quickly.")
    if vocab_size < 4_000:
        reasons.append("Vocabulary is small; language coverage may be limited.")
    elif vocab_size > 50_000:
        reasons.append("Vocabulary is large; tiny models may spend capacity on tokens.")
    else:
        reasons.append("Vocabulary size is in a reasonable small-model range.")
    if families >= 2:
        reasons.append("Dataset includes multiple content families.")
    if skipped_file_count or failed_file_count:
        reasons.append(f"Extraction skipped {skipped_file_count} file(s) and failed {failed_file_count} file(s).")
    if duplicate_ratio >= 0.5:
        reasons.append("Prepared corpus is heavily repeated; training may memorize instead of generalize.")
    elif duplicate_ratio >= 0.2:
        reasons.append("Prepared corpus has many repeated blocks; add more varied data or deduplicate.")
    elif duplicate_ratio >= 0.05:
        reasons.append("Prepared corpus has some repeated blocks.")
    else:
        reasons.append("Prepared corpus block diversity looks healthy.")
    if warning and warning != "none":
        reasons.append(str(warning))
    return {
        "score": round(score, 1),
        "stars": stars,
        "label": label,
        "reasons": reasons,
        "components": {
            "tokens": round(token_score, 1),
            "windows": round(window_score, 1),
            "vocabulary": round(vocab_score, 1),
            "documents": round(document_score, 1),
            "validation": round(validation_score, 1),
            "diversity": round(diversity_score, 1),
            "sequence": round(sequence_score, 1),
            "duplicate_penalty": round(duplicate_penalty, 1),
            "penalty": round(penalty, 1),
        },
    }


def _resume_checkpoint_for(training_config: TrainingConfig) -> Optional[Path]:
    """Return the checkpoint that will be used for resume, if any.

    Args:
        training_config: Training configuration.

    Returns:
        Resume checkpoint path or ``None``.
    """

    if not training_config.resume:
        return None
    if training_config.resume_from_checkpoint:
        return Path(training_config.resume_from_checkpoint)
    return latest_checkpoint(training_config.output_dir / "checkpoints")


def _compatible_model_config(model_config: ModelConfig) -> dict[str, Any]:
    """Return checkpoint compatibility fields for a model config.

    Args:
        model_config: Current model configuration.

    Returns:
        Dictionary of architecture-shape fields.
    """

    data = dataclass_to_jsonable(model_config)
    return {
        key: data.get(key)
        for key in (
            "vocab_size",
            "context_length",
            "embedding_size",
            "head_count",
            "layer_count",
            "bias",
            "norm_type",
            "position_encoding",
            "mlp_type",
            "rope_theta",
            "attention_type",
        )
    }


def _tokenizer_files_match(left: Path, right: Path) -> bool:
    """Return whether tokenizer files are byte-identical or JSON-equivalent.

    Args:
        left: First tokenizer path.
        right: Second tokenizer path.

    Returns:
        True when tokenizers are compatible.
    """

    if file_sha256(left) == file_sha256(right):
        return True
    left_json = read_json(left, default=None)
    right_json = read_json(right, default=None)
    return left_json is not None and left_json == right_json


def _validate_resume_compatibility(
    data_dir: Path,
    tokenizer_path: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> Optional[Path]:
    """Validate tokenizer and architecture before continuing training.

    Args:
        data_dir: Prepared dataset folder.
        tokenizer_path: Dataset tokenizer path.
        model_config: Current model architecture.
        training_config: Training configuration.

    Returns:
        Resume checkpoint path when one exists.

    Raises:
        ValueError: If tokenizer or architecture is incompatible.
    """

    resume_path = _resume_checkpoint_for(training_config)
    if not resume_path or not resume_path.exists() or not training_config.require_compatible_resume:
        return resume_path

    existing_tokenizer = training_config.output_dir / "tokenizer.json"
    if existing_tokenizer.exists():
        if not _tokenizer_files_match(existing_tokenizer, tokenizer_path):
            raise ValueError(
                "Resume safety check failed: the selected dataset tokenizer does not match the tokenizer "
                f"used by the existing model folder.\nExisting tokenizer: {existing_tokenizer}\n"
                f"Dataset tokenizer: {tokenizer_path}\n\nUse the same tokenizer policy for continued training, "
                "or choose a new model output folder to start a new model."
            )

    checkpoint = torch.load(resume_path, map_location="cpu")
    checkpoint_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Resume safety check failed: checkpoint has no model_config: {resume_path}")

    current = _compatible_model_config(model_config)
    legacy_defaults = {
        "bias": True,
        "norm_type": "layernorm",
        "position_encoding": "learned",
        "mlp_type": "gelu",
        "rope_theta": 10000.0,
        "attention_type": "mha",
    }
    previous = {key: checkpoint_config.get(key, legacy_defaults.get(key)) for key in current}
    mismatches = {
        key: {"checkpoint": previous.get(key), "current": current.get(key)}
        for key in current
        if previous.get(key) != current.get(key)
    }
    if mismatches:
        mismatch_text = ", ".join(
            f"{key} checkpoint={value['checkpoint']} current={value['current']}"
            for key, value in mismatches.items()
        )
        raise ValueError(
            "Resume safety check failed: model architecture does not match the checkpoint. "
            f"{mismatch_text}. Keep architecture settings identical for continued training, "
            "or use a new model output folder."
        )

    return resume_path


def train_from_dataset(
    data_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> TrainingResult:
    """Train a model using a prepared dataset folder.

    Args:
        data_dir: Prepared dataset folder.
        model_config: Model architecture settings.
        training_config: Optimizer and checkpoint settings.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when the user requested stop.

    Returns:
        Training result with final checkpoint and summary paths.

    Raises:
        FileNotFoundError: If the tokenizer is missing.
    """

    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    from .tokenizer import load_tokenizer

    dataset_summary = read_json(data_dir / "dataset_summary.json", default={}) or {}
    dataset_lineage = read_json(data_dir / "dataset_lineage.json", default={}) or {}
    tokenizer = load_tokenizer(tokenizer_path)
    validate_training_tokenizer(tokenizer)

    train_bin_path = data_dir / "train_tokens.bin"
    val_bin_path = data_dir / "val_tokens.bin"
    if train_bin_path.exists() and val_bin_path.exists():
        # Memory-mapped: opening costs almost no RAM regardless of corpus
        # size, and windows are paged in from disk lazily during training.
        dtype_name = dataset_summary.get("token_dtype") or str(token_dtype_for_vocab(tokenizer.get_vocab_size()))
        token_dtype = np.dtype(dtype_name)
        train_tokens = load_token_memmap(train_bin_path, token_dtype)
        val_tokens = load_token_memmap(val_bin_path, token_dtype)
    else:
        # Backward compatibility with datasets prepared before the .bin
        # format existed. Re-preparing the dataset will upgrade it.
        train_tokens = json.loads((data_dir / "train_tokens.json").read_text(encoding="utf-8"))
        val_tokens = json.loads((data_dir / "val_tokens.json").read_text(encoding="utf-8"))

    if model_config.vocab_size != tokenizer.get_vocab_size():
        model_config.vocab_size = tokenizer.get_vocab_size()

    training_config.output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = _validate_resume_compatibility(data_dir, tokenizer_path, model_config, training_config)
    if resume_path:
        _emit(progress, f"Resume safety check passed: {resume_path}", 3)
    shutil.copy2(tokenizer_path, training_config.output_dir / "tokenizer.json")
    result = train_model(
        model_config,
        training_config,
        train_tokens,
        val_tokens,
        pad_token_id=token_id(tokenizer, PAD_TOKEN),
        progress=progress,
        should_stop=should_stop,
        decode_preview=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
    )
    training_summary = read_json(result.summary_path, default={}) or {}
    run_id = f"run_{utc_timestamp()}_{stable_json_hash({'dataset': dataset_summary.get('dataset_version'), 'model': training_summary.get('model_config'), 'training': training_summary.get('training_config')})}"
    lineage = {
        "schema": "micro_llm_model_lineage",
        "version": 1,
        "training_run_id": run_id,
        "created_at": utc_timestamp(),
        "dataset_dir": str(data_dir),
        "dataset_id": dataset_summary.get("dataset_id") or dataset_lineage.get("dataset_id"),
        "dataset_version": dataset_summary.get("dataset_version"),
        "dataset_fingerprint": (dataset_summary.get("dataset_version") or {}).get("source_fingerprint"),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "training_mode": training_config.training_mode,
        "fine_tune_from_checkpoint": (
            str(training_config.fine_tune_from_checkpoint)
            if training_config.fine_tune_from_checkpoint
            else None
        ),
        "peft_method": training_config.peft_method,
        "lora_rank": training_config.lora_rank if training_config.peft_method == "lora" else None,
        "lora_alpha": training_config.lora_alpha if training_config.peft_method == "lora" else None,
        "lora_target_modules": training_config.lora_target_modules if training_config.peft_method == "lora" else None,
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "resume_safety_required": training_config.require_compatible_resume,
        "checkpoint_path": str(result.checkpoint_path),
        "summary_path": str(result.summary_path),
        "stopped": result.stopped,
    }
    training_summary["training_run_id"] = run_id
    training_summary["model_lineage"] = lineage
    write_json(result.summary_path, training_summary)
    write_json(training_config.output_dir / "model_lineage.json", lineage)
    write_json(training_config.output_dir / "dataset_summary.json", dataset_summary)
    return result