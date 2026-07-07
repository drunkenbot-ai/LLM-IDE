from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import PyPDF2
import torch

from .config import DatasetConfig
from .conversation_datasets import CONVERSATION_DATASET_PRESETS
from .data import (
    SUPPORTED_CODE_SUFFIXES,
    SUPPORTED_TEXT_SUFFIXES,
    file_fingerprint,
    load_structured_json_documents,
    supported_source_paths,
)
from .dataset_build import _local_structured_dataset_paths
from .lineage import read_json, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class ProjectHealthResult:
    status: str
    checks: list[dict[str, str]]
    summary: str


@dataclass
class DatasetPreviewResult:
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
    LOGGER.info(message)
    if progress:
        progress({"message": message, "percent": percent})


def _health_check(name: str, status: str, detail: str) -> dict[str, str]:
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
    required_dataset = ("tokenizer.json", "dataset_summary.json")
    missing_dataset = [name for name in required_dataset if not (dataset_dir / name).exists()]
    if not (
        (dataset_dir / "train_tokens.npy").exists()
        and (dataset_dir / "val_tokens.npy").exists()
        or (dataset_dir / "train_tokens.json").exists()
        and (dataset_dir / "val_tokens.json").exists()
    ):
        missing_dataset.append("train_tokens.(npy/json), val_tokens.(npy/json)")
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
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            if should_stop and should_stop():
                raise RuntimeError("Dataset preview stopped by user.")
            digest.update(chunk)
    return digest.hexdigest()


def _has_prepared_token_artifacts(dataset_dir: Path) -> bool:
    if not (dataset_dir / "tokenizer.json").exists():
        return False
    has_npy_tokens = (dataset_dir / "train_tokens.npy").exists() and (dataset_dir / "val_tokens.npy").exists()
    has_json_tokens = (dataset_dir / "train_tokens.json").exists() and (dataset_dir / "val_tokens.json").exists()
    return has_npy_tokens or has_json_tokens


def _preview_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    normalized = re.sub(r"\d+", "0", normalized)
    return hashlib.sha1(normalized[:4000].encode("utf-8", errors="ignore")).hexdigest()


def _bad_extraction_reasons(path: Path, preview: Optional[dict[str, str]], size: int) -> list[str]:
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
    prepared = _has_prepared_token_artifacts(config.output_dir)
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
    preview_cache_path = config.output_dir / "preview_scan_cache.json"
    preview_cache = read_json(preview_cache_path, default={}) or {}
    cached_files = preview_cache.get("files") if isinstance(preview_cache.get("files"), dict) else {}
    updated_cache: dict[str, Any] = {}
    readable = 0
    scan_limit = min(len(paths), max(sample_limit * 8, 80))
    for path in paths[:scan_limit]:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        try:
            stat = path.stat()
            size = stat.st_size
            mtime_ns = stat.st_mtime_ns
        except OSError:
            size = 0
            mtime_ns = 0
        cache_key = str(path.resolve())
        cached_entry = cached_files.get(cache_key, {}) if isinstance(cached_files, dict) else {}
        cached_preview = cached_entry.get("preview")
        cached_reasons = cached_entry.get("bad_extraction_reasons")
        if (
            cached_entry.get("size") == size
            and cached_entry.get("mtime_ns") == mtime_ns
            and isinstance(cached_preview, dict)
            and isinstance(cached_reasons, list)
        ):
            preview = cached_preview
            reasons = [str(item) for item in cached_reasons]
        else:
            try:
                preview = _preview_supported_document(path, config, should_stop)
            except RuntimeError:
                raise
            except Exception as exc:
                issues.append(f"Could not preview {path.name}: {exc}")
                continue
            reasons = _bad_extraction_reasons(path, preview, size)
        updated_cache[cache_key] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "preview": preview,
            "bad_extraction_reasons": reasons,
            "duplicate_digest": cached_entry.get("duplicate_digest"),
            "duplicate_digest_mode": cached_entry.get("duplicate_digest_mode"),
            "strict_duplicate_digest": cached_entry.get("strict_duplicate_digest"),
        }
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
            cache_key = str(path.resolve())
            entry = updated_cache.get(cache_key)
            try:
                stat = path.stat()
                size = stat.st_size
                mtime_ns = stat.st_mtime_ns
            except OSError:
                continue
            if entry is None:
                cached_entry = cached_files.get(cache_key, {}) if isinstance(cached_files, dict) else {}
                entry = {
                    "size": size,
                    "mtime_ns": mtime_ns,
                    "preview": cached_entry.get("preview"),
                    "bad_extraction_reasons": cached_entry.get("bad_extraction_reasons"),
                    "duplicate_digest": cached_entry.get("duplicate_digest"),
                    "duplicate_digest_mode": cached_entry.get("duplicate_digest_mode"),
                    "strict_duplicate_digest": cached_entry.get("strict_duplicate_digest"),
                }
            digest_mode = "fast" if config.fast_scan_mode else "full"
            if (
                entry.get("size") == size
                and entry.get("mtime_ns") == mtime_ns
                and entry.get("duplicate_digest_mode") == digest_mode
                and isinstance(entry.get("duplicate_digest"), str)
            ):
                digest = str(entry["duplicate_digest"])
            else:
                if config.fast_scan_mode:
                    digest = file_fingerprint(path, fast=True, sample_bytes=config.fast_scan_sample_bytes)
                else:
                    digest = _file_sha256_cancellable(path, should_stop)
                entry["duplicate_digest"] = digest
                entry["duplicate_digest_mode"] = digest_mode
            entry["size"] = size
            entry["mtime_ns"] = mtime_ns
            updated_cache[cache_key] = entry
            exact_hashes.setdefault(digest, []).append(str(path))
    if config.fast_scan_mode and config.strict_duplicate_verification:
        verified_hashes: dict[str, list[str]] = {}
        for fast_group in exact_hashes.values():
            if len(fast_group) < 2:
                continue
            for path_str in fast_group:
                path = Path(path_str)
                cache_key = str(path.resolve())
                entry = updated_cache.get(cache_key, {})
                try:
                    stat = path.stat()
                    size = stat.st_size
                    mtime_ns = stat.st_mtime_ns
                except OSError:
                    continue
                if (
                    entry.get("size") == size
                    and entry.get("mtime_ns") == mtime_ns
                    and isinstance(entry.get("strict_duplicate_digest"), str)
                ):
                    strict_digest = str(entry["strict_duplicate_digest"])
                else:
                    strict_digest = _file_sha256_cancellable(path, should_stop)
                    entry["strict_duplicate_digest"] = strict_digest
                entry["size"] = size
                entry["mtime_ns"] = mtime_ns
                updated_cache[cache_key] = entry
                verified_hashes.setdefault(strict_digest, []).append(path_str)
        exact_hashes = verified_hashes
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
    if config.output_dir.exists():
        write_json(preview_cache_path, {"version": 1, "files": updated_cache})
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


__all__ = [
    "ProjectHealthResult",
    "DatasetPreviewResult",
    "check_project_health",
    "scan_dataset_preview",
]
