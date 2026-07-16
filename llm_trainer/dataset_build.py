from __future__ import annotations

import json
import hashlib
import logging
import re
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .config import DatasetConfig, dataclass_to_jsonable
from .conversation_datasets import CONVERSATION_DATASET_PRESETS, \
    dataset_ids_for_stage, load_conversation_documents
from .data import (
    Document,
    SUPPORTED_CODE_SUFFIXES,
    SUPPORTED_TEXT_SUFFIXES,
    document_from_dict,
    document_to_dict,
    expand_code_documents,
    file_fingerprint,
    file_sha256,
    load_structured_json_documents,
    read_supported_document,
    supported_source_paths,
    write_training_corpus,
)
from .lineage import read_json, record_dataset_version, write_json
from .manifest_store import ManifestStore
from .tokenizer import (
    encode_file,
    load_tokenizer,
    save_tokenizer_package,
    train_tokenizer,
    validate_training_tokenizer,
)
from .training import split_tokens

LOGGER = logging.getLogger(__name__)


def _local_structured_dataset_paths(config: DatasetConfig) -> list[
    tuple[Path, str, str]]:
    """Return configured local structured dataset paths.

    Args:
        config: Dataset configuration.

    Returns:
        Tuples of path, document kind, and progress label.
    """

    items: list[tuple[Path, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in [config.conversation_dataset_path,
                 *config.conversation_dataset_paths]:
        if path is None or not str(path).strip():
            continue
        key = ("conversation", str(Path(path)))
        if key not in seen:
            seen.add(key)
            items.append((Path(path), "conversation", "local conversation"))
    for path in [config.instruction_dataset_path,
                 *config.instruction_dataset_paths]:
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


def _emit(progress: Optional[Callable[[Any], None]], message: str,
          percent: Optional[int] = None) -> None:
    """Emit a progress event if a callback is available.

    Args:
        progress: Optional callback for progress dictionaries.
        message: Human-readable progress message.
        percent: Optional progress percentage.
    """

    LOGGER.info(message)
    if progress:
        progress({"message": message, "percent": percent})


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


def _resolve_tokenizer_strategy(config: DatasetConfig, tokenizer_path: Path) -> \
tuple[str, bool]:
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

    strategy, reuse_tokenizer = _resolve_tokenizer_strategy(config,
                                                            tokenizer_path)
    imported = False
    source_path: Optional[str] = None

    if reuse_tokenizer:
        _emit(progress, "Reusing existing dataset tokenizer.json...", 62)
        return load_tokenizer(tokenizer_path), True, imported, source_path

    if strategy == "import_tokenizer":
        if config.tokenizer_path is None:
            raise ValueError(
                "Choose a tokenizer.json file when tokenizer strategy is Import tokenizer.json.")
        import_path = Path(config.tokenizer_path)
        if not import_path.exists():
            raise FileNotFoundError(
                f"Tokenizer import file not found: {import_path}")
        _emit(progress, f"Importing tokenizer from {import_path}...", 62)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        if import_path.resolve() != tokenizer_path.resolve():
            shutil.copy2(import_path, tokenizer_path)
        return load_tokenizer(tokenizer_path), False, True, str(import_path)

    training_mb = corpus_path.stat().st_size / (1024 * 1024)

    _emit(
        progress,
        f"Training tokenizer on the full {training_mb:.1f} MB corpus...",
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
            "conversation_dataset_path": str(
                config.conversation_dataset_path or ""),
            "instruction_dataset_path": str(
                config.instruction_dataset_path or ""),
            "conversation_dataset_paths": [str(path) for path in
                                           config.conversation_dataset_paths],
            "instruction_dataset_paths": [str(path) for path in
                                          config.instruction_dataset_paths],
            "default_data_paths": [str(path) for path in
                                   config.default_data_paths],
        },
        sort_keys=True,
    )


def _bad_extraction_reasons(path: Path, preview: Optional[dict[str, str]],
                            size: int) -> list[str]:
    """Return quality reasons when extracted preview text looks suspicious."""

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
        symbol_ratio = sum(not char.isalnum() for char in visible) / len(
            visible)
        if alpha_ratio < 0.25 and str(preview.get("kind")) != "code":
            reasons.append("low alphabetic text ratio")
        if symbol_ratio > 0.45:
            reasons.append("high symbol/noise ratio")
    if re.search(r"(.)\1{18,}", text):
        reasons.append("long repeated character run")
    if text.count("\ufffd") >= 3 or "Ã" in text[:500]:
        reasons.append("encoding artifacts detected")
    words = re.findall(r"[A-Za-z]{2,}", text)
    if suffix in {".pdf", ".txt", ".md", ".text"} and len(
            set(words)) < 8 and len(text) > 200:
        reasons.append("very low word variety")
    return reasons


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

    manifest_db_path = config.output_dir / "dataset_manifest.sqlite3"
    legacy_manifest_path = config.output_dir / "dataset_manifest.json"
    cache_dir = config.output_dir / "cache" / "documents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = ManifestStore.open(manifest_db_path,
                                  legacy_json_path=legacy_manifest_path)
    key = _cache_key(config)
    force_reprocess = config.prepare_mode == "force_reprocess"

    local_structured_paths = _local_structured_dataset_paths(config)
    selected_default_files = [
        Path(path)
        for path in config.default_data_paths
        if Path(path).exists() and Path(path).is_file()
    ]
    input_dir_resolved = config.input_dir.resolve() if config.input_dir.exists() else None
    default_files_under_input = bool(
        selected_default_files) and input_dir_resolved is not None and all(
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
    seen_source_paths = {path.resolve() for path in source_paths if
                         path.exists()}
    for candidate in selected_default_files:
        if not candidate.exists() or not candidate.is_file():
            _emit(progress, f"Skipped bundled data file: {candidate}")
            continue
        suffix = candidate.suffix.lower()
        if suffix not in SUPPORTED_TEXT_SUFFIXES and suffix not in SUPPORTED_CODE_SUFFIXES and suffix not in {
            ".pdf", ".json", ".jsonl"}:
            _emit(progress,
                  f"Skipped unsupported bundled data file: {candidate.name}")
            continue
        resolved = candidate.resolve()
        if resolved in seen_source_paths:
            continue
        seen_source_paths.add(resolved)
        default_paths.append(candidate)
    if default_paths:
        source_paths.extend(default_paths)
        source_paths = sorted(source_paths)
        _emit(progress,
              f"Bundled starter data enabled: {len(default_paths)} file(s).",
              8)
    _emit(progress,
          f"Found {len(source_paths)} supported files in {config.input_dir}.",
          8)
    documents: list[Any] = []
    cached_count = 0
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for index, path in enumerate(source_paths, start=1):
        if should_stop and should_stop():
            raise RuntimeError("Dataset preparation stopped by user.")
        percent = 10 + int(32 * index / max(len(source_paths), 1))
        stat = path.stat()
        digest = file_fingerprint(path, fast=config.fast_scan_mode,
                                  sample_bytes=config.fast_scan_sample_bytes)
        cache_path = cache_dir / f"{digest}.json"
        manifest_key = str(path.resolve())
        previous = manifest.get(manifest_key) or {}
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
                cached_text = "\n".join(
                    document.text for document in cached_documents)
                cached_extraction_reasons = _bad_extraction_reasons(
                    path,
                    {
                        "path": str(path),
                        "kind": cached_documents[
                            0].kind if cached_documents else "prose",
                        "language": cached_documents[
                            0].language if cached_documents else "",
                        "characters": str(len(cached_text)),
                        "preview": cached_text[:1200],
                    },
                    stat.st_size,
                )
            if cached_extraction_reasons:
                skipped_count += 1
                reason_text = "; ".join(cached_extraction_reasons)
                _emit(progress,
                      f"Skipped cached {path.name}: suspicious PDF extraction ({reason_text}).",
                      percent)
                manifest.upsert(
                    manifest_key,
                    {
                        "path": str(path),
                        "sha256": digest,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "cache_key": key,
                        "status": "skipped_bad_extraction",
                        "reasons": cached_extraction_reasons,
                    },
                    commit=False,
                )
                continue
            documents.extend(cached_documents)
            cached_count += 1
            _emit(progress,
                  f"Reused {path.name} from cache ({len(cached_documents)} sample(s)).",
                  percent)
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
                manifest.upsert(
                    manifest_key,
                    {
                        "path": str(path),
                        "sha256": digest,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "cache_key": key,
                        "status": "failed",
                        "error": str(exc),
                    },
                    commit=False,
                )
                continue
            if source_doc is None:
                skipped_count += 1
                _emit(progress,
                      f"Skipped {path.name}: no readable text found.", percent)
                manifest.upsert(
                    manifest_key,
                    {
                        "path": str(path),
                        "sha256": digest,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "cache_key": key,
                        "status": "skipped_empty",
                    },
                    commit=False,
                )
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
                _emit(progress,
                      f"Skipped {path.name}: suspicious PDF extraction ({reason_text}).",
                      percent)
                manifest.upsert(
                    manifest_key,
                    {
                        "path": str(path),
                        "sha256": digest,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "cache_key": key,
                        "status": "skipped_bad_extraction",
                        "reasons": extraction_reasons,
                    },
                    commit=False,
                )
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
                json.dumps([document_to_dict(doc) for doc in source_documents],
                           ensure_ascii=False),
                encoding="utf-8",
            )
            documents.extend(source_documents)
            processed_count += 1
            _emit(progress,
                  f"Processed {path.name}: {len(source_documents)} sample(s).",
                  percent)

        manifest.upsert(
            manifest_key,
            {
                "path": str(path),
                "sha256": digest,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "cache_key": key,
                "cache_file": str(cache_path.relative_to(config.output_dir)),
                "status": "cached" if can_use_cache else "processed",
            },
            commit=False,
        )

    for local_path, kind, label in local_structured_paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preparation stopped by user.")
        local_path = Path(local_path)
        _emit(progress, f"Loading {label} JSON/JSONL dataset: {local_path}",
              42)
        local_documents = load_structured_json_documents(local_path, kind=kind,
                                                         lowercase=config.lowercase)
        documents.extend(local_documents)
        processed_count += 1
        manifest_key = f"local-{kind}://{local_path.resolve()}"
        manifest.upsert(
            manifest_key,
            {
                "path": str(local_path),
                "kind": kind,
                "sample_count": len(local_documents),
                "cache_key": key,
                "status": "processed",
            },
            commit=False,
        )
        _emit(progress,
              f"Loaded {len(local_documents):,} {kind} sample(s) from {local_path.name}.",
              43)

    if config.conversation_datasets:
        allowed_dataset_ids = set(dataset_ids_for_stage(config.dataset_stage))
        skipped_stage_ids = [dataset_id for dataset_id in
                             config.conversation_datasets if
                             dataset_id not in allowed_dataset_ids]
        selected_dataset_ids = [dataset_id for dataset_id in
                                config.conversation_datasets if
                                dataset_id in allowed_dataset_ids]
        if skipped_stage_ids:
            skipped_labels = [
                CONVERSATION_DATASET_PRESETS[item].label
                for item in skipped_stage_ids
                if item in CONVERSATION_DATASET_PRESETS
            ]
            _emit(progress,
                  f"Skipping dataset(s) not recommended for {config.dataset_stage}: {', '.join(skipped_labels)}.")
        if not selected_dataset_ids:
            _emit(progress,
                  f"No online datasets selected for {config.dataset_stage}; continuing with local sources only.")
            config.conversation_datasets = []
            manifest.set_meta("dataset_config", dataclass_to_jsonable(config),
                              commit=False)
            manifest.set_meta("cache_key", key, commit=False)
            manifest.commit()
            return (
                sorted(documents, key=lambda document: (
                str(document.path), document.kind, document.language or "")),
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
        _emit(progress,
              f"Online training datasets enabled: {', '.join(labels)}.", 8)
        _emit(progress,
              f"Online training datasets will be cached in: {hf_cache_dir}", 8)
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
            manifest.upsert(
                f"hf://{dataset_id}",
                {
                    "path": f"hf://{dataset_id}",
                    "dataset": preset.hf_path if preset else dataset_id,
                    "config_name": preset.config_name if preset else "",
                    "split": preset.split if preset else "",
                    "sample_limit": config.conversation_sample_limit,
                    "cache_key": key,
                    "status": "processed",
                },
                commit=False,
            )
        processed_count += len(selected_dataset_ids)

    manifest.set_meta("dataset_config", dataclass_to_jsonable(config),
                      commit=False)
    manifest.set_meta("cache_key", key, commit=False)
    manifest.commit()
    return (
        sorted(documents, key=lambda document: (
        str(document.path), document.kind, document.language or "")),
        manifest,
        cached_count,
        processed_count,
        skipped_count,
        failed_count,
    )


from .dataset_mixture import (
    _deduplicate_documents,
    _filter_repetitive_documents,
)


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
        raise ValueError(
            "No supported text, PDF, JSONL, or structured JSON documents were found.")
    documents, exact_dedup_report = _deduplicate_documents(documents)
    if exact_dedup_report["removed_documents"]:
        _emit(
            progress,
            f"Removed {exact_dedup_report['removed_documents']:,} exact duplicate extracted document(s).",
            44,
        )
    documents, repetition_filter_report = _filter_repetitive_documents(documents)
    if repetition_filter_report["removed_documents"]:
        _emit(
            progress,
            (
                "Excluded "
                f"{repetition_filter_report['removed_documents']:,} low-diversity document(s) "
                f"({repetition_filter_report['removed_characters']:,} characters) "
                "instead of padding the corpus with repeated templates."
            ),
            45,
        )
    mixture_report = {
        "applied": False,
        "reason": "Dataset mixture disabled",
    }

    # leave documents unchanged
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    if not documents:
        raise ValueError(
            "Dataset mixture selected no documents. Adjust mixture weights and try again.")

    all_text = "\n".join(doc.text for doc in documents)
    character_count = len(all_text)
    unique_words = len({word.lower() for word in all_text.split()})
    suggested_vocab_size = estimate_vocab_size(character_count, unique_words)
    selected_vocab_size = config.vocab_size or suggested_vocab_size
    warning = content_warning(character_count)
    if character_count < 1_000_000:
        low_corpus_message = (
            "Prepared corpus is below 1M characters after quality filtering. "
            "Add licensed, provenance-tracked sources or select an approved online dataset; "
            "the app will not pad training data with synthetic repetition."
        )
        warning = f"{warning} {low_corpus_message}" if warning else low_corpus_message
    code_sample_count = sum(1 for doc in documents if doc.kind == "code")
    conversation_sample_count = sum(
        1 for doc in documents if doc.kind in {"conversation", "instruction"})
    prose_sample_count = sum(1 for doc in documents if
                             doc.kind not in {"code", "conversation",
                                              "instruction"})
    _emit(progress,
          f"Content size: {character_count:,} characters across {len(documents)} files.",
          45)
    if config.code_training_mode:
        _emit(progress,
              f"Code mode: {code_sample_count:,} code samples, {prose_sample_count:,} prose samples.",
              46)
    if conversation_sample_count:
        _emit(progress,
              f"Conversation data: {conversation_sample_count:,} dialogue/instruction samples.",
              46)
    if cached_file_count or processed_file_count:
        _emit(progress,
              f"Cache: reused {cached_file_count:,} file(s), processed {processed_file_count:,} file(s).",
              47)
    if skipped_file_count or failed_file_count:
        _emit(progress,
              f"Quality: skipped {skipped_file_count:,} empty file(s), failed {failed_file_count:,} file(s).",
              48)
    if mixture_report.get("applied") and config.mixture_weights:
        mixture_parts = []
        for key, value in config.mixture_weights.items():
            try:
                numeric_value = float(value or 0.0)
            except (TypeError, ValueError):
                numeric_value = 0.0
            if numeric_value > 0.0:
                mixture_parts.append(
                    f"{key.replace('_', ' ')} {numeric_value:.1f}%"
                )

        mixture_text = ", ".join(mixture_parts)

        if mixture_text:
            _emit(progress, f"Dataset mixture plan: {mixture_text}.", 49)

    if mixture_report.get("applied"):
        for family, row in mixture_report.get("families", {}).items():
            if int(row.get("selected_documents", 0) or 0) > 0 or float(
                    row.get("requested_weight", 0.0) or 0.0) > 0.0:
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
    save_tokenizer_package(tokenizer, tokenizer_path,
                           model_max_length=config.context_length)

    _emit(progress, "Encoding corpus into token IDs...", 78)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    tokens = encode_file(tokenizer, corpus_path, should_stop=should_stop)
    _emit(progress, f"Encoded {len(tokens):,} tokens.", 86)
    token_density = (
                len(tokens) / max(len(corpus_text), 1)) if corpus_text else 0.0
    document_token_lengths = [max(1, int(round(len(doc.text) * token_density)))
                              for doc in documents if doc.text]
    if document_token_lengths:
        sequence_stats = {
            "min": min(document_token_lengths),
            "average": sum(document_token_lengths) / len(
                document_token_lengths),
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
    train_tokens, val_tokens = split_tokens(tokens, config.validation_split)
    train_window_count = max(0, len(train_tokens) - config.context_length)
    val_window_count = max(0, len(val_tokens) - config.context_length)
    _emit(progress,
          f"Training tokens: {len(train_tokens):,}; validation tokens: {len(val_tokens):,}.",
          92)
    _emit(progress,
          f"Training windows: {train_window_count:,}; validation windows: {val_window_count:,}.",
          92)
    np.save(config.output_dir / "train_tokens.npy",
            np.asarray(train_tokens, dtype=np.int32))
    np.save(config.output_dir / "val_tokens.npy",
            np.asarray(val_tokens, dtype=np.int32))
    quality_report = _dataset_quality_report(
        document_count=len(documents),
        token_count=len(tokens),
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
        "token_count": len(tokens),
        "train_token_count": len(train_tokens),
        "val_token_count": len(val_tokens),
        "train_tokens_path": "train_tokens.npy",
        "val_tokens_path": "val_tokens.npy",
        "token_storage_format": "npy",
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
        "conversation_dataset_path": str(
            config.conversation_dataset_path or ""),
        "instruction_dataset_path": str(config.instruction_dataset_path or ""),
        "conversation_dataset_paths": [str(path) for path in
                                       config.conversation_dataset_paths],
        "instruction_dataset_paths": [str(path) for path in
                                      config.instruction_dataset_paths],
        "default_data_paths": [str(path) for path in
                               config.default_data_paths],
        "mixture_weights": config.mixture_weights,
        "mixture_report": mixture_report,
        "exact_duplicate_documents_removed": exact_dedup_report[
            "removed_documents"],
        "exact_duplicate_document_examples": exact_dedup_report["duplicates"],
        "low_diversity_documents_removed": repetition_filter_report["removed_documents"],
        "low_diversity_characters_removed": repetition_filter_report["removed_characters"],
        "low_diversity_duplicate_unit_threshold": repetition_filter_report["threshold"],
        "low_diversity_document_examples": repetition_filter_report["examples"],
        "suggested_vocab_size": suggested_vocab_size,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "warning": warning,
        "source_files": [str(doc.path) for doc in documents[:1000]],
        "source_files_truncated": len(documents) > 1000,
        "cached_file_count": cached_file_count,
        "processed_file_count": processed_file_count,
        "skipped_file_count": skipped_file_count,
        "failed_file_count": failed_file_count,
        "source_file_count": manifest.count(),
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
        "most_repeated_block_count": duplicate_report[
            "most_repeated_block_count"],
        "top_repeated_blocks": duplicate_report["top_repeated_blocks"],
    }
    dataset_version = record_dataset_version(config.output_dir, summary,
                                             manifest)
    write_json(config.output_dir / "dataset_summary.json", summary)
    manifest.close()
    _emit(progress,
          f"Dataset version recorded: {dataset_version['version_id']}.", 98)
    _emit(progress, f"Dataset ready: {config.output_dir}", 100)
    return DatasetBuildResult(
        config.output_dir,
        tokenizer_path,
        len(documents),
        len(tokens),
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


def _text_block_duplicate_report(corpus_text: str,
                                 max_blocks: int = 500_000) -> dict[str, Any]:
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
        raw_block = corpus_text[start: match.start()]
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
    families = sum(1 for count in (
    code_sample_count, prose_sample_count, conversation_sample_count) if
                   count > 0)
    diversity_score = 7.0 * _bounded_ratio(families, 3)
    average_sequence = float(sequence_stats.get("average", 0.0) or 0.0)
    sequence_score = 5.0 * _bounded_ratio(average_sequence, 256)
    penalty = min(20.0, failed_file_count * 3.0 + skipped_file_count * 0.5)
    duplicate_ratio = float(
        duplicate_report.get("duplicate_block_ratio", 0.0) or 0.0)
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
        reasons.append(
            "Vocabulary is small; language coverage may be limited.")
    elif vocab_size > 50_000:
        reasons.append(
            "Vocabulary is large; tiny models may spend capacity on tokens.")
    else:
        reasons.append("Vocabulary size is in a reasonable small-model range.")
    if families >= 2:
        reasons.append("Dataset includes multiple content families.")
    if skipped_file_count or failed_file_count:
        reasons.append(
            f"Extraction skipped {skipped_file_count} file(s) and failed {failed_file_count} file(s).")
    if duplicate_ratio >= 0.5:
        reasons.append(
            "Prepared corpus is heavily repeated; training may memorize instead of generalize.")
    elif duplicate_ratio >= 0.2:
        reasons.append(
            "Prepared corpus has many repeated blocks; add more varied data or deduplicate.")
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


__all__ = [
    "DatasetBuildResult",
    "build_dataset",
    "estimate_vocab_size",
    "content_warning",
]
