from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .conversation_datasets import CONVERSATION_DATASET_PRESETS
from .data import Document, SUPPORTED_CODE_SUFFIXES

LOGGER = logging.getLogger(__name__)

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


def _emit(progress: Optional[Callable[[Any], None]], message: str, percent: Optional[int] = None) -> None:
    LOGGER.info(message)
    if progress:
        progress({"message": message, "percent": percent})


def _canonical_corpus_block(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _slugify_category(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "general_prose"


def _mixture_label(category: str) -> str:
    return MIXTURE_LABELS.get(category, category.replace("_", " ").title())


def _category_from_text(value: str) -> Optional[str]:
    tokens = [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]
    for token in tokens:
        if token in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[token]
    return None


def _default_data_category(path: Path) -> Optional[str]:
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
    text_digest = hashlib.sha256(document.text[:4096].encode("utf-8", errors="ignore")).hexdigest()
    key = f"{document.path}|{document.kind}|{document.language or ''}|{len(document.text)}|{text_digest}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _chunk_document_for_mixture(document: Document, chunk_chars: int = MIXTURE_CHUNK_CHARS) -> list[Document]:
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
    chunks: list[Document] = []
    for document in documents:
        chunks.extend(_chunk_document_for_mixture(document))
    return chunks


def _empty_mixture_report(weights: dict[str, float], documents: list[Document], applied: bool, reason: str = "") -> dict[str, Any]:
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
    if strict_total_budget >= total_available_chars:
        # Every category can supply its proportional share — no trimming needed,
        # include all documents so that 100%/100%/… means "use everything".
        _emit(progress, "Dataset mixture: all categories fit within budget, using full corpus.", 49)
        return documents, _empty_mixture_report(clean_weights, documents, applied=False, reason="All data fits within mixture budget.")
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

    for family, items in sorted_groups.items():
        target_chars = family_targets.get(family, 0)
        selected_chars = 0
        for document in items:
            if selected_chars >= target_chars and selected_by_family[family]:
                break
            selected_by_family[family].append(document)
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

__all__ = [
    "MIXTURE_LABELS",
    "DOMAIN_MIXTURE_FAMILIES",
    "AGGREGATE_MIXTURE_FAMILIES",
    "MIXTURE_CHUNK_CHARS",
    "GENERIC_DEFAULT_DATA_FOLDERS",
    "DEFAULT_STAGE_CATEGORY_FOLDERS",
    "CATEGORY_ALIASES",
    "_apply_dataset_mixture",
]
