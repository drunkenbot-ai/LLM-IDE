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
    """
    Apply Dataset Blueprint percentages independently.

    Unlike the previous implementation, percentages are NOT normalized.

    100% means:
        Include every document in that category.

    50% means:
        Include approximately half of the documents from that category.

    Categories never reduce one another.
    """

    original_document_count = len(documents)

    documents = _chunk_documents_for_mixture(documents)

    if len(documents) != original_document_count:
        _emit(
            progress,
            f"Dataset mixture: split {original_document_count:,} source file(s) into "
            f"{len(documents):,} sampling chunk(s).",
            49,
        )

    # ---------------------------------------------------------
    # Group documents by category
    # ---------------------------------------------------------

    families: dict[str, list[Document]] = {}

    for doc in documents:
        family = _document_mixture_family(doc)
        families.setdefault(family, []).append(doc)

    selected_documents: list[Document] = []

    report = {
        "applied": True,
        "reason": "",
        "total_available_documents": len(documents),
        "total_selected_documents": 0,
        "total_available_characters": sum(len(d.text) for d in documents),
        "total_selected_characters": 0,
        "families": {},
    }

    # ---------------------------------------------------------
    # Process each category independently
    # ---------------------------------------------------------

    selected_documents = []

    for family in sorted(families.keys()):

        docs = families[family]

        docs.sort(key=_stable_document_sort_key)

        percentage = float(weights.get(family, 100.0))

        percentage = max(0.0, min(100.0, percentage))

        available_documents = len(docs)
        available_characters = sum(len(d.text) for d in docs)

        if percentage >= 100.0:

            chosen = docs

        elif percentage <= 0.0:

            chosen = []

        else:

            keep = round(available_documents * percentage / 100.0)

            chosen = docs[:keep]

        selected_documents.extend(chosen)

        selected_characters = sum(len(d.text) for d in chosen)

        report["families"][family] = {
            "label": _mixture_label(family),
            "requested_weight": percentage,
            "available_documents": available_documents,
            "available_characters": available_characters,
            "selected_documents": len(chosen),
            "selected_characters": selected_characters,
            "actual_percent": (
                len(chosen) * 100.0 / available_documents
                if available_documents
                else 0.0
            ),
            "dropped_documents": available_documents - len(chosen),
            "dropped_characters": available_characters - selected_characters,
        }

    report["total_selected_documents"] = len(selected_documents)
    report["total_selected_characters"] = sum(
        len(d.text) for d in selected_documents
    )

    _emit(
        progress,
        f"Dataset mixture selected "
        f"{len(selected_documents):,} of {len(documents):,} document chunks.",
        50,
    )

    return selected_documents, report

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
