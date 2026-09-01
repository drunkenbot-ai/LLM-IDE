"""Select inference and resume artifacts from engine training summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingArtifactSelection:
    """Separate chat/inference artifacts from resumable training checkpoints."""

    inference_path: Path
    resume_path: Path | None


def select_training_artifacts(
    summary: dict[str, Any],
    fallback_checkpoint: Path,
) -> TrainingArtifactSelection:
    """Choose the engine-recommended inference and resume artifacts."""
    inference_value = (
        summary.get("recommended_checkpoint_path")
        or summary.get("best_checkpoint_path")
        or fallback_checkpoint
    )
    resume_value = summary.get("best_resume_checkpoint_path") or summary.get("resume_checkpoint")
    resume_path = Path(resume_value) if resume_value else None
    if (
        resume_path is not None
        and resume_path.name == "checkpoint_best_val.pt"
        and summary.get("best_resume_checkpoint_path")
    ):
        resume_path = Path(summary["best_resume_checkpoint_path"])
    return TrainingArtifactSelection(
        inference_path=Path(inference_value),
        resume_path=resume_path,
    )
