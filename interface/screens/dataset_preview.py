from __future__ import annotations

# DatasetPreviewMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class DatasetPreviewMixin:
    @staticmethod
    def _compact_preview_text(text: str, limit: int = 220) -> str:
        """Normalize a training preview into a compact single line.

        Args:
            text: Raw decoded preview text.
            limit: Maximum number of displayed characters.

        Returns:
            Single-line text preview.
        """

        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."

    def preview_dataset(self) -> None:
        """Run a dataset preview and quality scan in the background."""

        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self.dataset_log.append("Previewing dataset...")
        self.project_state.setText("Previewing dataset")
        self._run_task(
            scan_dataset_preview,
            (self._dataset_config_from_ui(),),
            self._dataset_preview_finished,
            self.dataset_log,
            self.dataset_progress,
            with_progress=True,
            button=self.preview_dataset_button,
            stop_button=self.stop_dataset_button,
            busy_text="Previewing Dataset",
            isolate_process=True,
        )

    @Slot(object)
    def _dataset_preview_finished(self, result: Any) -> None:
        """Finish preview UI cleanup even when rendering a result fails."""
        try:
            self._render_dataset_preview_result(result)
        finally:
            # Keep the action available after both successful and malformed
            # worker results; otherwise the spinner can leave it disabled.
            self._clear_button_busy("Preview Dataset")

    def _render_dataset_preview_result(self, result: Any) -> None:
        """Display dataset preview and quality scan results.

        Args:
            result: Dataset preview result.
        """

        self.dataset_progress.setValue(100)
        suffix_text = ", ".join(f"{suffix}: {count}" for suffix, count in
                                result.suffix_counts.items()) or "none"
        self.dataset_log.append("")
        self.dataset_log.append(
            f"Source files: {result.source_file_count:,}; size: {result.total_bytes / (1024 * 1024):.2f} MB")
        self.dataset_log.append(f"File types: {suffix_text}")
        self.dataset_log.append(
            f"Prepared dataset artifacts: {'found' if result.prepared else 'not complete'}")
        self.dataset_log.append(
            f"Duplicate scan: {result.duplicate_count:,} file entries in {len(result.duplicate_groups):,} likely group(s).")
        self.dataset_log.append(
            f"Bad extraction scan: {result.bad_extraction_count:,} suspicious file(s).")
        self.dataset_log.append(
            f"Code/prose balance: {result.balance_label} ({result.code_preview_count:,}/{result.prose_preview_count:,}).")
        self.dataset_log.append(
            f"Training readiness: {result.readiness_label} ({result.readiness_score}/100).")
        for reason in result.readiness_reasons[:8]:
            self.dataset_log.append(f"- {reason}")
        self.dataset_quality_duplicates.setText(
            f"Duplicates: {result.duplicate_count:,}")
        self.dataset_quality_extraction.setText(
            f"Extraction: {result.bad_extraction_count:,} flagged")
        self.dataset_quality_balance.setText(
            f"Balance: {result.balance_label}")
        self.dataset_quality_readiness.setText(
            f"Readiness: {result.readiness_label} {result.readiness_score}/100")
        if result.summary:
            self._update_dataset_quality_report(result.summary)
            # dataset_quality_duplicates is intentionally left alone here:
            # _update_dataset_quality_report() just set it to the block-level
            # duplication percentage from the prepared corpus (the more useful,
            # actionable metric). Re-setting it to result.duplicate_count (a
            # raw duplicate *file* count from the earlier preview scan) would
            # silently discard that and always show the old metric instead.
            self.dataset_quality_extraction.setText(
                f"Extraction: {result.bad_extraction_count:,} flagged")
            self.dataset_quality_balance.setText(
                f"Balance: {result.balance_label}")
            self.dataset_quality_readiness.setText(
                f"Readiness: {result.readiness_label} {result.readiness_score}/100")
            tokens = int(result.summary.get("token_count", 0) or 0)
            vocab = int(result.summary.get("tokenizer_vocab_size", 0) or 0)
            self.dataset_log.append(
                f"Prepared summary: {tokens:,} tokens, vocab {vocab:,}.")
        else:
            self.dataset_quality_samples.setText(
                f"Preview: {len(result.sample_previews):,} shown")
            self.dataset_quality_tokens.setText("Tokens: not prepared")
            self.dataset_quality_windows.setText("Windows: not prepared")
            self.dataset_quality_vocab.setText("Vocab: not prepared")
            self.dataset_quality_code.setText(
                f"Code/prose: {result.code_preview_count:,}/{result.prose_preview_count:,}")
            self.dataset_quality_cache.setText(
                f"Files: {result.source_file_count:,} source")
        if result.duplicate_groups:
            self.dataset_log.append("")
            self.dataset_log.append("Likely duplicates:")
            for group in result.duplicate_groups[:8]:
                self.dataset_log.append(
                    f"- {group.get('type')}: {group.get('count')} file(s)")
                for path in group.get("files", [])[:4]:
                    self.dataset_log.append(f"    {Path(path).name}")
        if result.bad_extraction_files:
            self.dataset_log.append("")
            self.dataset_log.append("Suspicious extraction files:")
            for item in result.bad_extraction_files[:12]:
                self.dataset_log.append(
                    f"- {Path(item.get('path', '')).name}: {item.get('reasons')}")
        suggestions: list[str] = []
        if result.duplicate_groups:
            suggestions.append(
                "Remove or move duplicate files before preparing the final dataset.")
        if result.bad_extraction_files:
            suggestions.append(
                "Replace flagged PDFs with text/source versions, or remove files with bad extraction.")
        if result.balance_label == "Prose heavy" and self.code_training_mode.isChecked():
            suggestions.append(
                "Add real source-code folders or enable source-file inclusion for a stronger coding model.")
        if result.balance_label == "Code heavy":
            suggestions.append(
                "Add README/tutorial/prose explanations if you want the model to explain code well.")
        if result.readiness_label in {"Needs cleanup", "Not ready"}:
            suggestions.append(
                "Run Preview Dataset again after cleanup and only train once readiness improves.")
        if hasattr(self, "dataset_advisor"):
            if suggestions:
                self.dataset_advisor.setPlainText(
                    "\n".join(f"- {suggestion}" for suggestion in suggestions))
            else:
                self.dataset_advisor.setPlainText(
                    "No immediate cleanup suggestions. Dataset looks acceptable for the current preview.")
        if suggestions:
            self.dataset_log.append("")
            self.dataset_log.append("Cleanup suggestions:")
            for suggestion in suggestions:
                self.dataset_log.append(f"- {suggestion}")
        if result.issues:
            self.dataset_quality_warning.setText(
                f"Warnings: {len(result.issues)}")
            self.dataset_log.append("")
            self.dataset_log.append("Quality notes:")
            for issue in result.issues[:12]:
                self.dataset_log.append(f"- {issue}")
        else:
            self.dataset_quality_warning.setText("Warnings: none")
        if result.sample_previews:
            self.dataset_log.append("")
            self.dataset_log.append("Preview samples:")
            for index, sample in enumerate(result.sample_previews, start=1):
                label = sample.get("language") or sample.get("kind") or "text"
                self.dataset_log.append(
                    f"\n[{index}] {Path(sample.get('path', '')).name} ({label}, {sample.get('characters')} chars)")
                self.dataset_log.append(
                    sample.get("preview", "").replace("\n", "\n    ")[:1400])
        self.project_state.setText("Dataset previewed")
