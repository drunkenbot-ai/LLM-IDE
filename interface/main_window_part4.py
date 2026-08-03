from __future__ import annotations

# MainWindow implementation mixin. Runtime names are provided by app.py.
from typing import Any
from . import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})

class MainWindowPart4:
    def _update_dataset_quality_report(self, summary: dict[str, Any]) -> None:
        """Update dataset quality chips from a summary dictionary.

        Args:
            summary: Dataset summary fields.
        """

        document_count = int(summary.get("document_count", 0) or 0)
        token_count = int(summary.get("token_count", 0) or 0)
        train_window_count = int(summary.get("train_window_count", 0) or 0)
        val_window_count = int(summary.get("val_window_count", 0) or 0)
        character_count = int(summary.get("character_count", 0) or 0)
        vocab_size = int(summary.get("tokenizer_vocab_size", summary.get("vocab_size", 0)) or 0)
        code_count = int(summary.get("code_sample_count", 0) or 0)
        prose_count = int(summary.get("prose_sample_count", 0) or 0)
        conversation_count = int(summary.get("conversation_sample_count", 0) or 0)
        cached_count = int(summary.get("cached_file_count", 0) or 0)
        processed_count = int(summary.get("processed_file_count", 0) or 0)
        skipped_count = int(summary.get("skipped_file_count", 0) or 0)
        failed_count = int(summary.get("failed_file_count", 0) or 0)
        warning = str(summary.get("warning") or "none")
        sequence_stats = summary.get("sequence_token_stats", {}) or {}
        quality_score = float(summary.get("quality_score", 0.0) or 0.0)
        quality_stars = float(summary.get("quality_stars", 0.0) or 0.0)
        quality_label = str(summary.get("quality_label") or "")
        corpus_block_count = int(summary.get("corpus_block_count", 0) or 0)
        unique_block_count = int(summary.get("unique_block_count", 0) or 0)
        duplicate_block_count = int(summary.get("duplicate_block_count", 0) or 0)
        duplicate_block_ratio = float(summary.get("duplicate_block_ratio", 0.0) or 0.0)
        if not quality_score and (token_count or train_window_count or vocab_size):
            quality_score, quality_stars, quality_label = self._estimate_dataset_rating(
                token_count,
                vocab_size,
                train_window_count,
                val_window_count,
                document_count,
                code_count,
                prose_count,
                conversation_count,
                skipped_count,
                failed_count,
                warning,
                sequence_stats,
            )
        self.dataset_quality_samples.setText(f"Documents: {document_count:,}")
        self.dataset_quality_tokens.setText(f"Tokens: {token_count:,}")
        if train_window_count or val_window_count:
            self.dataset_quality_windows.setText(f"Windows: {train_window_count:,}/{val_window_count:,}")
        else:
            self.dataset_quality_windows.setText("Windows: -")
        self.dataset_quality_vocab.setText(f"Vocab: {vocab_size:,}" if vocab_size else "Vocab: -")
        self.dataset_quality_rating.setText(
            f"Rating: {self._star_text(quality_stars)} {quality_stars:.1f}/5"
            if quality_stars
            else "Rating: -"
        )
        self.dataset_quality_code.setText(f"Code/prose/chat: {code_count:,}/{prose_count:,}/{conversation_count:,}")
        self.dataset_quality_balance.setText("Balance: prepared")
        self.dataset_quality_readiness.setText("Readiness: preview needed")
        self.dataset_quality_cache.setText(f"Files: {processed_count:,} ok, {cached_count:,} cached, {skipped_count:,} skipped, {failed_count:,} failed")
        if corpus_block_count:
            self.dataset_quality_duplicates.setText(f"Duplicates: {duplicate_block_ratio * 100:.1f}%")
            self._tip(
                self.dataset_quality_duplicates,
                (
                    f"{duplicate_block_count:,} repeated blocks out of {corpus_block_count:,}; "
                    f"{unique_block_count:,} unique blocks."
                ),
            )
        else:
            self.dataset_quality_duplicates.setText("Duplicates: -")
        self.dataset_quality_warning.setText(f"Warnings: {warning}")
        self._tip(self.dataset_quality_samples, f"{character_count:,} source characters across prepared documents.")
        if quality_stars:
            self._tip(
                self.dataset_quality_rating,
                f"{quality_label or 'Rated'} dataset: {quality_score:.1f}/100. Higher scores usually mean more usable tokens, richer vocabulary, more windows, and fewer extraction issues.",
            )
        self._tip(
            self.dataset_quality_windows,
            f"{train_window_count:,} training and {val_window_count:,} validation sliding windows.",
        )
        self._update_dataset_stat_charts(summary, code_count, prose_count, conversation_count, sequence_stats)
        if hasattr(self, "dataset_advisor") and (train_window_count or val_window_count):
            advice = [
                "Documents are source items. Windows are the actual context slices used by training.",
                f"This dataset can provide about {train_window_count:,} training windows and {val_window_count:,} validation windows.",
            ]
            if sequence_stats:
                advice.append(
                    "Approx token distribution per source: "
                    f"min {int(sequence_stats.get('min', 0) or 0):,}, "
                    f"avg {float(sequence_stats.get('average', 0.0) or 0.0):,.0f}, "
                    f"median {float(sequence_stats.get('median', 0.0) or 0.0):,.0f}, "
                    f"max {int(sequence_stats.get('max', 0) or 0):,}."
                )
            if document_count < 100 and train_window_count >= 10_000:
                advice.append(
                    "A low document count can still be useful when each document is long, because the trainer samples many overlapping windows."
                )
            if train_window_count < 1_000:
                advice.append("Add more text or lower context length if training looks repetitive.")
            if corpus_block_count:
                advice.append(
                    f"Block diversity: {unique_block_count:,}/{corpus_block_count:,} unique blocks "
                    f"({duplicate_block_ratio * 100:.1f}% repeated)."
                )
            if quality_stars:
                advice.append(f"Dataset rating: {quality_stars:.1f}/5 stars ({quality_label or 'rated'}, score {quality_score:.1f}/100).")
                for reason in list(summary.get("quality_reasons", []) or [])[:4]:
                    advice.append(f"- {reason}")
            self.dataset_advisor.setPlainText("\n".join(advice))

    def _star_text(self, stars: float) -> str:
        """Return a compact five-star display string.

        Args:
            stars: Rating from zero to five.

        Returns:
            Unicode star display with rounded whole stars.
        """

        whole = max(0, min(5, int(round(float(stars)))))
        return "*" * whole + "-" * (5 - whole)

    def _estimate_dataset_rating(
        self,
        token_count: int,
        vocab_size: int,
        train_window_count: int,
        val_window_count: int,
        document_count: int,
        code_count: int,
        prose_count: int,
        conversation_count: int,
        skipped_count: int,
        failed_count: int,
        warning: str,
        sequence_stats: dict[str, Any],
    ) -> tuple[float, float, str]:
        """Estimate a dataset rating for older summaries that lack saved quality fields.

        Args:
            token_count: Total prepared token count.
            vocab_size: Tokenizer vocabulary size.
            train_window_count: Number of training windows.
            val_window_count: Number of validation windows.
            document_count: Number of source documents.
            code_count: Code sample count.
            prose_count: Prose sample count.
            conversation_count: Conversation/instruction sample count.
            skipped_count: Skipped source file count.
            failed_count: Failed source file count.
            warning: Dataset warning text.
            sequence_stats: Approximate source sequence statistics.

        Returns:
            Score, stars, and label.
        """

        def ratio(value: float, target: float) -> float:
            return max(0.0, min(1.0, float(value) / float(target))) if target > 0 else 0.0

        families = sum(1 for count in (code_count, prose_count, conversation_count) if count > 0)
        score = (
            30.0 * ratio(token_count, 1_000_000)
            + 20.0 * ratio(train_window_count, 50_000)
            + 18.0 * ratio(vocab_size, 8_000)
            + 12.0 * ratio(document_count, 1_000)
            + 8.0 * ratio(val_window_count, 2_000)
            + 7.0 * ratio(families, 3)
            + 5.0 * ratio(float(sequence_stats.get("average", 0.0) or 0.0), 256)
        )
        score -= min(20.0, failed_count * 3.0 + skipped_count * 0.5)
        if warning and warning != "none":
            score -= 5.0
        score = max(0.0, min(100.0, score))
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
        return score, stars, label

    def _update_dataset_stat_charts(
        self,
        summary: dict[str, Any],
        code_count: int,
        prose_count: int,
        conversation_count: int,
        sequence_stats: dict[str, Any],
    ) -> None:
        """Update dataset statistics charts.

        Args:
            summary: Dataset summary fields.
            code_count: Number of code samples.
            prose_count: Number of prose samples.
            conversation_count: Number of conversation or instruction samples.
            sequence_stats: Approximate token distribution statistics.
        """

        if not hasattr(self, "dataset_mix_chart"):
            return
        mixture_report = summary.get("mixture_report", {}) or {}
        family_rows = list((mixture_report.get("families", {}) or {}).values())
        labels: list[str] = []
        values: list[float] = []
        for row in family_rows:
            actual = float(row.get("actual_percent", 0.0) or 0.0)
            selected = int(row.get("selected_documents", 0) or 0)
            if actual > 0.0 or selected > 0:
                labels.append(str(row.get("label") or "source"))
                values.append(actual)
        if not labels:
            total = max(code_count + prose_count + conversation_count, 1)
            labels = ["Code", "Prose", "Conversation"]
            values = [
                code_count * 100.0 / total,
                prose_count * 100.0 / total,
                conversation_count * 100.0 / total,
            ]
        self.dataset_mix_chart.set_values(labels, values, "%")
        if sequence_stats:
            self.dataset_sequence_chart.set_values(
                ["Min", "Average", "Median", "Max"],
                [
                    float(sequence_stats.get("min", 0) or 0),
                    float(sequence_stats.get("average", 0.0) or 0.0),
                    float(sequence_stats.get("median", 0.0) or 0.0),
                    float(sequence_stats.get("max", 0) or 0),
                ],
            )
        else:
            self.dataset_sequence_chart.clear()

    def _reset_dataset_quality_report(self) -> None:
        """Reset dataset quality chips to their empty state."""

        self.dataset_quality_samples.setText("Documents: -")
        self.dataset_quality_tokens.setText("Tokens: -")
        self.dataset_quality_windows.setText("Windows: -")
        self.dataset_quality_vocab.setText("Vocab: -")
        self.dataset_quality_rating.setText("Rating: -")
        self.dataset_quality_code.setText("Code/prose: -")
        self.dataset_quality_balance.setText("Balance: -")
        self.dataset_quality_readiness.setText("Readiness: -")
        self.dataset_quality_cache.setText("Cache: -")
        self.dataset_quality_duplicates.setText("Duplicates: -")
        self.dataset_quality_extraction.setText("Extraction: -")
        self.dataset_quality_warning.setText("Warnings: none")
        if hasattr(self, "dataset_mix_chart"):
            self.dataset_mix_chart.clear()
        if hasattr(self, "dataset_sequence_chart"):
            self.dataset_sequence_chart.clear()
        if hasattr(self, "dataset_advisor"):
            self.dataset_advisor.setPlainText("Run Preview Dataset to get cleanup suggestions.")

    def _card(self, title: str, content_layout: Union[QVBoxLayout, QFormLayout, QGridLayout, QHBoxLayout]) -> QWidget:
        """Create a neon module card.

        Args:
            title: Card heading.
            content_layout: Layout to place inside the card.

        Returns:
            Card widget.
        """

        card = QWidget()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("SectionLabel")
        layout.addWidget(title_label)
        layout.addLayout(content_layout)
        return card

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        """Create a bounded integer input.

        Args:
            minimum: Minimum value.
            maximum: Maximum value.
            value: Initial value.

        Returns:
            Configured spin box.
        """

        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setMaximumWidth(220)
        return spin

    def _double_spin(self, minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        """Create a bounded float input.

        Args:
            minimum: Minimum value.
            maximum: Maximum value.
            value: Initial value.
            step: Increment step.
            decimals: Number of displayed decimal places.

        Returns:
            Configured double spin box.
        """

        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setMaximumWidth(220)
        return spin

    def _path_row(self, field: QLineEdit, directory: bool = True, file_filter: str = "Checkpoints (*.pt)") -> QWidget:
        """Create a path field with a browse button.

        Args:
            field: Path input widget.
            directory: Whether the browse dialog selects folders.
            file_filter: File dialog filter used when ``directory`` is false.

        Returns:
            Row widget containing the path input and button.
        """

        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        browse = QPushButton("Browse")
        browse.setFixedWidth(88)
        self._tip(browse, "Open a file/folder picker for this path.")
        field.setMinimumWidth(180)
        field.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        browse.clicked.connect(lambda: self._browse(field, directory, file_filter))
        layout.addWidget(field, 1)
        layout.addWidget(browse)
        return row

    def _multi_file_path_row(self, field: QLineEdit, file_filter: str = "All files (*)") -> QWidget:
        """Create a path field with a multi-file browse button.

        Args:
            field: Path input widget. Multiple paths are separated with semicolons.
            file_filter: File dialog filter.

        Returns:
            Row widget containing the path input and browse button.
        """

        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        browse = QPushButton("Browse")
        browse.setFixedWidth(88)
        self._tip(browse, "Choose one or more JSON/JSONL files. You can also paste a folder path.")
        field.setMinimumWidth(180)
        field.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        browse.clicked.connect(lambda: self._browse_multiple_files(field, file_filter))
        layout.addWidget(field, 1)
        layout.addWidget(browse)
        return row

    def _configure_form(self, form: QFormLayout) -> None:
        """Apply common form spacing and growth policy.

        Args:
            form: Form layout to configure.
        """

        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)

    def _configure_device_options(self) -> None:
        """Populate training device choices without duplicate CPU entries."""

        self.device.clear()
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            self.device.addItem("cuda")
            self.device.addItem("cpu")
            self.device_info.setText(f"CUDA ready: {device_name}")
            self.use_amp_default = True
        else:
            self.device.addItem("cpu")
            cuda_build = getattr(torch.backends, "cuda", None)
            built_with_cuda = bool(cuda_build and torch.backends.cuda.is_built())
            if built_with_cuda:
                detail = "CUDA build found, but no usable NVIDIA GPU/driver was detected."
            else:
                detail = "CUDA is not available in this PyTorch install."
            self.device_info.setText(detail)
            self.use_amp_default = False

    def _thin_progress(self) -> QProgressBar:
        """Create a thin bottom progress bar.

        Returns:
            Configured progress bar.
        """

        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setTextVisible(False)
        progress.setFixedHeight(4)
        progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(progress, "Progress indicator for the current page operation.")
        return progress

