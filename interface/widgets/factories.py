from __future__ import annotations

# WidgetFactoryMixin mixin. Shared runtime names are provided by interface.app.
from typing import Any, Optional, Union  # noqa: F401
from interface import app as _app

globals().update({name: value for name, value in vars(_app).items() if not name.startswith("__")})


class WidgetFactoryMixin:
    def _panel(self) -> QWidget:
        """Create a base page panel.

        Returns:
            Panel widget.
        """

        page = QWidget()
        page.setObjectName("Panel")
        return page

    def _page_title(self, text: str) -> QLabel:
        """Create a page title label.

        Args:
            text: Title text.

        Returns:
            Label configured as a page title.
        """

        label = QLabel(text)
        label.setObjectName("PageTitle")
        return label

    def _metric_chip(self, text: str, tooltip: str) -> QLabel:
        """Create a compact metric display label.

        Args:
            text: Initial metric text.
            tooltip: User-facing explanation.

        Returns:
            Configured metric label.
        """

        label = QLabel(text)
        label.setObjectName("MetricChip")
        label.setMinimumWidth(150)
        label.setMinimumHeight(28)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(label, tooltip)
        return label

    def _hardware_meter(self, name: str) -> QProgressBar:
        """Create a slider-like hardware utilization meter.

        Args:
            name: Display name for the meter.

        Returns:
            Configured progress bar.
        """

        meter = QProgressBar()
        meter.setObjectName("HardwareMeter")
        meter.setRange(0, 100)
        meter.setValue(0)
        meter.setTextVisible(False)
        meter.setFixedHeight(8)
        meter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(meter, f"Live {name} utilization.")
        return meter

    def _set_meter(self, meter: QProgressBar, name: str, value: Optional[float]) -> None:
        """Update a hardware utilization meter.

        Args:
            meter: Meter to update.
            name: Display name for the meter.
            value: Utilization percentage.
        """

        if value is None:
            meter.setValue(0)
            label = self.hardware_meter_labels.get(id(meter))
            if label is not None:
                label.setText(f"{name}: -")
            return
        bounded = max(0.0, min(100.0, float(value)))
        meter.setValue(int(round(bounded)))
        label = self.hardware_meter_labels.get(id(meter))
        if label is not None:
            label.setText(f"{name}: {bounded:.1f}%")

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

    def _tip(self, widget: QWidget, text: str) -> None:
        """Attach tooltip and status tip text.

        Args:
            widget: Widget receiving the tip.
            text: Tooltip text.
        """

        widget.setToolTip(text)
        widget.setStatusTip(text)

    def _browse(self, field: QLineEdit, directory: bool, file_filter: str = "Checkpoints (*.pt)") -> None:
        """Open a file or folder picker for a path field.

        Args:
            field: Path input to update.
            directory: Whether to select a folder instead of a file.
            file_filter: File dialog filter used for files.
        """

        start_dir = self._browse_start_dir(field, directory)
        if directory:
            value = QFileDialog.getExistingDirectory(self, "Choose folder", start_dir)
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Choose file", start_dir, file_filter)
        if value:
            field.setText(value)

    def _browse_multiple_files(self, field: QLineEdit, file_filter: str) -> None:
        """Open a multi-file picker and write selected paths to a field.

        Args:
            field: Path field to update.
            file_filter: File dialog filter.
        """

        values, _ = QFileDialog.getOpenFileNames(self, "Choose files", self._browse_start_dir(field, False), file_filter)
        if values:
            field.setText("; ".join(values))

    def _browse_start_dir(self, field: QLineEdit, directory: bool) -> str:
        """Return the best initial folder for a browse dialog.

        Args:
            field: Path field being browsed.
            directory: Whether the dialog selects a folder.

        Returns:
            Existing field path, active project folder, or current folder.
        """

        text = field.text().strip()
        if text:
            path = Path(text)
            if path.exists():
                if path.is_dir():
                    return str(path)
                return str(path.parent)
            parent = path if directory else path.parent
            if parent.exists():
                return str(parent)
        if self.current_project_file is not None:
            return str(self.current_project_file.parent)
        return str(Path.cwd())
