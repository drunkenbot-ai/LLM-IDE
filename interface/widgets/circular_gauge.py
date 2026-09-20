"""Circular radial gauge widget for hardware and compute telemetry."""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QConicalGradient,
    QFont,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget


class CircularGaugeWidget(QWidget):
    """Circular radial meter with customizable range, value, unit, and accent glow."""

    def __init__(
        self,
        title: str = "",
        value_text: str = "0",
        unit_text: str = "",
        percent: float = 0.0,
        accent_color: str = "#06b6d4",
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize the circular gauge.

        Args:
            title: Metric title displayed above or below the gauge.
            value_text: Main formatted readout text in center.
            unit_text: Sub-label below readout (e.g. 'NVIDIA', 'GB', '°C').
            percent: Percentage from 0.0 to 100.0 for arc length.
            accent_color: Primary neon color for the active arc.
            parent: Optional parent QWidget.
        """
        super().__init__(parent)
        self.title = title
        self.value_text = value_text
        self.unit_text = unit_text
        self.percent = max(0.0, min(100.0, percent))
        self.accent_color = QColor(accent_color)
        self.setMinimumSize(110, 110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(
        self,
        value_text: str,
        percent: float,
        unit_text: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """Update gauge values and trigger repaint.

        Args:
            value_text: Formatted string to display.
            percent: Completion percentage (0 to 100).
            unit_text: Optional updated unit string.
            title: Optional updated title string.
        """
        self.value_text = value_text
        self.percent = max(0.0, min(100.0, percent))
        if unit_text is not None:
            self.unit_text = unit_text
        if title is not None:
            self.title = title
        self.update()

    def sizeHint(self) -> QSize:
        """Provide default size recommendation."""
        return QSize(120, 120)

    def paintEvent(self, event) -> None:
        """Render the circular radial gauge with antialiasing."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        side = min(width, height) - 16
        if side <= 20:
            return

        x = (width - side) / 2
        y = (height - side) / 2
        rect = QRectF(x, y, side, side)

        # Gauge arc geometry: 240 degrees total, starting at 150 deg (5 o'clock clockwise to 7 o'clock)
        start_angle = 225
        span_angle_total = 270

        pen_width = max(7.0, side * 0.08)

        # 1. Background Groove Arc
        groove_pen = QPen(QColor("#1e2433"), pen_width, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(groove_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawArc(rect, int(start_angle * 16), int(-span_angle_total * 16))

        # 2. Active Accent Arc
        if self.percent > 0:
            active_span = -(span_angle_total * (self.percent / 100.0))
            active_pen = QPen(self.accent_color, pen_width, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(active_pen)
            painter.drawArc(rect, int(start_angle * 16), int(active_span * 16))

        # 3. Text In Center
        center_x = width / 2
        center_y = height / 2

        # Title at Top
        if self.title:
            painter.setPen(QColor("#94a3b8"))
            title_font = QFont("Segoe UI", max(8, int(side * 0.085)), QFont.Bold)
            painter.setFont(title_font)
            title_rect = QRectF(0, y - 2, width, side * 0.2)
            painter.drawText(title_rect, Qt.AlignCenter, self.title)

        # Center Main Value Text
        painter.setPen(QColor("#f8fafc"))
        val_font = QFont("Segoe UI", max(10, int(side * 0.13)), QFont.Bold)
        painter.setFont(val_font)
        val_rect = QRectF(0, center_y - (side * 0.14), width, side * 0.28)
        painter.drawText(val_rect, Qt.AlignCenter, self.value_text)

        # Unit / Sub-Label
        if self.unit_text:
            # If unit is NVIDIA or similar, color green
            color = QColor("#10b981") if "NVIDIA" in self.unit_text.upper() else QColor("#94a3b8")
            painter.setPen(color)
            unit_font = QFont("Segoe UI", max(7, int(side * 0.085)), QFont.Bold)
            painter.setFont(unit_font)
            unit_rect = QRectF(0, center_y + (side * 0.10), width, side * 0.22)
            painter.drawText(unit_rect, Qt.AlignCenter, self.unit_text)

        painter.end()
