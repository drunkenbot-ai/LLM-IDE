"""Parameter Weight Breakdown Donut Chart Widget."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget


class ParameterDonutChartWidget(QWidget):
    """Donut chart visualizing Attention, MLP, and Embedding parameter breakdown."""

    def __init__(
        self,
        attention_pct: float = 38.0,
        mlp_pct: float = 41.0,
        embed_pct: float = 21.0,
        total_params_str: str = "20.3B",
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialize the parameter donut chart widget.

        Args:
            attention_pct: Percentage of parameters in attention layers.
            mlp_pct: Percentage of parameters in MLP layers.
            embed_pct: Percentage of parameters in embedding layers.
            total_params_str: Formatted string of total parameters (e.g. 20.3B).
            parent: Optional parent QWidget.
        """
        super().__init__(parent)
        self.attention_pct = attention_pct
        self.mlp_pct = mlp_pct
        self.embed_pct = embed_pct
        self.total_params_str = total_params_str

        # Palette from Reference Image 1
        self.color_attn = QColor("#f59e0b")  # Amber / Gold
        self.color_mlp = QColor("#a855f7")   # Purple / Violet
        self.color_embed = QColor("#64748b") # Slate Gray

        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_breakdown(
        self,
        attention_pct: float,
        mlp_pct: float,
        embed_pct: float,
        total_params_str: str,
    ) -> None:
        """Update breakdown proportions and repaint.

        Args:
            attention_pct: New Attention percentage.
            mlp_pct: New MLP percentage.
            embed_pct: New Embedding percentage.
            total_params_str: New total parameters text.
        """
        self.attention_pct = max(0.0, attention_pct)
        self.mlp_pct = max(0.0, mlp_pct)
        self.embed_pct = max(0.0, embed_pct)
        self.total_params_str = total_params_str
        self.update()

    def sizeHint(self) -> QSize:
        """Default recommended size."""
        return QSize(260, 160)

    def paintEvent(self, event) -> None:
        """Render the donut chart, legend, and total params readout."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        # Left area for donut, right area for legend
        donut_size = min(90.0, height - 40.0)
        donut_x = 16.0
        donut_y = 12.0
        donut_rect = QRectF(donut_x, donut_y, donut_size, donut_size)

        # Calculate angles (in 1/16th degrees)
        total = self.attention_pct + self.mlp_pct + self.embed_pct
        if total <= 0:
            total = 1.0

        span_attn = int((self.attention_pct / total) * 360 * 16)
        span_mlp = int((self.mlp_pct / total) * 360 * 16)
        span_embed = int(360 * 16) - span_attn - span_mlp

        pen_width = max(14.0, donut_size * 0.22)
        inner_rect = donut_rect.adjusted(pen_width / 2, pen_width / 2, -pen_width / 2, -pen_width / 2)

        start_angle = 90 * 16

        # Draw Slices with pen
        painter.setBrush(Qt.NoBrush)

        # 1. Attention Slice (Amber)
        painter.setPen(QPen(self.color_attn, pen_width, Qt.SolidLine, Qt.FlatCap))
        painter.drawArc(inner_rect, start_angle, -span_attn)

        # 2. MLP Slice (Purple)
        painter.setPen(QPen(self.color_mlp, pen_width, Qt.SolidLine, Qt.FlatCap))
        painter.drawArc(inner_rect, start_angle - span_attn, -span_mlp)

        # 3. Embedding Slice (Gray)
        painter.setPen(QPen(self.color_embed, pen_width, Qt.SolidLine, Qt.FlatCap))
        painter.drawArc(inner_rect, start_angle - span_attn - span_mlp, -span_embed)

        # 4. Legend on Right
        legend_x = donut_x + donut_size + 24.0
        legend_y = donut_y + 6.0

        legend_items = [
            ("Attention", f"{self.attention_pct:.0f}%", self.color_attn),
            ("MLP", f"{self.mlp_pct:.0f}%", self.color_mlp),
            ("Embedding", f"{self.embed_pct:.0f}%", self.color_embed),
        ]

        font_label = QFont("Segoe UI", 8, QFont.Medium)
        font_val = QFont("Segoe UI", 8, QFont.Bold)

        for i, (name, val, col) in enumerate(legend_items):
            row_y = legend_y + (i * 18.0)

            # Bullet dot
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(col))
            painter.drawEllipse(QRectF(legend_x, row_y + 3.0, 7.0, 7.0))

            # Name label
            painter.setPen(QColor("#cbd5e1"))
            painter.setFont(font_label)
            painter.drawText(QPointF(legend_x + 14.0, row_y + 10.0), name)

            # Value percentage
            painter.setFont(font_val)
            painter.drawText(QPointF(legend_x + 85.0, row_y + 10.0), val)

        # 5. Total Params Readout
        total_y = legend_y + 62.0
        painter.setPen(QColor("#f8fafc"))
        total_font = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(total_font)
        painter.drawText(QPointF(legend_x, total_y), f"Total Params: {self.total_params_str}")

        # 6. Mini 12-Bar Layer Depth Distribution Chart (Matching Concept Image 4)
        bars_y = total_y + 8.0
        bar_w = 6.0
        bar_spacing = 3.0
        bar_heights = [8.0, 12.0, 14.0, 11.0, 15.0, 16.0, 13.0, 15.0, 12.0, 14.0, 10.0, 6.0]
        bar_colors = [
            self.color_attn, self.color_attn, self.color_mlp, self.color_mlp,
            self.color_mlp, self.color_mlp, self.color_attn, self.color_mlp,
            self.color_mlp, self.color_attn, self.color_mlp, self.color_embed
        ]

        painter.setPen(Qt.NoPen)
        for idx, (bh, col) in enumerate(zip(bar_heights, bar_colors)):
            bx = legend_x + (idx * (bar_w + bar_spacing))
            by = bars_y + (16.0 - bh)
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(QRectF(bx, by, bar_w, bh), 1.5, 1.5)

        painter.end()
