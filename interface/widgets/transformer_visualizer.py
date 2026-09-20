"""Transformer architecture visualizer diagram widget."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget


class TransformerVisualizerWidget(QWidget):
    """Custom interactive visualizer showing the Transformer Block, Attention, MLP, and Residuals."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the transformer architecture visualizer widget."""
        super().__init__(parent)
        self.setMinimumSize(260, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        """Default recommended size."""
        return QSize(280, 440)

    def paintEvent(self, event) -> None:
        """Draw the Transformer Block with residual connections matching Reference Image 1."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        # Center column bounds
        block_w = 110.0
        block_h = 28.0
        center_x = width * 0.48
        box_left = center_x - (block_w / 2)

        # Scale coordinates vertically based on available height
        margin_top = 22.0
        available_h = height - margin_top - 20.0
        step = max(34.0, available_h / 9.5)

        # 1. Output label at Top
        painter.setPen(QColor("#f8fafc"))
        font_label = QFont("Segoe UI", 9, QFont.Bold)
        font_pill = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(font_label)
        painter.drawText(QRectF(box_left, margin_top, block_w, 18), Qt.AlignCenter, "Output")

        # Arrow down from Output into Block
        y_cursor = margin_top + 20.0
        painter.setPen(QPen(QColor("#a855f7"), 2))
        painter.drawLine(QPointF(center_x, y_cursor), QPointF(center_x, y_cursor + 14.0))

        # 2. Transformer Block Outer Container (Nx)
        transformer_box_top = y_cursor + 14.0
        transformer_box_h = step * 5.4
        transformer_box_w = block_w + 50.0
        transformer_box_left = center_x - (transformer_box_w / 2) + 12.0
        transformer_rect = QRectF(transformer_box_left, transformer_box_top, transformer_box_w, transformer_box_h)

        # Draw outer dashed/purple container
        painter.setPen(QPen(QColor("#8b5cf6"), 1.5, Qt.SolidLine))
        painter.setBrush(QBrush(QColor("#131522")))
        painter.drawRoundedRect(transformer_rect, 10, 10)

        # "Transformer Block" and "Nx" label on the left
        painter.setPen(QColor("#cbd5e1"))
        painter.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        painter.drawText(QPointF(transformer_box_left - 85.0, transformer_box_top + 30.0), "Transformer")
        painter.drawText(QPointF(transformer_box_left - 85.0, transformer_box_top + 45.0), "Block")
        painter.setPen(QColor("#a855f7"))
        painter.drawText(QPointF(transformer_box_left - 24.0, transformer_box_top + (transformer_box_h / 2)), "Nx")

        # 3. Inside Transformer Block: Layers & Residuals
        # 3a. Add & Norm (Top)
        pill_y1 = transformer_box_top + 12.0
        rect_an1 = QRectF(box_left, pill_y1, block_w, block_h)
        painter.setPen(QPen(QColor("#d97706"), 1.5))
        painter.setBrush(QBrush(QColor("#332415")))
        painter.drawRoundedRect(rect_an1, 6, 6)
        painter.setPen(QColor("#fed7aa"))
        painter.setFont(font_pill)
        painter.drawText(rect_an1, Qt.AlignCenter, "Add & Norm")

        # Arrow down from Add & Norm to Feed Forward
        arrow1_y = pill_y1 + block_h
        arrow1_end = arrow1_y + 12.0
        painter.setPen(QPen(QColor("#0284c7"), 1.8))
        painter.drawLine(QPointF(center_x, arrow1_y), QPointF(center_x, arrow1_end))

        # 3b. Feed Forward
        pill_y2 = arrow1_end
        rect_ff = QRectF(box_left, pill_y2, block_w, block_h)
        painter.setPen(QPen(QColor("#0284c7"), 1.5))
        painter.setBrush(QBrush(QColor("#112436")))
        painter.drawRoundedRect(rect_ff, 6, 6)
        painter.setPen(QColor("#bae6fd"))
        painter.drawText(rect_ff, Qt.AlignCenter, "Feed Forward")

        # Arrow down from Feed Forward to MLP
        arrow2_y = pill_y2 + block_h
        arrow2_end = arrow2_y + 12.0
        painter.setPen(QPen(QColor("#9333ea"), 1.8))
        painter.drawLine(QPointF(center_x, arrow2_y), QPointF(center_x, arrow2_end))

        # 3c. MLP
        pill_y3 = arrow2_end
        rect_mlp = QRectF(box_left, pill_y3, block_w, block_h)
        painter.setPen(QPen(QColor("#9333ea"), 1.5))
        painter.setBrush(QBrush(QColor("#241436")))
        painter.drawRoundedRect(rect_mlp, 6, 6)
        painter.setPen(QColor("#f3e8ff"))
        painter.drawText(rect_mlp, Qt.AlignCenter, "MLP")

        # Arrow down from MLP to Add & Norm 2
        arrow3_y = pill_y3 + block_h
        arrow3_end = arrow3_y + 12.0
        painter.setPen(QPen(QColor("#d97706"), 1.8))
        painter.drawLine(QPointF(center_x, arrow3_y), QPointF(center_x, arrow3_end))

        # 3d. Add & Norm (Bottom)
        pill_y4 = arrow3_end
        rect_an2 = QRectF(box_left, pill_y4, block_w, block_h)
        painter.setPen(QPen(QColor("#d97706"), 1.5))
        painter.setBrush(QBrush(QColor("#332415")))
        painter.drawRoundedRect(rect_an2, 6, 6)
        painter.setPen(QColor("#fed7aa"))
        painter.drawText(rect_an2, Qt.AlignCenter, "Add & Norm")

        # Arrow down from Add & Norm 2 to Attention
        arrow4_y = pill_y4 + block_h
        arrow4_end = arrow4_y + 12.0
        painter.setPen(QPen(QColor("#ea580c"), 1.8))
        painter.drawLine(QPointF(center_x, arrow4_y), QPointF(center_x, arrow4_end))

        # 3e. Attention
        pill_y5 = arrow4_end
        rect_attn = QRectF(box_left, pill_y5, block_w, block_h)
        painter.setPen(QPen(QColor("#ea580c"), 1.5))
        painter.setBrush(QBrush(QColor("#381e10")))
        painter.drawRoundedRect(rect_attn, 6, 6)
        painter.setPen(QColor("#ffedd5"))
        painter.drawText(rect_attn, Qt.AlignCenter, "Attention")

        # Residual Bypass Connections (Curved arrows on the right side)
        residual_x = transformer_box_left + transformer_box_w - 14.0

        # Residual 1: From below Add & Norm 2 up to Add & Norm 1
        path1 = QPainterPath()
        path1.moveTo(center_x + (block_w / 2), pill_y4 + (block_h / 2))
        path1.lineTo(residual_x, pill_y4 + (block_h / 2))
        path1.lineTo(residual_x, pill_y1 + (block_h / 2))
        path1.lineTo(center_x + (block_w / 2), pill_y1 + (block_h / 2))
        painter.setPen(QPen(QColor("#f59e0b"), 1.8))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path1)

        # Residual 2: Around Attention into Add & Norm 2
        path2 = QPainterPath()
        path2.moveTo(center_x + (block_w / 2), pill_y5 + block_h)
        path2.lineTo(residual_x + 6.0, pill_y5 + block_h)
        path2.lineTo(residual_x + 6.0, pill_y4 + (block_h / 2))
        painter.setPen(QPen(QColor("#f97316"), 1.8))
        painter.drawPath(path2)

        # 4. Positional Encoding Node & Input Embed below Transformer Block
        post_y = transformer_box_top + transformer_box_h + 12.0

        # Arrow down from Attention out of Transformer Block
        painter.setPen(QPen(QColor("#94a3b8"), 1.8))
        painter.drawLine(QPointF(center_x, pill_y5 + block_h), QPointF(center_x, post_y))

        # Positional Encoding Node (Circle with +)
        pe_radius = 8.0
        painter.setPen(QPen(QColor("#cbd5e1"), 1.5))
        painter.setBrush(QBrush(QColor("#181b28")))
        painter.drawEllipse(QPointF(center_x, post_y + pe_radius), pe_radius, pe_radius)
        painter.drawLine(QPointF(center_x - 4.0, post_y + pe_radius), QPointF(center_x + 4.0, post_y + pe_radius))
        painter.drawLine(QPointF(center_x, post_y + pe_radius - 4.0), QPointF(center_x, post_y + pe_radius + 4.0))

        # Positional Encoding Label
        painter.setPen(QColor("#94a3b8"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QPointF(center_x + 18.0, post_y + pe_radius + 4.0), "Positional Encoding")

        # Arrow from Positional Encoding to Input Embed
        pe_end_y = post_y + (pe_radius * 2)
        arrow5_end = pe_end_y + 12.0
        painter.setPen(QPen(QColor("#9333ea"), 1.8))
        painter.drawLine(QPointF(center_x, pe_end_y), QPointF(center_x, arrow5_end))

        # 5. Input Embed Pill
        pill_embed_y = arrow5_end
        rect_embed = QRectF(box_left, pill_embed_y, block_w, block_h)
        painter.setPen(QPen(QColor("#9333ea"), 1.5))
        painter.setBrush(QBrush(QColor("#241436")))
        painter.drawRoundedRect(rect_embed, 6, 6)
        painter.setPen(QColor("#f3e8ff"))
        painter.setFont(font_pill)
        painter.drawText(rect_embed, Qt.AlignCenter, "Input Embed")

        # Arrow down to bottom label
        arrow6_y = pill_embed_y + block_h
        painter.setPen(QPen(QColor("#94a3b8"), 1.8))
        painter.drawLine(QPointF(center_x, arrow6_y), QPointF(center_x, arrow6_y + 12.0))

        painter.setPen(QColor("#f8fafc"))
        painter.setFont(font_label)
        painter.drawText(QRectF(box_left, arrow6_y + 12.0, block_w, 18), Qt.AlignCenter, "Output")

        painter.end()
