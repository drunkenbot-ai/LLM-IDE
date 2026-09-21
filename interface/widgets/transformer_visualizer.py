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
    """Interactive visualizer dynamically displaying model transformer architecture blocks."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the transformer architecture visualizer widget."""
        super().__init__(parent)
        self.setMinimumSize(260, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.model_name = "GPT-NeoX-20B"
        self.norm_type = "RMSNorm"
        self.activation_fn = "SwiGLU"
        self.attention_type = "Multi-Head (MHA)"
        self.pos_encoding = "RoPE"
        self.use_bias = False

    def sizeHint(self) -> QSize:
        """Default recommended size."""
        return QSize(280, 440)

    def set_architecture(
        self,
        model_name: str = "GPT-NeoX-20B",
        norm_type: str = "RMSNorm",
        activation: str = "SwiGLU",
        attention_type: str = "Multi-Head (MHA)",
        pos_encoding: str = "RoPE",
        use_bias: bool = False,
    ) -> None:
        """Update visualizer elements based on the selected model architecture."""
        self.model_name = model_name
        self.norm_type = norm_type
        self.activation_fn = activation
        self.attention_type = attention_type
        self.pos_encoding = pos_encoding
        self.use_bias = use_bias
        self.update()

    def paintEvent(self, event) -> None:
        """Draw the Transformer Block with residual connections matching current model specs."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        # Center column bounds
        block_w = 124.0
        block_h = 28.0
        center_x = width * 0.48
        box_left = center_x - (block_w / 2)

        # Scale coordinates vertically based on available height
        margin_top = 22.0
        available_h = height - margin_top - 20.0
        step = max(34.0, available_h / 9.5)

        is_llama = "llama" in self.model_name.lower() or "mistral" in self.model_name.lower()
        is_neox = "neox" in self.model_name.lower() or "pythia" in self.model_name.lower()

        # Dynamic labels
        top_norm_text = f"Add & {self.norm_type}" if self.norm_type else "Add & Norm"
        bottom_norm_text = f"Add & {self.norm_type}" if self.norm_type else "Add & Norm"

        if is_llama:
            block_title = "Llama-3 Block" if "llama" in self.model_name.lower() else "Mistral Block"
            ff_text = f"{self.activation_fn} Gate" if self.activation_fn else "SwiGLU Gate"
            mlp_text = "Up & Down Proj"
            attn_text = "GQA Attention" if "GQA" in self.attention_type or "Grouped" in self.attention_type else "MHA Attention"
            pe_text = "Rotary Embed (RoPE)"
            border_color = QColor("#8b5cf6")
            norm_bg = QColor("#1e1b4b")
            norm_pen = QColor("#6366f1")
            norm_fg = QColor("#c7d2fe")
        elif is_neox:
            block_title = "GPT-NeoX Block"
            ff_text = "Feed Forward"
            mlp_text = f"{self.activation_fn} MLP" if self.activation_fn else "GELU MLP"
            attn_text = "Parallel Attention"
            pe_text = "Rotary Pos (RoPE)"
            border_color = QColor("#ec4899")
            norm_bg = QColor("#332415")
            norm_pen = QColor("#d97706")
            norm_fg = QColor("#fed7aa")
        else:
            block_title = "Transformer Block"
            ff_text = "Feed Forward"
            mlp_text = f"{self.activation_fn} MLP" if self.activation_fn else "MLP"
            attn_text = "Self Attention"
            pe_text = "Positional Encoding"
            border_color = QColor("#8b5cf6")
            norm_bg = QColor("#332415")
            norm_pen = QColor("#d97706")
            norm_fg = QColor("#fed7aa")

        # 1. Output label at Top
        painter.setPen(QColor("#f8fafc"))
        font_label = QFont("Arial", 9, QFont.Bold)
        font_pill = QFont("Arial", 9, QFont.Bold)
        painter.setFont(font_label)
        painter.drawText(QRectF(box_left, margin_top, block_w, 18), Qt.AlignCenter, "Output")

        # Arrow down from Output into Block
        y_cursor = margin_top + 20.0
        painter.setPen(QPen(border_color, 2))
        painter.drawLine(QPointF(center_x, y_cursor), QPointF(center_x, y_cursor + 14.0))

        # 2. Transformer Block Outer Container (Nx)
        transformer_box_top = y_cursor + 14.0
        transformer_box_h = step * 5.4
        transformer_box_w = block_w + 54.0
        transformer_box_left = center_x - (transformer_box_w / 2) + 12.0
        transformer_rect = QRectF(transformer_box_left, transformer_box_top, transformer_box_w, transformer_box_h)

        # Draw outer container
        painter.setPen(QPen(border_color, 1.5, Qt.SolidLine))
        painter.setBrush(QBrush(QColor("#131522")))
        painter.drawRoundedRect(transformer_rect, 10, 10)

        # Block name and "Nx" label on the left
        painter.setPen(QColor("#cbd5e1"))
        painter.setFont(QFont("Arial", 8, QFont.DemiBold))
        name_parts = block_title.split(" ")
        painter.drawText(QPointF(transformer_box_left - 82.0, transformer_box_top + 28.0), name_parts[0])
        if len(name_parts) > 1:
            painter.drawText(QPointF(transformer_box_left - 82.0, transformer_box_top + 42.0), name_parts[1])
        painter.setPen(border_color)
        painter.drawText(QPointF(transformer_box_left - 24.0, transformer_box_top + (transformer_box_h / 2)), "Nx")

        # 3. Inside Transformer Block: Layers & Residuals
        # 3a. Add & Norm (Top)
        pill_y1 = transformer_box_top + 12.0
        rect_an1 = QRectF(box_left, pill_y1, block_w, block_h)
        painter.setPen(QPen(norm_pen, 1.5))
        painter.setBrush(QBrush(norm_bg))
        painter.drawRoundedRect(rect_an1, 6, 6)
        painter.setPen(norm_fg)
        painter.setFont(font_pill)
        painter.drawText(rect_an1, Qt.AlignCenter, top_norm_text)

        # Arrow down from Add & Norm to Feed Forward
        arrow1_y = pill_y1 + block_h
        arrow1_end = arrow1_y + 12.0
        painter.setPen(QPen(QColor("#0284c7"), 1.8))
        painter.drawLine(QPointF(center_x, arrow1_y), QPointF(center_x, arrow1_end))

        # 3b. Feed Forward / Gate
        pill_y2 = arrow1_end
        rect_ff = QRectF(box_left, pill_y2, block_w, block_h)
        painter.setPen(QPen(QColor("#0284c7"), 1.5))
        painter.setBrush(QBrush(QColor("#112436")))
        painter.drawRoundedRect(rect_ff, 6, 6)
        painter.setPen(QColor("#bae6fd"))
        painter.drawText(rect_ff, Qt.AlignCenter, ff_text)

        # Arrow down from Feed Forward to MLP
        arrow2_y = pill_y2 + block_h
        arrow2_end = arrow2_y + 12.0
        painter.setPen(QPen(QColor("#9333ea"), 1.8))
        painter.drawLine(QPointF(center_x, arrow2_y), QPointF(center_x, arrow2_end))

        # 3c. MLP / Up-Down Projection
        pill_y3 = arrow2_end
        rect_mlp = QRectF(box_left, pill_y3, block_w, block_h)
        painter.setPen(QPen(QColor("#9333ea"), 1.5))
        painter.setBrush(QBrush(QColor("#241436")))
        painter.drawRoundedRect(rect_mlp, 6, 6)
        painter.setPen(QColor("#f3e8ff"))
        painter.drawText(rect_mlp, Qt.AlignCenter, mlp_text)

        # Arrow down from MLP to Add & Norm 2
        arrow3_y = pill_y3 + block_h
        arrow3_end = arrow3_y + 12.0
        painter.setPen(QPen(norm_pen, 1.8))
        painter.drawLine(QPointF(center_x, arrow3_y), QPointF(center_x, arrow3_end))

        # 3d. Add & Norm (Bottom)
        pill_y4 = arrow3_end
        rect_an2 = QRectF(box_left, pill_y4, block_w, block_h)
        painter.setPen(QPen(norm_pen, 1.5))
        painter.setBrush(QBrush(norm_bg))
        painter.drawRoundedRect(rect_an2, 6, 6)
        painter.setPen(norm_fg)
        painter.drawText(rect_an2, Qt.AlignCenter, bottom_norm_text)

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
        painter.drawText(rect_attn, Qt.AlignCenter, attn_text)

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

        pe_radius = 8.0
        if is_llama or is_neox:
            # RoPE Rotary embedding indicator pill/diamond
            pe_rect = QRectF(center_x - 10.0, post_y, 20.0, 16.0)
            painter.setPen(QPen(QColor("#06b6d4"), 1.5))
            painter.setBrush(QBrush(QColor("#083344")))
            painter.drawRoundedRect(pe_rect, 4, 4)
            painter.setPen(QColor("#67e8f9"))
            painter.setFont(QFont("Arial", 7, QFont.Bold))
            painter.drawText(pe_rect, Qt.AlignCenter, "θ")
        else:
            # Positional Encoding Node (Circle with +)
            painter.setPen(QPen(QColor("#cbd5e1"), 1.5))
            painter.setBrush(QBrush(QColor("#181b28")))
            painter.drawEllipse(QPointF(center_x, post_y + pe_radius), pe_radius, pe_radius)
            painter.drawLine(QPointF(center_x - 4.0, post_y + pe_radius), QPointF(center_x + 4.0, post_y + pe_radius))
            painter.drawLine(QPointF(center_x, post_y + pe_radius - 4.0), QPointF(center_x, post_y + pe_radius + 4.0))

        # Positional Encoding Label
        painter.setPen(QColor("#94a3b8"))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(QPointF(center_x + 18.0, post_y + pe_radius + 4.0), pe_text)

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
        painter.drawText(QRectF(box_left, arrow6_y + 12.0, block_w, 18), Qt.AlignCenter, "Input Tokens")

        painter.end()
