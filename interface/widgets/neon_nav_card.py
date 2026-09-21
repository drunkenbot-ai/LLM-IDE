"""Navigation card widgets strictly matching Architecture Studio and Concept References."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QPushButton, QWidget

ICONS_DIR = Path(__file__).resolve().parent.parent / "icons"
_ICON_CACHE: dict[tuple[str, str, int], QPixmap] = {}


def get_tinted_pixmap(name: str, color: QColor, size: int = 30) -> QPixmap:
    """Load high-DPI transparent PNG icon and tint its pixels with high quality."""
    key = (name, color.name(), size)
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]

    path = ICONS_DIR / name
    if not path.is_file():
        path = Path(name)
    if not path.is_file():
        return QPixmap()

    orig = QImage(str(path))
    if orig.isNull():
        return QPixmap()

    scaled = orig.scaled(
        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    ).convertToFormat(QImage.Format_ARGB32)

    out = QImage(scaled.size(), QImage.Format_ARGB32)
    out.fill(Qt.transparent)

    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setCompositionMode(QPainter.CompositionMode_Source)
    painter.drawImage(0, 0, scaled)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(out.rect(), color)
    painter.end()

    pix = QPixmap.fromImage(out)
    _ICON_CACHE[key] = pix
    return pix


def _draw_cloche_icon(painter: QPainter, cx: float, cy: float, color: QColor, scale: float = 1.0) -> None:
    """Draw glowing gourmet cloche dish icon with steam."""
    painter.save()
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(color, 2.0 * scale, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawLine(int(cx), int(cy - 16 * scale), int(cx), int(cy - 10 * scale))
    painter.drawLine(int(cx - 7 * scale), int(cy - 14 * scale), int(cx - 7 * scale), int(cy - 8 * scale))
    painter.drawLine(int(cx + 7 * scale), int(cy - 14 * scale), int(cx + 7 * scale), int(cy - 8 * scale))
    painter.setBrush(QBrush(color))
    painter.drawEllipse(QRectF(cx - 3 * scale, cy - 8 * scale, 6 * scale, 6 * scale))
    painter.setBrush(Qt.NoBrush)
    painter.drawArc(QRectF(cx - 18 * scale, cy - 5 * scale, 36 * scale, 26 * scale), 0, 180 * 16)
    painter.drawRoundedRect(QRectF(cx - 22 * scale, cy + 9 * scale, 44 * scale, 4.5 * scale), 2.2 * scale, 2.2 * scale)
    painter.restore()


def _draw_brain_icon(painter: QPainter, cx: float, cy: float, color: QColor, scale: float = 1.0) -> None:
    """Draw glowing dual-lobe neural brain icon."""
    painter.save()
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(color, 2.0 * scale, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawLine(int(cx), int(cy - 15 * scale), int(cx), int(cy + 15 * scale))
    painter.drawArc(QRectF(cx - 17 * scale, cy - 16 * scale, 17 * scale, 16 * scale), 90 * 16, 180 * 16)
    painter.drawArc(QRectF(cx - 20 * scale, cy - 6 * scale, 20 * scale, 15 * scale), 100 * 16, 160 * 16)
    painter.drawArc(QRectF(cx - 16 * scale, cy + 5 * scale, 16 * scale, 12 * scale), 140 * 16, 160 * 16)
    painter.drawArc(QRectF(cx, cy - 16 * scale, 17 * scale, 16 * scale), 270 * 16, 180 * 16)
    painter.drawArc(QRectF(cx, cy - 6 * scale, 20 * scale, 15 * scale), 280 * 16, 160 * 16)
    painter.drawArc(QRectF(cx, cy + 5 * scale, 16 * scale, 12 * scale), 240 * 16, 160 * 16)
    painter.restore()


def _draw_chip_icon(painter: QPainter, cx: float, cy: float, color: QColor, scale: float = 1.0) -> None:
    """Draw glowing GPU microchip with circuit pins."""
    painter.save()
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(color, 1.9 * scale, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawRoundedRect(QRectF(cx - 15 * scale, cy - 15 * scale, 30 * scale, 30 * scale), 4 * scale, 4 * scale)
    painter.setPen(QPen(color.lighter(120), 1.2 * scale))
    painter.drawRoundedRect(QRectF(cx - 10 * scale, cy - 10 * scale, 20 * scale, 20 * scale), 2 * scale, 2 * scale)
    font = QFont("Arial", int(7.0 * scale), QFont.Bold)
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(QRectF(cx - 10 * scale, cy - 10 * scale, 20 * scale, 20 * scale), Qt.AlignCenter, "GPU")
    painter.setPen(QPen(color, 1.8 * scale, Qt.SolidLine, Qt.RoundCap))
    for off in (-6.0 * scale, 6.0 * scale):
        painter.drawLine(int(cx - 19 * scale), int(cy + off), int(cx - 15 * scale), int(cy + off))
        painter.drawLine(int(cx + 15 * scale), int(cy + off), int(cx + 19 * scale), int(cy + off))
        painter.drawLine(int(cx + off), int(cy - 19 * scale), int(cx + off), int(cy - 15 * scale))
        painter.drawLine(int(cx + off), int(cy + 15 * scale), int(cx + off), int(cy + 19 * scale))
    painter.restore()


class StudioNavButton(QPushButton):
    """Luxury obsidian navigation button strictly matching Architecture Studio and Concept References."""

    def __init__(
        self,
        title: str,
        icon_name: str = "",
        glow_color: str = "#f59e0b",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("NavButton")
        self.setCheckable(True)
        self.setFixedSize(68, 54)
        self._title = title
        self._icon_name = icon_name
        self._glow_color = QColor(glow_color)
        self.setToolTip(title)
        self.setAccessibleName(title)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event: Any) -> None:
        """Render obsidian navigation button with sharp icon artwork and glowing active card."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w = float(self.width())
        h = float(self.height())
        rect = QRectF(2.0, 1.5, w - 4.0, h - 3.0)

        is_active = self.isChecked()
        is_hover = self.underMouse()
        is_expanded = w > 100.0

        if is_active:
            # Active selected state matching Concept: golden amber rounded rectangle
            painter.setBrush(QBrush(QColor(245, 158, 11, 230)))
            painter.setPen(QPen(QColor("#fbbf24"), 1.8))
            painter.drawRoundedRect(rect, 10.0, 10.0)
            icon_color = QColor("#ffffff")
        elif is_hover:
            painter.setBrush(QBrush(QColor(32, 38, 52, 220)))
            painter.setPen(QPen(QColor(64, 76, 104), 1.2))
            painter.drawRoundedRect(rect, 10.0, 10.0)
            icon_color = QColor("#ffffff")
        else:
            painter.setBrush(QBrush(QColor(21, 24, 34, 210)))
            painter.setPen(QPen(QColor(36, 42, 58, 170), 1.0))
            painter.drawRoundedRect(rect, 10.0, 10.0)
            icon_color = QColor("#cbd5e1")

        # Draw icon and optional expanded label
        if is_expanded:
            pix = get_tinted_pixmap(self._icon_name, icon_color, 24)
            if not pix.isNull():
                painter.drawPixmap(12, int((h - pix.height()) / 2.0), pix)
            font = QFont("Arial", 9, QFont.Weight.Bold if is_active else QFont.Weight.Normal)
            painter.setFont(font)
            painter.setPen(QColor("#ffffff" if is_active else ("#f1f5f9" if is_hover else "#94a3b8")))
            text_rect = QRectF(46.0, 0.0, w - 50.0, h)
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self._title)
        else:
            pix = get_tinted_pixmap(self._icon_name, icon_color, 30)
            if not pix.isNull():
                painter.drawPixmap(int((w - pix.width()) / 2.0), int((h - pix.height()) / 2.0), pix)

        painter.end()


class NeonNavCard(StudioNavButton):
    """Backward compatible NeonNavCard subclassing StudioNavButton."""

    def __init__(
        self,
        title: str,
        icon_path: str = "",
        glow_color: str = "#8b5cf6",
        parent: QWidget | None = None,
    ) -> None:
        icon_name = Path(icon_path).name if icon_path else "AI_tab_icon.png"
        t_low = title.lower()
        if "recipe" in t_low:
            icon_name = "ingestion_tab_icon.png"
            glow_color = "#f59e0b"
        elif "compute" in t_low or "job" in t_low or "cluster" in t_low:
            icon_name = "job_tab_icon.png"
            glow_color = "#06b6d4"
        elif "architecture" in t_low or "forge" in t_low or "train" in t_low:
            icon_name = "fine_tune_tab.png"
            glow_color = "#8b5cf6"
        super().__init__(title, icon_name, glow_color, parent)


class CompactNavButton(StudioNavButton):
    """Backward compatible CompactNavButton subclassing StudioNavButton."""

    def __init__(
        self,
        title: str,
        icon_name: str = "",
        parent: QWidget | None = None,
    ) -> None:
        t_low = title.lower()
        glow_color = "#f59e0b"
        if not icon_name:
            if "dataset" in t_low and "recipe" in t_low:
                icon_name = "ingestion_tab_icon.png"
            elif "dataset" in t_low or "ingest" in t_low:
                icon_name = "AI_tab_icon.png"
            elif "source" in t_low or "blueprint" in t_low or "today" in t_low or "plan" in t_low:
                icon_name = "plan_tab_icon.png"
            elif "train" in t_low or "arch" in t_low:
                icon_name = "fine_tune_tab.png"
                glow_color = "#8b5cf6"
            elif "cluster" in t_low or "job" in t_low or "deploy" in t_low:
                icon_name = "job_tab_icon.png"
                glow_color = "#06b6d4"
            elif "live" in t_low:
                icon_name = "live_tab_icon.png"
                glow_color = "#10b981"
            elif "bench" in t_low or "ui" in t_low:
                icon_name = "benchmark_tab_icon.png"
                glow_color = "#38bdf8"
            elif "export" in t_low or "doc" in t_low:
                icon_name = "export_tab_icon.png"
                glow_color = "#ec4899"
            else:
                icon_name = "chat_tab_icon.png"
                glow_color = "#a855f7"
        super().__init__(title, icon_name, glow_color, parent)


