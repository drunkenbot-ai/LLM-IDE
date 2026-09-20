"""Neon navigation card widgets strictly matching Reference Images 1 and 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QPushButton, QWidget


class NeonNavCard(QPushButton):
    """Glowing hero navigation card with glowing icon and label.

    Matches the prominent amber, purple, and cyan cards seen in Reference
    Images 1 and 2 for Dataset Recipe, Architecture, and Compute Engine.
    """

    def __init__(
        self,
        title: str,
        icon_path: str,
        glow_color: str = "#8b5cf6",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("NavButton")
        self.setCheckable(True)
        self.setFixedSize(86, 74)
        self._title = title
        self._glow_color = QColor(glow_color)
        self._icon_path = icon_path
        self._pixmap: QPixmap | None = None

        if Path(icon_path).is_file():
            self._pixmap = QPixmap(icon_path)
            self.setIcon(QIcon(self._pixmap))
            self.setProperty("_nav_icon_path", icon_path)

        self.setToolTip(title)
        self.setAccessibleName(title)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event: Any) -> None:
        """Render the neon card with glowing active border and centered typography."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w = float(self.width())
        h = float(self.height())
        rect = QRectF(1.5, 1.5, w - 3.0, h - 3.0)

        is_active = self.isChecked()
        is_hover = self.underMouse()

        # 1. Background and Border
        if is_active:
            # Active neon glow state
            bg_color = QColor(self._glow_color)
            bg_color.setAlpha(36)
            painter.setBrush(QBrush(bg_color))
            pen = QPen(self._glow_color, 1.8)
            painter.setPen(pen)
            painter.drawRoundedRect(rect, 10.0, 10.0)
        elif is_hover:
            # Hover state
            painter.setBrush(QBrush(QColor(255, 255, 255, 16)))
            painter.setPen(QPen(QColor(255, 255, 255, 45), 1.0))
            painter.drawRoundedRect(rect, 10.0, 10.0)
        else:
            # Subtle resting card state
            painter.setBrush(QBrush(QColor(16, 19, 28, 140)))
            painter.setPen(QPen(QColor(30, 36, 50, 90), 1.0))
            painter.drawRoundedRect(rect, 10.0, 10.0)

        # 2. Icon (centered in top area)
        icon_size = 34.0
        icon_x = (w - icon_size) / 2.0
        icon_y = 6.0
        if self._pixmap and not self._pixmap.isNull():
            target_rect = QRectF(icon_x, icon_y, icon_size, icon_size)
            painter.drawPixmap(target_rect.toRect(), self._pixmap)

        # 3. Label text (centered below icon)
        font = QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal)
        painter.setFont(font)

        if is_active:
            text_color = QColor("#ffffff")
        elif is_hover:
            text_color = QColor("#e2e8f0")
        else:
            text_color = QColor("#94a3b8")

        painter.setPen(text_color)
        text_rect = QRectF(2.0, 42.0, w - 4.0, 28.0)
        painter.drawText(
            text_rect,
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
            self._title,
        )
        painter.end()


class CompactNavButton(QPushButton):
    """Compact navigation button with vertical icon and title.

    Used for Datasets, Today, UI, Training, Deploy, Docs, and Settings.
    """

    def __init__(
        self,
        title: str,
        icon_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("NavButton")
        self.setCheckable(True)
        self.setFixedSize(86, 44)
        self._title = title
        self._pixmap: QPixmap | None = None

        icon_path = Path(__file__).resolve().parent.parent / "icons" / icon_name
        if icon_path.is_file():
            self._pixmap = QPixmap(str(icon_path))
            self.setIcon(QIcon(self._pixmap))
            self.setProperty("_nav_icon_path", str(icon_path))

        self.setToolTip(title)
        self.setAccessibleName(title)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event: Any) -> None:
        """Render compact navigation button with vertical icon and title."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w = float(self.width())
        h = float(self.height())
        rect = QRectF(2.0, 1.0, w - 4.0, h - 2.0)

        is_active = self.isChecked()
        is_hover = self.underMouse()

        if is_active:
            painter.setBrush(QBrush(QColor(99, 102, 241, 35)))
            painter.setPen(QPen(QColor(129, 140, 248), 1.4))
            painter.drawRoundedRect(rect, 6.0, 6.0)
        elif is_hover:
            painter.setBrush(QBrush(QColor(255, 255, 255, 12)))
            painter.setPen(QPen(QColor(255, 255, 255, 30), 1.0))
            painter.drawRoundedRect(rect, 6.0, 6.0)
        else:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(Qt.NoPen)

        # Icon
        icon_size = 18.0
        icon_x = (w - icon_size) / 2.0
        icon_y = 4.0
        if self._pixmap and not self._pixmap.isNull():
            painter.drawPixmap(QRectF(icon_x, icon_y, icon_size, icon_size).toRect(), self._pixmap)

        # Label
        font = QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QColor("#f8fafc" if is_active else ("#cbd5e1" if is_hover else "#64748b")))
        text_rect = QRectF(0.0, 24.0, w, 16.0)
        painter.drawText(text_rect, Qt.AlignCenter, self._title)
        painter.end()
