from __future__ import annotations
from typing import List
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from .wiki_download_backend import WikipediaDownloaderBackend
from .wiki_download_worker import DownloadWorker

class _GuiStyle:
    def apply_styles(self):
        """Apply modern stylesheet to the application with proper colors"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d0d7de;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                padding-bottom: 15px;
                background-color: #ffffff;
                color: #1a1a1a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #1a1a1a;
                background-color: #ffffff;
            }
            QLabel {
                color: #1a1a1a;
            }
            QCheckBox {
                color: #1a1a1a;
                background-color: transparent;
            }
            QSpinBox {
                padding: 5px;
                border: 1px solid #d0d7de;
                border-radius: 4px;
                background-color: #ffffff;
                color: #1a1a1a;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #f0f2f5;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #d0d7de;
                border-radius: 4px;
                background-color: #ffffff;
                color: #1a1a1a;
            }
            QTextEdit {
                background-color: #f8f9fa;
                color: #1a1a1a;
                border: 1px solid #d0d7de;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #2ea44f;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #22863a;
            }
            QPushButton:disabled {
                background-color: #d0d7de;
                color: #8b949e;
            }
            QPushButton#danger {
                background-color: #da3633;
            }
            QPushButton#danger:hover {
                background-color: #b62324;
            }
            QSplitter::handle {
                background-color: #d0d7de;
                width: 2px;
            }
            QSplitter::handle:hover {
                background-color: #2ea44f;
            }
            QListWidget {
                background-color: #ffffff;
                color: #1a1a1a;
                border: 1px solid #d0d7de;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                color: #1a1a1a;
                padding: 8px;
                border-bottom: 1px solid #f0f2f5;
            }
            QListWidget::item:selected {
                background-color: #ddf4ff;
                color: #1a1a1a;
                border: none;
            }
            QListWidget::item:hover {
                background-color: #f6f8fa;
            }
            QProgressBar {
                border: 1px solid #d0d7de;
                border-radius: 5px;
                text-align: center;
                background-color: #ffffff;
                color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #2ea44f;
                border-radius: 5px;
            }
            QStatusBar {
                background-color: #f8f9fa;
                color: #1a1a1a;
                border-top: 1px solid #d0d7de;
                padding: 5px;
            }
            QScrollBar:vertical {
                background-color: #f6f8fa;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #d0d7de;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #8b949e;
            }
            QScrollBar:horizontal {
                background-color: #f6f8fa;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #d0d7de;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #8b949e;
            }
            QMenuBar {
                background-color: #ffffff;
                color: #1a1a1a;
            }
            QMenuBar::item:selected {
                background-color: #f0f2f5;
            }
            QMenu {
                background-color: #ffffff;
                color: #1a1a1a;
            }
            QMenu::item:selected {
                background-color: #f0f2f5;
            }
            QMessageBox {
                background-color: #ffffff;
                color: #1a1a1a;
            }
            QMessageBox QLabel {
                color: #1a1a1a;
            }
            QMessageBox QPushButton {
                background-color: #2ea44f;
                color: #ffffff;
                min-width: 80px;
                padding: 8px;
            }
            QMessageBox QPushButton:hover {
                background-color: #22863a;
            }
            QDialog {
                background-color: #ffffff;
                color: #1a1a1a;
            }
            QDialog QLabel {
                color: #1a1a1a;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #ffffff;
                border: 2px solid #d0d7de;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #2ea44f;
                border: 2px solid #2ea44f;
                border-radius: 4px;
            }
        """)

