#!/usr/bin/env python3
"""
Wikipedia Dataset Downloader GUI - PySide6
Fixed color theme for better visibility
"""

import sys
import os
import json
import time
import re
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import requests
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *



# ============================================================
# Cleanup Configuration
# ============================================================

WORDS_PER_CHUNK = 1200

REMOVE_SECTIONS = [
    "References",
    "External links",
    "Bibliography",
    "Further reading",
    "See also",
    "Notes",
    "Sources",
    "Citations",
]

# ============================================================

css_patterns = [
    r"\.mw-parser-output.*?(?=(?:[A-Z][a-z].*?\n)|$)",
    r"@media.*?(?=\n[A-Z]|\Z)",
]

line_patterns = [
    r"^\s*v\s*t\s*e\s*$",
    r"^\s*Main article:.*$",
    r"^\s*Further information:.*$",
    r"^\s*See also:.*$",
    r"^\s*Coordinates:.*$",
    r"^\s*This article is about.*$",
]

reference_pattern = re.compile(r"\[[^\]]+\]")
whitespace_pattern = re.compile(r"\s+")


# ============================================================================
# Backend: Wikipedia Downloader
# ============================================================================

class WikipediaDownloaderBackend:
    """Backend class for downloading Wikipedia pages"""

    def __init__(self):
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.min_request_interval = 1.0
        self.last_request_time = 0
        self.is_running = False

    def _rate_limit(self):
        """Rate limiting for Wikipedia API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def _make_request(self, params: Dict) -> Dict:
        """Make API request with rate limiting"""
        self._rate_limit()
        try:
            response = self.session.get(
                self.api_url,
                params=params,
                headers={'User-Agent': 'TinyLLM-Wikipedia-GUI/1.0'}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'error': str(e)}

    def search_pages(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for Wikipedia pages"""
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': limit
        }

        data = self._make_request(params)
        if 'error' in data:
            return []

        results = data.get('query', {}).get('search', [])
        pages = []
        for result in results:
            pages.append({
                'title': result['title'],
                'pageid': result['pageid'],
                'snippet': result.get('snippet', ''),
                'size': result.get('size', 0),
                'wordcount': result.get('wordcount', 0)
            })
        return pages

    def get_page_content(self, title: str) -> Optional[Dict]:
        """Get full page content"""
        params = {
            'action': 'parse',
            'page': title,
            'format': 'json',
            'prop': 'text|revid|categories|links',
            'formatversion': 2
        }

        data = self._make_request(params)
        if 'error' in data:
            return None

        parse_data = data.get('parse', {})
        if not parse_data:
            return None

        html_content = parse_data.get('text', '')
        plain_text = self._clean_html(html_content)

        return {
            'title': title,
            'text': plain_text,
            'revid': parse_data.get('revid', 0),
            'categories': parse_data.get('categories', []),
            'timestamp': datetime.utcnow().isoformat()
        }

    def _clean_html(self, html_content: str) -> str:
        """Extract plain text from HTML"""
        import html
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = re.sub(r'\[\d+\]', '', text)
        return text

    def sanitize_filename(self, title: str) -> str:
        """Create safe filename"""
        safe = re.sub(r'[<>:"/\\|?*]', '_', title)
        if len(safe) > 200:
            safe = safe[:200]
        return safe


# ============================================================================
# Worker Thread for Downloading
# ============================================================================

class DownloadWorker(QThread):
    """Worker thread for downloading pages without blocking UI"""

    # Signals
    progress_updated = Signal(int, int)  # current, total
    page_downloaded = Signal(str, bool)  # title, success
    status_updated = Signal(str)  # status message
    download_complete = Signal(dict)  # summary stats
    error_occurred = Signal(str)  # error message

    def __init__(self, pages: List[str], output_dir: str,
                 save_metadata: bool = False):
        super().__init__()
        self.pages = pages
        self.output_dir = output_dir
        self.save_metadata = save_metadata
        self.is_running = True
        self.downloader = WikipediaDownloaderBackend()

    def run(self):
        """Main download process"""
        total_pages = len(self.pages)
        downloaded = 0
        failed = 0
        skipped = 0
        successful_titles = []
        failed_titles = []

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.status_updated.emit(
            f"Starting download of {total_pages} pages...")

        for idx, title in enumerate(self.pages, 1):
            if not self.is_running:
                self.status_updated.emit("Download cancelled")
                break

            self.progress_updated.emit(idx, total_pages)
            self.status_updated.emit(
                f"Downloading: {title} ({idx}/{total_pages})")

            # Check if already exists
            safe_title = self.downloader.sanitize_filename(title)
            file_path = output_path / f"{safe_title}.txt"

            if file_path.exists():
                skipped += 1
                self.page_downloaded.emit(title, False)
                self.status_updated.emit(f"Skipped {title} (already exists)")
                continue

            # Download page
            content = self.downloader.get_page_content(title)

            if content and content.get('text'):
                try:
                    # Save text
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content['text'])

                    # Save metadata if requested
                    if self.save_metadata:
                        meta_path = output_path / f"{safe_title}.meta.json"
                        with open(meta_path, 'w', encoding='utf-8') as f:
                            json.dump(content, f, indent=2)

                    downloaded += 1
                    successful_titles.append(title)
                    self.page_downloaded.emit(title, True)

                except Exception as e:
                    failed += 1
                    failed_titles.append(title)
                    self.error_occurred.emit(f"Error saving {title}: {str(e)}")
            else:
                failed += 1
                failed_titles.append(title)
                self.page_downloaded.emit(title, False)

            # Small delay between requests
            time.sleep(0.5)

        # Save index file
        self._save_index(successful_titles, failed_titles, output_path)

        # Emit completion signal
        summary = {
            'total': total_pages,
            'downloaded': downloaded,
            'failed': failed,
            'skipped': skipped,
            'successful_titles': successful_titles,
            'failed_titles': failed_titles,
            'output_dir': str(output_path)
        }

        self.download_complete.emit(summary)
        self.status_updated.emit(
            f"Download complete! Downloaded: {downloaded}, Failed: {failed}, Skipped: {skipped}")

    def _save_index(self, successful_titles: List[str],
                    failed_titles: List[str], output_path: Path):
        """Save index file"""
        index = {
            'download_date': datetime.utcnow().isoformat(),
            'total_pages': len(successful_titles) + len(failed_titles),
            'successful': len(successful_titles),
            'failed': len(failed_titles),
            'successful_titles': successful_titles,
            'failed_titles': failed_titles
        }

        index_path = output_path / 'download_index.json'
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

    def stop(self):
        """Stop the download process"""
        self.is_running = False


# ============================================================================
# Main GUI Application
# ============================================================================

class WikipediaDownloaderGUI(QMainWindow):
    """Main GUI window for Wikipedia Downloader"""

    def __init__(self):
        super().__init__()
        self.downloader = WikipediaDownloaderBackend()
        self.current_pages = []
        self.worker = None
        self.output_dir = str(Path.home() / "wikipedia_dataset")

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Wikipedia Dataset Downloader - TinyLLM")
        self.setGeometry(100, 100, 1100, 800)

        # Apply modern color scheme
        self.apply_styles()

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # ====================================================================
        # Search Section
        # ====================================================================
        search_group = QGroupBox("🔍 Search Wikipedia")
        search_layout = QVBoxLayout()

        # Search input row
        input_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Enter topic to search (e.g., Artificial Intelligence)")
        self.search_input.returnPressed.connect(self.search_pages)
        self.search_input.setMinimumHeight(35)

        self.search_button = QPushButton("🔍 Search")
        self.search_button.clicked.connect(self.search_pages)
        self.search_button.setMinimumHeight(35)

        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(5, 100)
        self.limit_spin.setValue(20)
        self.limit_spin.setPrefix("Max results: ")
        self.limit_spin.setMinimumHeight(35)

        input_layout.addWidget(self.search_input, 3)
        input_layout.addWidget(self.limit_spin, 1)
        input_layout.addWidget(self.search_button, 1)

        search_layout.addLayout(input_layout)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(80)
        self.results_text.setPlaceholderText(
            "Search results will appear here...")
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                color: #212529;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
            }
        """)

        search_layout.addWidget(self.results_text)
        search_group.setLayout(search_layout)
        main_layout.addWidget(search_group)

        # ====================================================================
        # Page Selection Section
        # ====================================================================
        selection_group = QGroupBox("📄 Pages to Download")
        selection_layout = QVBoxLayout()

        # Control buttons for selection
        selection_controls = QHBoxLayout()
        self.select_all_button = QPushButton("✅ Select All")
        self.select_all_button.clicked.connect(self.select_all_pages)
        self.select_none_button = QPushButton("❌ Select None")
        self.select_none_button.clicked.connect(self.select_none_pages)
        self.add_selected_button = QPushButton("➕ Add Selected to Download")
        self.add_selected_button.clicked.connect(self.add_selected_pages)
        self.clear_list_button = QPushButton("🗑️ Clear List")
        self.clear_list_button.clicked.connect(self.clear_page_list)
        self.clear_list_button.setObjectName("danger")

        for btn in [self.select_all_button, self.select_none_button,
                    self.add_selected_button, self.clear_list_button]:
            btn.setMinimumHeight(30)

        selection_controls.addWidget(self.select_all_button)
        selection_controls.addWidget(self.select_none_button)
        selection_controls.addWidget(self.add_selected_button)
        selection_controls.addWidget(self.clear_list_button)
        selection_controls.addStretch()

        selection_layout.addLayout(selection_controls)

        # Split view for search results and selected pages
        splitter = QSplitter(Qt.Horizontal)

        # Search results list with checkboxes
        self.search_results_list = QListWidget()
        self.search_results_list.setSelectionMode(
            QListWidget.ExtendedSelection)
        self.search_results_list.setMinimumHeight(200)
        self.search_results_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                color: #212529;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #f0f0f0;
                color: #212529;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #212529;
            }
            QListWidget::item:hover {
                background-color: #f8f9fa;
            }
        """)

        # Selected pages list
        self.selected_pages_list = QListWidget()
        self.selected_pages_list.setMinimumHeight(200)
        self.selected_pages_list.setStyleSheet("""
            QListWidget {
                background-color: #f8f9fa;
                color: #212529;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #e0e0e0;
                color: #212529;
            }
            QListWidget::item:selected {
                background-color: #c8e6c9;
                color: #212529;
            }
            QListWidget::item:hover {
                background-color: #e8f5e9;
            }
        """)

        # Labels for lists
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_label = QLabel("📋 Search Results")
        left_label.setStyleSheet(
            "font-weight: bold; color: #212529; padding: 5px;")
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.search_results_list)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_label = QLabel("📥 Download Queue")
        right_label.setStyleSheet(
            "font-weight: bold; color: #212529; padding: 5px;")
        right_layout.addWidget(right_label)
        right_layout.addWidget(self.selected_pages_list)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 500])

        selection_layout.addWidget(splitter)
        selection_group.setLayout(selection_layout)
        main_layout.addWidget(selection_group)

        # ====================================================================
        # Settings Section
        # ====================================================================
        settings_group = QGroupBox("⚙️ Download Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(10)

        # Output directory
        settings_layout.addWidget(QLabel("📁 Output Directory:"), 0, 0)
        self.output_dir_edit = QLineEdit(self.output_dir)
        self.output_dir_edit.textChanged.connect(self.update_output_dir)
        self.output_dir_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
                color: #212529;
            }
        """)
        settings_layout.addWidget(self.output_dir_edit, 0, 1)

        self.browse_button = QPushButton("📂 Browse...")
        self.browse_button.clicked.connect(self.browse_output_dir)
        self.browse_button.setMinimumHeight(30)
        settings_layout.addWidget(self.browse_button, 0, 2)

        # Options
        self.save_metadata_check = QCheckBox("💾 Save metadata (JSON)")
        self.save_metadata_check.setChecked(True)
        self.save_metadata_check.setStyleSheet("color: #212529;")
        settings_layout.addWidget(self.save_metadata_check, 1, 0, 1, 2)

        self.overwrite_check = QCheckBox("🔄 Overwrite existing files")
        self.overwrite_check.setChecked(False)
        self.overwrite_check.setStyleSheet("color: #212529;")
        settings_layout.addWidget(self.overwrite_check, 1, 2)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # Filter controls
        filters_layout = QHBoxLayout()
        filters_layout.addWidget(QLabel("Min Size (KB):"))
        self.min_size_spin = QDoubleSpinBox()
        self.min_size_spin.setRange(0, 10000)
        self.min_size_spin.setValue(100)
        self.min_size_spin.setSuffix(" KB")
        filters_layout.addWidget(self.min_size_spin)

        filters_layout.addWidget(QLabel("Min Words:"))
        self.min_words_spin = QSpinBox()
        self.min_words_spin.setRange(0, 100000)
        self.min_words_spin.setValue(5000)
        filters_layout.addWidget(self.min_words_spin)

        # Add to your settings layout
        settings_layout.addLayout(filters_layout, 2, 0, 1, 3)

        # ====================================================================
        # Download Controls
        # ====================================================================
        download_group = QGroupBox("⬇️ Download")
        download_layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                text-align: center;
                background-color: white;
                color: #212529;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """)
        download_layout.addWidget(self.progress_bar)

        # Control buttons
        control_layout = QHBoxLayout()
        self.download_button = QPushButton("🚀 Start Download")
        self.download_button.clicked.connect(self.start_download)
        self.download_button.setMinimumHeight(40)
        self.download_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #b0bec5;
                color: #ffffff;
            }
        """)

        self.cancel_button = QPushButton("⏹️ Cancel")
        self.cancel_button.clicked.connect(self.cancel_download)
        self.cancel_button.setObjectName("danger")
        self.cancel_button.setMinimumHeight(40)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #ef9a9a;
                color: #ffffff;
            }
        """)
        self.cancel_button.setEnabled(False)

        control_layout.addWidget(self.download_button)
        control_layout.addWidget(self.cancel_button)
        control_layout.addStretch()

        # Page count label
        self.page_count_label = QLabel("Pages in queue: 0")
        self.page_count_label.setStyleSheet(
            "color: #212529; font-weight: bold;")
        control_layout.addWidget(self.page_count_label)

        download_layout.addLayout(control_layout)
        download_group.setLayout(download_layout)
        main_layout.addWidget(download_group)

        # ====================================================================
        # Status Bar
        # ====================================================================
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #f8f9fa;
                color: #212529;
                border-top: 1px solid #dee2e6;
                padding: 5px;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("✅ Ready")

        # Add progress label to status bar
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #212529; font-weight: bold;")
        self.status_bar.addPermanentWidget(self.progress_label)

        # Update initial state
        self.update_download_button_state()

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

    def setup_connections(self):
        """Setup signal/slot connections"""
        # These are set up in the UI initialization

    # ========================================================================
    # Search Methods
    # ========================================================================

    def search_pages(self):
        """Search for Wikipedia pages with size/wordcount filtering"""
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "⚠️ Warning",
                                "Please enter a search query")
            return

        # Get filter thresholds from UI
        min_size_kb = self.min_size_spin.value() * 1024  # Convert KB to bytes
        min_wordcount = self.min_words_spin.value()

        self.search_button.setEnabled(False)
        self.results_text.clear()
        self.search_results_list.clear()
        self.status_bar.showMessage(f"🔍 Searching for '{query}'...")

        try:
            limit = self.limit_spin.value()
            pages = self.downloader.search_pages(query, limit)

            # Filter pages by size and wordcount
            filtered_pages = []
            for page in pages:
                size_bytes = page.get('size', 0)
                wordcount = page.get('wordcount', 0)

                # Apply filters
                if size_bytes >= min_size_kb and wordcount >= min_wordcount:
                    filtered_pages.append(page)

            self.current_pages = filtered_pages

            if filtered_pages:
                self.results_text.setHtml(f"""
                    <b style='color: #2e7d32;'>✅ Found {len(filtered_pages)} pages</b>
                    <span style='color: #424242;'> (Filtered from {len(pages)} total, min size: {min_size_kb / 1024:.0f}KB, min words: {min_wordcount})</span>
                """)

                for page in filtered_pages:
                    size_kb = page.get('size', 0) / 1024
                    item = QListWidgetItem(
                        f"📄 {page['title']}  |  Size: {size_kb:.1f} KB  |  Words: {page.get('wordcount', 0)}"
                    )
                    item.setData(Qt.UserRole, page['title'])
                    item.setCheckState(Qt.Unchecked)
                    self.search_results_list.addItem(item)

                self.status_bar.showMessage(
                    f"✅ Found {len(filtered_pages)} pages meeting criteria")
            else:
                self.results_text.setHtml(f"""
                    <b style='color: #c62828;'>❌ No pages meeting criteria</b><br>
                    <span style='color: #424242;'>Try lowering the minimum size or word count thresholds.</span>
                """)
                self.status_bar.showMessage("❌ No pages meeting criteria")

        except Exception as e:
            error_msg = f"Error searching: {str(e)}"
            self.results_text.setHtml(
                f"<b style='color: #c62828;'>❌ {error_msg}</b>")
            self.status_bar.showMessage(f"❌ {error_msg}")
            QMessageBox.critical(self, "❌ Error", error_msg)

        self.search_button.setEnabled(True)

    def select_all_pages(self):
        """Select all pages in search results"""
        for i in range(self.search_results_list.count()):
            item = self.search_results_list.item(i)
            item.setCheckState(Qt.Checked)
        self.status_bar.showMessage("✅ All pages selected")

    def select_none_pages(self):
        """Deselect all pages in search results"""
        for i in range(self.search_results_list.count()):
            item = self.search_results_list.item(i)
            item.setCheckState(Qt.Unchecked)
        self.status_bar.showMessage("❌ All pages deselected")

    def add_selected_pages(self):
        """Add selected pages to download list"""
        added_count = 0
        existing_titles = set()

        # Get existing titles in download list
        for i in range(self.selected_pages_list.count()):
            item = self.selected_pages_list.item(i)
            existing_titles.add(item.text())

        for i in range(self.search_results_list.count()):
            item = self.search_results_list.item(i)
            if item.checkState() == Qt.Checked:
                title = item.data(Qt.UserRole)
                if title not in existing_titles:
                    self.selected_pages_list.addItem(title)
                    existing_titles.add(title)
                    added_count += 1

        if added_count > 0:
            self.status_bar.showMessage(
                f"✅ Added {added_count} pages to download list")
            self.update_download_button_state()
        else:
            QMessageBox.information(self, "ℹ️ Info",
                                    "No new pages added (may already be in list)")

    def clear_page_list(self):
        """Clear the download list"""
        if self.selected_pages_list.count() > 0:
            reply = QMessageBox.question(
                self, "⚠️ Confirm Clear",
                "Are you sure you want to clear all pages from the download list?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.selected_pages_list.clear()
                self.status_bar.showMessage("🗑️ Download list cleared")
                self.update_download_button_state()

    # ========================================================================
    # Settings Methods
    # ========================================================================

    def update_output_dir(self, text: str):
        """Update output directory"""
        self.output_dir = text

    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "📂 Select Output Directory",
            self.output_dir,
            QFileDialog.ShowDirsOnly
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            self.output_dir = dir_path

    # ========================================================================
    # Download Methods
    # ========================================================================

    def get_pages_to_download(self) -> List[str]:
        """Get list of pages to download"""
        pages = []
        for i in range(self.selected_pages_list.count()):
            pages.append(self.selected_pages_list.item(i).text())
        return pages

    def update_download_button_state(self):
        """Update download button state based on list content"""
        count = self.selected_pages_list.count()
        has_pages = count > 0
        self.download_button.setEnabled(has_pages and not self.worker)
        self.page_count_label.setText(f"📊 Pages in queue: {count}")

    def start_download(self):
        """Start the download process"""
        pages = self.get_pages_to_download()
        if not pages:
            QMessageBox.warning(self, "⚠️ Warning", "No pages to download")
            return

        # Check output directory
        output_dir = self.output_dir_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "⚠️ Warning",
                                "Please specify an output directory")
            return

        # Confirm
        reply = QMessageBox.question(
            self,
            "🚀 Confirm Download",
            f"Download {len(pages)} pages to:\n{output_dir}\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Disable UI
        self.download_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.search_button.setEnabled(False)
        self.progress_bar.setValue(0)

        # Create and start worker
        self.worker = DownloadWorker(
            pages,
            output_dir,
            self.save_metadata_check.isChecked()
        )

        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.page_downloaded.connect(self.on_page_downloaded)
        self.worker.status_updated.connect(self.update_status)
        self.worker.download_complete.connect(self.on_download_complete)
        self.worker.error_occurred.connect(self.on_error)

        self.worker.start()
        self.status_bar.showMessage("⏳ Downloading...")

    def cancel_download(self):
        """Cancel the download"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "⏹️ Cancel Download",
                "Are you sure you want to cancel the download?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.status_bar.showMessage("⏹️ Cancelling download...")

    def update_progress(self, current: int, total: int):
        """Update progress bar"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"📊 {current}/{total}")

    def on_page_downloaded(self, title: str, success: bool):
        """Handle page download status"""
        status = "✅" if success else "❌"
        if success:
            self.status_bar.showMessage(f"{status} Downloaded: {title}")
        else:
            self.status_bar.showMessage(f"{status} Failed: {title}")

    def update_status(self, message: str):
        """Update status message"""
        self.status_bar.showMessage(message)

    def on_error(self, error_message: str):
        """Handle error"""
        self.status_bar.showMessage(f"❌ Error: {error_message}")
        # Log error but continue
        print(f"Error: {error_message}")

    def on_download_complete(self, summary: dict):
        """Handle download completion"""

        # Wait for worker to completely terminate
        if self.worker is not None:
            self.worker.wait()
            self.worker.deleteLater()

        # Enable UI
        self.download_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.search_button.setEnabled(True)

        self.progress_bar.setValue(100)

        msg = (
            f"🎉 Download Complete!\n\n"
            f"📊 Total pages: {summary['total']}\n"
            f"✅ Downloaded: {summary['downloaded']}\n"
            f"❌ Failed: {summary['failed']}\n"
            f"⏭️ Skipped: {summary['skipped']}\n\n"
            f"📁 Output directory:\n{summary['output_dir']}"
        )

        QMessageBox.information(
            self,
            "Download Complete",
            msg
        )

        self.status_bar.showMessage("Download complete")
        self.progress_label.setText("Done")

        self.update_download_button_state()

        try:
            cleanup(
                INPUT_DIR=self.output_dir_edit.text(),
                OUTPUT_DIR=os.path.join(
                    self.output_dir_edit.text(),
                    "cleaned_files"
                )
            )
        except Exception:
            import traceback
            traceback.print_exc()

        self.worker = None

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    app = QApplication(sys.argv)

    # ----------------------------------------------------------------------
    # Force Fusion style instead of Windows native style
    # This prevents Windows Dark Mode from overriding widget colors.
    # ----------------------------------------------------------------------
    app.setStyle("Fusion")

    # ----------------------------------------------------------------------
    # Light application palette
    # ----------------------------------------------------------------------
    palette = QPalette()

    palette.setColor(QPalette.Window, QColor("#f0f2f5"))
    palette.setColor(QPalette.WindowText, QColor("#1a1a1a"))

    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#f8f9fa"))

    palette.setColor(QPalette.Text, QColor("#1a1a1a"))

    palette.setColor(QPalette.Button, QColor("#2ea44f"))
    palette.setColor(QPalette.ButtonText, QColor("#ffffff"))

    palette.setColor(QPalette.BrightText, QColor("#ffffff"))

    palette.setColor(QPalette.Highlight, QColor("#4CAF50"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))

    palette.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipText, QColor("#1a1a1a"))

    palette.setColor(QPalette.PlaceholderText, QColor("#777777"))

    app.setPalette(palette)

    app.setApplicationName("Wikipedia Dataset Downloader")
    app.setOrganizationName("TinyLLM")

    window = WikipediaDownloaderGUI()
    window.show()

    sys.exit(app.exec())




def remove_sections(text):

    for section in REMOVE_SECTIONS:

        pattern = (
            rf"\n{section}\n.*"
        )

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

    return text


def clean_text(text):
    import re

    # ---------------------------------------------------------
    # Remove CSS
    # ---------------------------------------------------------
    text = re.sub(
        r"\.mw-parser-output.*?(?=The |\# |\n[A-Z])",
        "",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"@media.*?(?=The |\# |\n[A-Z])",
        "",
        text,
        flags=re.DOTALL,
    )

    # ---------------------------------------------------------
    # Remove references like [1], [23], [a]
    # ---------------------------------------------------------
    text = re.sub(r"\[[^\]]+\]", "", text)

    # ---------------------------------------------------------
    # Remove edit markers
    # ---------------------------------------------------------
    text = text.replace("[edit]", "")

    # ---------------------------------------------------------
    # Collapse whitespace first
    # ---------------------------------------------------------
    text = re.sub(r"\s+", " ", text).strip()

    # ---------------------------------------------------------
    # Remove everything before the first real paragraph.
    # Most Wikipedia pages begin with
    #
    # "The ..."
    # "A ..."
    # "An ..."
    #
    # This removes infoboxes/navigation.
    # ---------------------------------------------------------
    m = re.search(r"\b(The|A|An)\b.+", text)

    if m:
        text = text[m.start():]

    # ---------------------------------------------------------
    # Sentence splitting
    # ---------------------------------------------------------
    text = re.sub(
        r"([.!?])\s+",
        r"\1\n",
        text
    )

    # ---------------------------------------------------------
    # Rebuild paragraphs
    # ---------------------------------------------------------
    paragraph_starters = (
        "The ",
        "In ",
        "On ",
        "At ",
        "After ",
        "Before ",
        "During ",
        "By ",
        "Following ",
        "Meanwhile ",
        "However ",
        "Although ",
        "Later ",
        "Since ",
        "From ",
        "As ",
        "When ",
        "While ",
    )

    paragraphs = []
    current = ""

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if current == "":
            current = line
            continue

        if line.startswith(paragraph_starters):
            paragraphs.append(current.strip())
            current = line
        else:
            current += " " + line

    if current:
        paragraphs.append(current.strip())

    # ---------------------------------------------------------
    # Remove obvious junk paragraphs
    # ---------------------------------------------------------
    cleaned = []

    junk_words = (
        "Belligerents",
        "Campaign",
        "Atlantic Theater",
        "West Indies",
        "Result",
        "Date",
        "Location",
        "Combatants",
        "Casualties",
        "Commander",
        "References",
        "External links",
        "Bibliography",
        "Further reading",
        "See also",
    )

    for p in paragraphs:

        if len(p) < 40:
            continue

        if any(word in p for word in junk_words):
            continue

        cleaned.append(p)

    return "\n\n".join(cleaned)


def chunk_text(text, words_per_chunk):

    words = text.split()

    chunks = []

    for i in range(0, len(words), words_per_chunk):

        chunks.append(
            " ".join(words[i:i + words_per_chunk])
        )

    return chunks


def process_file(file_path, output_dir):

    out = Path(output_dir) / file_path.name

    # Skip if already cleaned
    if out.exists():
        print(f"Skipping (already cleaned): {file_path.name}")
        return

    text = file_path.read_text(
        encoding="utf8",
        errors="ignore",
    )

    cleaned = clean_text(text)

    out.write_text(
        cleaned,
        encoding="utf8",
    )

    print(f"Cleaned: {file_path.name}")


def cleanup(INPUT_DIR, OUTPUT_DIR):

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    files = list(input_dir.glob("*.txt"))

    print(f"Found {len(files)} files")

    cleaned_count = 0
    skipped_count = 0

    for i, file in enumerate(files, 1):

        print(f"[{i}/{len(files)}] {file.name}")

        out = output_dir / file.name

        if out.exists():
            print("   -> Already cleaned, skipping.")
            skipped_count += 1
            continue

        process_file(file, output_dir)
        cleaned_count += 1

    print()
    print(f"Cleanup Done. Cleaned: {cleaned_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()