from __future__ import annotations
from typing import List
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from .wiki_download_backend import WikipediaDownloaderBackend
from .wiki_download_worker import DownloadWorker

class _GuiLayout:
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Wikipedia Dataset Downloader - DrunkenBot")
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
        self.limit_spin.setRange(5, 1000)
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
        self.min_words_spin.setValue(15000)
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

