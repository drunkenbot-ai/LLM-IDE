from __future__ import annotations
from typing import List
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from .wiki_download_backend import WikipediaDownloaderBackend
from .wiki_download_worker import DownloadWorker

class _GuiActions:
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

