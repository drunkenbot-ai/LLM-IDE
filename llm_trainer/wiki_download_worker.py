from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import List
from PySide6.QtCore import QThread, Signal
from .wiki_download_backend import WikipediaDownloaderBackend

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

