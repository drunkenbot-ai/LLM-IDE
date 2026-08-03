"""Compatibility facade for Wikipedia download APIs."""
from PySide6.QtWidgets import QApplication, QMainWindow
from .wiki_download_backend import WikipediaDownloaderBackend
from .wiki_download_worker import DownloadWorker
from .wiki_download_layout import _GuiLayout
from .wiki_download_style import _GuiStyle
from .wiki_download_actions import _GuiActions
from .wiki_download_processing import *

class WikipediaDownloaderGUI(_GuiLayout, _GuiStyle, _GuiActions, QMainWindow):
    """Qt GUI compatibility wrapper for the Wikipedia downloader."""
    pass

def main() -> None:
    """Launch the Wikipedia downloader GUI."""
    app = QApplication.instance() or QApplication([])
    window = WikipediaDownloaderGUI()
    window.show()
    app.exec()
