from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from interface.screens.screen_builders import SCREEN_BUILDERS


def create_nav_button(icon_name: str, tooltip: str) -> QPushButton:
    """Create a checkable side-rail navigation button."""
    button = QPushButton()
    icon_path = Path(__file__).resolve().parent.parent / "icons" / icon_name
    if icon_path.is_file():
        button.setIcon(QIcon(str(icon_path)))
    button.setIconSize(QSize(40, 40))
    button.setMinimumHeight(52)
    button.setToolTip(tooltip)
    button.setAccessibleName(tooltip)
    button.setObjectName("NavButton")
    button.setCheckable(True)
    return button


def build_top_bar(window, app_name: str) -> QWidget:
    """Build the shared application header and expose its controls on window."""
    top = QWidget()
    top.setObjectName("TopBar")
    window.top_bar = top
    layout = QHBoxLayout(top)
    layout.setContentsMargins(16, 8, 16, 8)
    layout.setSpacing(8)

    logo = QLabel()
    logo.setObjectName("Logo")
    pixmap = window._app_logo_pixmap(36)
    logo.setPixmap(pixmap) if not pixmap.isNull() else logo.setText("DB")
    logo.setFixedSize(42, 42)
    logo.setScaledContents(False)

    window.search_box = QLineEdit()
    window.search_box.setReadOnly(True)
    window.search_box.setMaximumWidth(260)
    window._tip(window.search_box, f"Name of the active {app_name} project.")
    window.menu_bar = QMenuBar()
    window.menu_bar.setObjectName("AppMenuBar")
    window.file_menu = window.menu_bar.addMenu("File")
    window.new_project_action = window.file_menu.addAction("New Project", window.new_project)
    window.new_project_action.setShortcut("Ctrl+N")
    window.save_project_action = window.file_menu.addAction("Save Project", window.save_project)
    window.save_project_action.setShortcut("Ctrl+S")
    window.open_project_action = window.file_menu.addAction("Open Project", window.open_project)
    window.open_project_action.setShortcut("Ctrl+O")

    window.edit_menu = window.menu_bar.addMenu("Edit")
    for text, method, shortcut in (
        ("Undo", "undo", "Ctrl+Z"),
        ("Redo", "redo", "Ctrl+Shift+Z"),
        ("Cut", "cut", "Ctrl+X"),
        ("Copy", "copy", "Ctrl+C"),
        ("Paste", "paste", "Ctrl+V"),
        ("Select All", "selectAll", "Ctrl+A"),
    ):
        action = window.edit_menu.addAction(text)
        action.setShortcut(shortcut)
        action.triggered.connect(
            lambda _checked=False, method=method: window.edit_focused_widget(method)
        )
    window.theme_menu = window.edit_menu.addMenu("Themes")
    window.system_theme_action = window.theme_menu.addAction("System")
    window.system_theme_action.setCheckable(True)
    window.system_theme_action.triggered.connect(lambda: window.set_theme("system"))
    window.dark_theme_action = window.theme_menu.addAction("Dark")
    window.dark_theme_action.setCheckable(True)
    window.dark_theme_action.triggered.connect(lambda: window.set_theme("dark"))
    window.update_theme_actions()

    window.about_menu = window.menu_bar.addMenu("About")
    window.about_menu.addAction(f"About {app_name}", window.show_about_dialog)

    for name, text in (
        ("dataset_status", "Dataset: not prepared"),
        ("train_status", "Training: idle"),
        ("export_status", "Export: waiting"),
        ("chat_status", "Chat: no model loaded"),
    ):
        label = QLabel(text)
        label.setObjectName("TopStatus")
        label.setMaximumWidth(180)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        setattr(window, name, label)
    window.project_state = QLabel("Ready")
    window.project_state.setObjectName("Metric")
    window.project_state.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

    layout.addWidget(logo)
    layout.addSpacing(12)
    layout.addWidget(window.search_box)
    layout.addWidget(window.menu_bar)
    layout.addSpacing(10)
    for name in ("dataset_status", "train_status", "export_status", "chat_status"):
        layout.addWidget(getattr(window, name))
    layout.addStretch(1)
    layout.addWidget(window.project_state)
    return top


def build_side_rail(window) -> QWidget:
    """Build the shared navigation rail and connect page selection."""
    rail = QWidget()
    rail.setObjectName("SideRail")
    window.side_rail = rail
    rail.setFixedWidth(82)
    layout = QVBoxLayout(rail)
    layout.setContentsMargins(12, 18, 12, 18)
    layout.setSpacing(12)
    entries = (
        ("dataset_plan_nav", "Dataset Blueprint", "plan_tab_icon.png"),
        ("dataset_nav", "Ingestion", "ingestion_tab_icon.png"),
        ("training_nav", "Training", "AI_tab_icon.png"),
        ("fine_tune_nav", "Fine-tuning", "fine_tune_tab.png"),
        ("live_nav", "Live training", "live_tab_icon.png"),
        ("jobs_nav", "Job manager", "job_tab_icon.png"),
        ("benchmark_nav", "Benchmarks", "benchmark_tab_icon.png"),
        ("export_nav", "Export", "export_tab_icon.png"),
        ("chat_nav", "Chat", "chat_tab_icon.png"),
    )
    for index, (name, text, icon) in enumerate(entries):
        button = create_nav_button(icon, text)
        window._tip(button, f"Open {text}.")
        button.clicked.connect(lambda _checked=False, index=index: window._switch_page(index))
        setattr(window, name, button)
        layout.addWidget(button)
    window.dataset_plan_nav.setChecked(True)
    layout.addStretch(1)
    return rail


def build_main_shell(window, app_name: str) -> QWidget:
    """Assemble shared chrome and screen widgets into the application shell."""
    shell = QWidget()
    shell.setObjectName("AppShell")
    root = QVBoxLayout(shell)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(0)
    root.addWidget(build_top_bar(window, app_name))

    body = QHBoxLayout()
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(0)
    body.addWidget(build_side_rail(window))
    window.pages = QStackedWidget()
    for builder in SCREEN_BUILDERS:
        page = builder(window)
        page.setObjectName("Page")
        window.pages.addWidget(page)
    window.live_page_index = 4
    body.addWidget(window.pages, 1)
    root.addLayout(body, 1)
    return shell
