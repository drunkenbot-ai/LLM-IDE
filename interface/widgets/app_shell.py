from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    window.search_box.setPlaceholderText("Project name...")
    window.search_box.setMaximumWidth(260)
    window._tip(window.search_box, f"Project name used when saving or reopening a {app_name} project.")
    for name, text, slot in (
        ("new_project_button", "New Project", window.new_project),
        ("save_project_button", "Save Project", window.save_project),
        ("open_project_button", "Open Project", window.open_project),
    ):
        button = QPushButton(text)
        button.setMaximumWidth(130)
        button.clicked.connect(slot)
        window._tip(button, f"{text} in the current project.")
        setattr(window, name, button)

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
    for name in ("new_project_button", "save_project_button", "open_project_button"):
        layout.addWidget(getattr(window, name))
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
        window.pages.addWidget(builder(window))
    window.live_page_index = 4
    body.addWidget(window.pages, 1)
    root.addLayout(body, 1)
    return shell
