from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
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


def create_nav_button(icon_name: str, tooltip: str) -> QPushButton:
    """Create a checkable side-rail navigation button."""
    button = QPushButton()
    icon_path = Path(__file__).resolve().parent.parent / "icons" / icon_name
    if icon_path.is_file():
        button.setIcon(QIcon(str(icon_path)))
        button.setProperty("_nav_icon_path", str(icon_path))
    button.setIconSize(QSize(32, 32))
    button.setMinimumHeight(44)
    button.setToolTip(tooltip)
    button.setAccessibleName(tooltip)
    button.setObjectName("NavButton")
    button.setCheckable(True)
    return button


def update_navigation_icons(window) -> None:
    """Recolor white navigation artwork for the selected application theme."""
    color = QColor("#dddddd" if window.theme_name == "dark" else "#202020")
    for button in window.side_rail.findChildren(QPushButton, "NavButton"):
        icon_path = Path(str(button.property("_nav_icon_path") or ""))
        if not icon_path.is_file() or "hero_" in icon_path.name:
            continue
        source = QPixmap(str(icon_path))
        image = source.toImage().convertToFormat(QImage.Format_ARGB32)
        for y_coord in range(image.height()):
            for x_coord in range(image.width()):
                pixel = image.pixelColor(x_coord, y_coord)
                if pixel.alpha():
                    pixel.setRed(color.red())
                    pixel.setGreen(color.green())
                    pixel.setBlue(color.blue())
                    image.setPixelColor(x_coord, y_coord, pixel)
        button.setIcon(QIcon(QPixmap.fromImage(image)))


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

    window.edit_menu.addSeparator()
    window.plugins_action = window.edit_menu.addAction("Plugins & Extensions...", window.open_plugins_dialog)

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
    """Build the shared navigation rail and connect page selection strictly matching Reference Images."""
    from interface.widgets.neon_nav_card import CompactNavButton, NeonNavCard

    rail = QWidget()
    rail.setObjectName("SideRail")
    window.side_rail = rail
    window.sidebar_expanded = False
    rail.setFixedWidth(102)
    layout = QVBoxLayout(rail)
    layout.setContentsMargins(8, 10, 8, 10)
    layout.setSpacing(6)

    toggle_row = QHBoxLayout()
    toggle_row.setContentsMargins(0, 0, 0, 0)
    window.side_rail_toggle = QPushButton("☰")
    window.side_rail_toggle.setFixedSize(28, 22)
    window.side_rail_toggle.setStyleSheet(
        "QPushButton { background-color: #151821; color: #a5b4fc; border: 1px solid #282e42; border-radius: 4px; font-size: 13px; font-weight: bold; }"
        "QPushButton:hover { background-color: #23283a; color: white; border-color: #8b5cf6; }"
    )
    window.side_rail_toggle.setToolTip("Navigation menu")
    window.side_rail_toggle.clicked.connect(window.toggle_side_rail)
    toggle_row.addStretch(1)
    toggle_row.addWidget(window.side_rail_toggle)
    layout.addLayout(toggle_row)

    icons_dir = Path(__file__).resolve().parent.parent / "icons"

    # Top items matching Reference Image 1
    window.dataset_nav = CompactNavButton("Datasets", "ingestion_tab_icon.png")
    window._tip(window.dataset_nav, "Open Ingestion & Datasets.")
    window.dataset_nav.clicked.connect(lambda: window._switch_page(2))
    layout.addWidget(window.dataset_nav)

    window.dataset_plan_nav = CompactNavButton("Today", "plan_tab_icon.png")
    window._tip(window.dataset_plan_nav, "Open Dataset Blueprint.")
    window.dataset_plan_nav.clicked.connect(lambda: window._switch_page(0))
    layout.addWidget(window.dataset_plan_nav)

    window.benchmark_nav = CompactNavButton("UI", "benchmark_tab_icon.png")
    window._tip(window.benchmark_nav, "Open Benchmarks & UI.")
    window.benchmark_nav.clicked.connect(lambda: window._switch_page(7))
    layout.addWidget(window.benchmark_nav)

    layout.addSpacing(6)

    # 3 Glowing Hero Cards matching Reference Images 1 & 2
    # 1. Dataset Recipe (Amber cloche)
    window.dataset_recipe_nav = NeonNavCard(
        title="Dataset Recipe",
        icon_path=str(icons_dir / "hero_dataset_recipe.png"),
        glow_color="#f59e0b",
    )
    window._tip(window.dataset_recipe_nav, "Open Dataset Recipe Matrix.")
    window.dataset_recipe_nav.clicked.connect(lambda: window._switch_page(1))
    layout.addWidget(window.dataset_recipe_nav)

    # 2. Architecture Studio (Purple brain)
    window.training_nav = NeonNavCard(
        title="Architecture",
        icon_path=str(icons_dir / "hero_architecture.png"),
        glow_color="#8b5cf6",
    )
    window._tip(window.training_nav, "Open Model Architecture Studio.")
    window.training_nav.clicked.connect(lambda: window._switch_page(3))
    layout.addWidget(window.training_nav)

    # 3. Compute Engine (Cyan chip)
    window.fine_tune_nav = NeonNavCard(
        title="Compute Engine",
        icon_path=str(icons_dir / "hero_compute_engine.png"),
        glow_color="#06b6d4",
    )
    window._tip(window.fine_tune_nav, "Open Compute & Runtime Engine.")
    window.fine_tune_nav.clicked.connect(lambda: window._switch_page(4))
    layout.addWidget(window.fine_tune_nav)

    layout.addSpacing(6)

    # Bottom items matching Reference Images 1 & 2
    window.live_nav = CompactNavButton("Training", "live_tab_icon.png")
    window._tip(window.live_nav, "Open Live Training.")
    window.live_nav.clicked.connect(lambda: window._switch_page(5))
    layout.addWidget(window.live_nav)

    window.jobs_nav = CompactNavButton("Deploy", "job_tab_icon.png")
    window._tip(window.jobs_nav, "Open Job Manager & Deploy.")
    window.jobs_nav.clicked.connect(lambda: window._switch_page(6))
    layout.addWidget(window.jobs_nav)

    window.export_nav = CompactNavButton("Docs", "export_tab_icon.png")
    window._tip(window.export_nav, "Open Model Export & Docs.")
    window.export_nav.clicked.connect(lambda: window._switch_page(8))
    layout.addWidget(window.export_nav)

    window.chat_nav = CompactNavButton("Settings", "chat_tab_icon.png")
    window._tip(window.chat_nav, "Open Settings & Chat.")
    window.chat_nav.clicked.connect(lambda: window._switch_page(9))
    layout.addWidget(window.chat_nav)

    # Aliases for direct semantic access
    window.architecture_nav = window.training_nav
    window.compute_nav = window.fine_tune_nav
    window.compute_engine_nav = window.fine_tune_nav
    window.recipe_nav = window.dataset_recipe_nav

    window.dataset_plan_nav.setChecked(True)
    layout.addStretch(1)

    if hasattr(window, "_refresh_plugin_navigation"):
        window._refresh_plugin_navigation()
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
    from interface.screens.screen_builders import SCREEN_BUILDERS

    window.pages = QStackedWidget()
    for builder in SCREEN_BUILDERS:
        page = builder(window)
        page.setObjectName("Page")
        window.pages.addWidget(page)
    window.blueprint_page_index = 0
    window.recipe_page_index = 1
    window.ingestion_page_index = 2
    window.training_page_index = 3
    window.fine_tuning_page_index = 4
    window.live_page_index = 5
    window.job_manager_page_index = 6
    window.benchmark_page_index = 7
    window.export_page_index = 8
    window.chat_page_index = 9
    body.addWidget(window.pages, 1)
    root.addLayout(body, 1)

    if hasattr(window, "connect_cluster_training_sync"):
        window.connect_cluster_training_sync()

    return shell
