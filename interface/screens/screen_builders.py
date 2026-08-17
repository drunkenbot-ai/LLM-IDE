from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QWidget

from interface.tabs.benchmark_tab import build_benchmark_tab
from interface.tabs.chat_tab import build_chat_tab
from interface.tabs.dataset_plan_tab import build_dataset_plan_tab
from interface.tabs.dataset_tab import build_dataset_tab
from interface.tabs.export_tab import build_export_tab
from interface.tabs.fine_tuning_tab import build_fine_tuning_tab
from interface.tabs.job_manager_tab import build_job_manager_tab
from interface.tabs.live_tab import build_live_training_tab
from interface.tabs.training_tab import build_training_tab

ScreenBuilder = Callable[[object], QWidget]

SCREEN_BUILDERS: tuple[ScreenBuilder, ...] = (
    build_dataset_plan_tab,
    build_dataset_tab,
    build_training_tab,
    build_fine_tuning_tab,
    build_live_training_tab,
    build_job_manager_tab,
    build_benchmark_tab,
    build_export_tab,
    build_chat_tab,
)
