"""Progress backlog coalescing helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def coalesce_final_events(
    events: Iterable[object],
    is_terminal: Callable[[dict[str, Any]], bool],
) -> list[object]:
    """Keep terminal events plus the latest ordinary event in source order."""

    event_list = list(events)
    latest_ordinary_index = next(
        (
            index
            for index in range(len(event_list) - 1, -1, -1)
            if isinstance(event_list[index], dict)
            and not is_terminal(event_list[index])
        ),
        None,
    )
    return [
        event
        for index, event in enumerate(event_list)
        if not isinstance(event, dict)
        or is_terminal(event)
        or index == latest_ordinary_index
    ]
