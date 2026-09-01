"""Cross-platform termination guarded by durable process identity."""

from __future__ import annotations

from typing import Any

from engine.training_worker_protocol import manifest_process_is_current

import psutil


def terminate_verified_process(manifest: dict[str, Any]) -> None:
    """Terminate only a process whose durable creation identity still matches."""
    if not manifest_process_is_current(manifest):
        raise RuntimeError("Training process identity is no longer verifiable; refusing force stop")
    pid = int(manifest["pid"])
    process = psutil.Process(pid)
    expected = dict(manifest.get("process_identity") or {})
    actual = {
        "kind": "psutil-create-time",
        "value": f"{process.create_time():.6f}",
    }
    if actual != expected:
        raise RuntimeError("Training process identity changed before force stop; refusing")
    process.terminate()
