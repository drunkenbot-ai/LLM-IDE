"""Compatibility entry point for the engine command-line interface.

The implementation lives in :mod:`engine.cli`; this module keeps the
historical ``python -m llm_trainer.cli`` command working for existing users.
"""

from __future__ import annotations

from engine.cli import *  # noqa: F401,F403
from engine.cli import main


if __name__ == "__main__":
    main()
