#!/usr/bin/env python3
"""CLI utility to stream frontier training datasets directly into cluster-safe 28MB shards."""

import sys
from pathlib import Path

# Ensure LLM-IDE repository root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.dataset_streamer import main

if __name__ == "__main__":
    main()
