"""Standalone entry point for Cluster Coordinator daemon."""
from __future__ import annotations

import sys
from cluster.coordinator import main

if __name__ == "__main__":
    sys.exit(main())
