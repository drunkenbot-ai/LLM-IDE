from __future__ import annotations

import unittest
from unittest.mock import patch

from packaging.runtime_setup import choose_runtime


class RuntimeSelectionTests(unittest.TestCase):
    def test_new_nvidia_driver_selects_cuda_124(self) -> None:
        self.assertEqual(choose_runtime("Windows", 550).profile, "cu124")

    def test_older_supported_driver_selects_cuda_121(self) -> None:
        self.assertEqual(choose_runtime("Linux", 525).profile, "cu121")

    def test_missing_driver_selects_cpu(self) -> None:
        with patch("packaging.runtime_setup._nvidia_driver_major", return_value=None):
            self.assertEqual(choose_runtime("Windows").profile, "cpu")

    def test_macos_selects_cpu(self) -> None:
        self.assertEqual(choose_runtime("Darwin", 600).profile, "cpu")


if __name__ == "__main__":
    unittest.main()
