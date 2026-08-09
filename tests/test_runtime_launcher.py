from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from packaging.runtime_launcher import private_python


class RuntimeLauncherTests(unittest.TestCase):
    def test_windows_runtime_path(self) -> None:
        with patch("packaging.runtime_launcher.platform.system", return_value="Windows"):
            self.assertEqual(private_python(Path("app")), Path("app/runtime/python.exe"))

    def test_posix_runtime_path(self) -> None:
        with patch("packaging.runtime_launcher.platform.system", return_value="Linux"):
            self.assertEqual(private_python(Path("app")), Path("app/runtime/bin/python"))


if __name__ == "__main__":
    unittest.main()
