from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtWidgets import QApplication

from interface.app import _run_startup_tests
from interface.startup import ProjectChoiceDialog, _run_startup_validations, is_dev_mode


class StartupTestsReportingTests(unittest.TestCase):
    def test_project_choice_dialog_constructs_after_ui_module_split(self) -> None:
        app = QApplication.instance() or QApplication([])
        dialog = ProjectChoiceDialog()
        self.assertIsNotNone(dialog)
        dialog.close()
        app.processEvents()

    def test_verbose_test_names_are_reported_to_splash_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tests_root = root / "tests"
            tests_root.mkdir()
            process = MagicMock()
            process.stdout = iter(
                [
                    "test_resume (tests.test_resume_checks.ResumeChecksTests.test_resume) ...\n",
                    "ok\n",
                    "\n",
                ]
            )
            process.wait.return_value = 0
            reported: list[str] = []

            with patch("interface.app.subprocess.Popen", return_value=process) as popen:
                _run_startup_tests(root, tests_root, reported.append)

            command = popen.call_args.args[0]
            self.assertIn("-v", command)
            self.assertEqual(
                reported,
                ["Test: test_resume (tests.test_resume_checks.ResumeChecksTests.test_resume) ..."],
            )

    def test_is_dev_mode_command_line_flags(self) -> None:
        self.assertTrue(is_dev_mode(["-dev"]))
        self.assertTrue(is_dev_mode(["--dev"]))
        self.assertTrue(is_dev_mode(["-DEV"]))
        self.assertTrue(is_dev_mode(["--DEV"]))
        self.assertTrue(is_dev_mode(["foo.py", "-dev", "bar"]))
        self.assertFalse(is_dev_mode([]))
        self.assertFalse(is_dev_mode(["--device", "cuda"]))
        self.assertFalse(is_dev_mode(["run_app.py"]))

    def test_is_dev_mode_environment_variable(self) -> None:
        with patch.dict(os.environ, {"DRUNKENBOT_DEV": "1"}, clear=False):
            with patch("sys.argv", ["run_app.py"]):
                self.assertTrue(is_dev_mode())

        with patch.dict(os.environ, {"DRUNKENBOT_DEV": "true"}, clear=False):
            with patch("sys.argv", ["run_app.py"]):
                self.assertTrue(is_dev_mode())

        with patch.dict(os.environ, {"DRUNKENBOT_DEV": "0", "DRUNKENBOT_DEV_MODE": ""}, clear=False):
            with patch("sys.argv", ["run_app.py"]):
                with patch.object(QApplication, "instance", return_value=None):
                    self.assertFalse(is_dev_mode())

    def test_run_startup_validations_dev_mode_false_skips_tests(self) -> None:
        splash = MagicMock()
        with patch("interface.startup_validation._run_startup_tests") as mock_run_tests:
            _run_startup_validations(splash, dev_mode=False)
            mock_run_tests.assert_not_called()
            splash.set_checks.assert_called_with([])

    def test_run_startup_validations_dev_mode_true_runs_tests(self) -> None:
        splash = MagicMock()
        with patch("interface.startup_validation._run_startup_tests") as mock_run_tests:
            _run_startup_validations(splash, dev_mode=True)
            mock_run_tests.assert_called_once()


if __name__ == "__main__":
    unittest.main()
