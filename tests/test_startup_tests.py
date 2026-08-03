from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_trainer.ui.app import _run_startup_tests


class StartupTestsReportingTests(unittest.TestCase):
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

            with patch("llm_trainer.ui.app.subprocess.Popen", return_value=process) as popen:
                _run_startup_tests(root, tests_root, reported.append)

            command = popen.call_args.args[0]
            self.assertIn("-v", command)
            self.assertEqual(
                reported,
                ["Test: test_resume (tests.test_resume_checks.ResumeChecksTests.test_resume) ..."],
            )


if __name__ == "__main__":
    unittest.main()
