from __future__ import annotations

import unittest
from unittest.mock import patch

import run_app


class StartupBootstrapTests(unittest.TestCase):
    def test_app_module_is_loaded_lazily(self) -> None:
        result: dict[str, object] = {}

        with patch("run_app.importlib.import_module", return_value="loaded") as importer:
            run_app._load_app_module(result)

        importer.assert_called_once_with("llm_trainer.ui.app")
        self.assertEqual(result["module"], "loaded")
        self.assertNotIn("error", result)

    def test_import_errors_are_reported_to_bootstrap(self) -> None:
        error = RuntimeError("dependency failed")
        result: dict[str, object] = {}

        with patch("run_app.importlib.import_module", side_effect=error):
            run_app._load_app_module(result)

        self.assertIs(result["error"], error)


if __name__ == "__main__":
    unittest.main()
