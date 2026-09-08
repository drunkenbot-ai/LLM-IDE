from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from engine.config import ModelConfig, TrainingConfig
from engine.training_impl import _try_compile_model


class CompileModelTests(unittest.TestCase):
    def test_compile_model_defaults_to_false(self) -> None:
        config = TrainingConfig(output_dir=Path("tmp"))
        self.assertFalse(config.compile_model)

    def test_try_compile_model_returns_eager_when_disabled(self) -> None:
        model = torch.nn.Linear(10, 10)
        compiled = _try_compile_model(model, "cuda", enabled=False)
        self.assertIs(compiled, model)

    def test_try_compile_model_skips_on_cpu(self) -> None:
        model = torch.nn.Linear(10, 10)
        compiled = _try_compile_model(model, "cpu", enabled=True)
        self.assertIs(compiled, model)

    @patch("torch.compile", side_effect=RuntimeError("Inductor compiler error"))
    def test_try_compile_model_catches_compiler_error_and_falls_back(self, _mock_compile) -> None:
        model = torch.nn.Linear(10, 10)
        with patch("torch.cuda.is_available", return_value=True):
            compiled = _try_compile_model(model, "cuda", enabled=True)
        self.assertIs(compiled, model)


if __name__ == "__main__":
    unittest.main()
