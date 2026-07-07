from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from llm_trainer.training_orchestrator import _load_tokens_for_training


class TrainingOrchestratorTokenLoadingTests(unittest.TestCase):
    def test_prefers_npy_tokens_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            np.save(data_dir / "train_tokens.npy", np.asarray([1, 2, 3], dtype=np.int32))
            np.save(data_dir / "val_tokens.npy", np.asarray([4, 5], dtype=np.int32))
            (data_dir / "train_tokens.json").write_text(json.dumps([9]), encoding="utf-8")
            (data_dir / "val_tokens.json").write_text(json.dumps([9]), encoding="utf-8")

            train_tokens, val_tokens = _load_tokens_for_training(data_dir)
            self.assertEqual(np.asarray(train_tokens).tolist(), [1, 2, 3])
            self.assertEqual(np.asarray(val_tokens).tolist(), [4, 5])
            del train_tokens
            del val_tokens

    def test_falls_back_to_json_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "train_tokens.json").write_text(json.dumps([7, 8, 9]), encoding="utf-8")
            (data_dir / "val_tokens.json").write_text(json.dumps([10]), encoding="utf-8")

            train_tokens, val_tokens = _load_tokens_for_training(data_dir)
            self.assertEqual(train_tokens, [7, 8, 9])
            self.assertEqual(val_tokens, [10])


if __name__ == "__main__":
    unittest.main()
