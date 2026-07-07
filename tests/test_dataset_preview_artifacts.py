from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_trainer.dataset_preview import _has_prepared_token_artifacts


class DatasetPreviewArtifactsTests(unittest.TestCase):
    def test_detects_npy_token_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            (dataset_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (dataset_dir / "train_tokens.npy").write_bytes(b"npy")
            (dataset_dir / "val_tokens.npy").write_bytes(b"npy")
            self.assertTrue(_has_prepared_token_artifacts(dataset_dir))

    def test_detects_json_token_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            (dataset_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (dataset_dir / "train_tokens.json").write_text("[]", encoding="utf-8")
            (dataset_dir / "val_tokens.json").write_text("[]", encoding="utf-8")
            self.assertTrue(_has_prepared_token_artifacts(dataset_dir))


if __name__ == "__main__":
    unittest.main()
