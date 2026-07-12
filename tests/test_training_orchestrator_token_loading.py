from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from llm_trainer.training_orchestrator import _load_tokens_for_training
from llm_trainer.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, save_tokenizer_package
from tokenizers import Tokenizer
from tokenizers.models import BPE


class TrainingOrchestratorTokenLoadingTests(unittest.TestCase):
    def test_writes_standard_tokenizer_sidecar_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer_path = Path(tmp) / "tokenizer.json"
            tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
            tokenizer.add_special_tokens([PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN])
            tokenizer.save(str(tokenizer_path))
            save_tokenizer_package(tokenizer, tokenizer_path, model_max_length=512)
            config = json.loads((tokenizer_path.parent / "tokenizer_config.json").read_text(encoding="utf-8"))
            token_map = json.loads((tokenizer_path.parent / "special_tokens_map.json").read_text(encoding="utf-8"))
            self.assertEqual(config["tokenizer_class"], "PreTrainedTokenizerFast")
            self.assertEqual(config["model_max_length"], 512)
            self.assertIn("chat_template", config)
            self.assertEqual(token_map["pad_token"], PAD_TOKEN)
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
