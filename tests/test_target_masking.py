"""Unit tests for prompt loss masking (target masking) in instruction fine-tuning."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from engine.config import ModelConfig, TrainingConfig
from engine.target_masking import (
    IGNORE_INDEX,
    encode_file_with_targets,
    find_completion_spans,
    mask_prompt_targets,
)
from engine.training_core import TokenDataset, split_tokens_to_files
from engine.training_impl import train_model


class TestTargetMasking(unittest.TestCase):
    def setUp(self) -> None:
        # Build a small byte-level BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(special_tokens=["<unk>", "<|endoftext|>", "<pad>"], vocab_size=200)
        sample_corpus = [
            "System: You are helpful.\nUser: Hello\nAssistant: Hi there!\n<|endoftext|>\n",
            "System: Follow constraints.\nUser: Do task.\nAssistant: Result 1.\n2. Result 2.\n<|endoftext|>\n",
        ]
        self.tokenizer.train_from_iterator(sample_corpus, trainer)

    def test_find_completion_spans_detects_assistant_spans(self) -> None:
        text = "System: Sys\nUser: Prompt\nAssistant: Here is the answer.\n<|endoftext|>"
        spans = find_completion_spans(text)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertIn("Here is the answer.", text[start:end])
        self.assertTrue(text[start:end].endswith("<|endoftext|>"))

    def test_find_completion_spans_multi_turn(self) -> None:
        text = (
            "System: Sys\n"
            "User: Turn 1\n"
            "Assistant: Answer 1\n"
            "User: Turn 2\n"
            "Assistant: Answer 2\n"
            "<|endoftext|>"
        )
        spans = find_completion_spans(text)
        self.assertEqual(len(spans), 2)
        span1_text = text[spans[0][0]:spans[0][1]]
        span2_text = text[spans[1][0]:spans[1][1]]
        self.assertIn("Answer 1", span1_text)
        self.assertNotIn("Turn 2", span1_text)
        self.assertIn("Answer 2", span2_text)

    def test_find_completion_spans_fallback_for_pretraining_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        spans = find_completion_spans(text)
        self.assertEqual(spans, [(0, len(text))])

    def test_mask_prompt_targets(self) -> None:
        text = "System: Sys\nUser: Question\nAssistant: Answer!\n<|endoftext|>"
        enc = self.tokenizer.encode(text)
        tids = enc.ids
        targets = mask_prompt_targets(text, self.tokenizer, tids, enc.offsets)
        self.assertEqual(len(targets), len(tids))

        # First tokens (System, User) must be IGNORE_INDEX
        self.assertEqual(targets[0], IGNORE_INDEX)
        # Last token (<|endoftext|>) must be included in targets
        self.assertEqual(targets[-1], tids[-1])
        # Count masked vs unmasked
        masked_count = sum(1 for t in targets if t == IGNORE_INDEX)
        unmasked_count = sum(1 for t in targets if t != IGNORE_INDEX)
        self.assertGreater(masked_count, 0)
        self.assertGreater(unmasked_count, 0)

    def test_encode_and_split_with_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path = tmp / "corpus.txt"
            corpus_path.write_text(
                "System: Sys\nUser: Q1\nAssistant: A1\n<|endoftext|>\n"
                "System: Sys\nUser: Q2\nAssistant: A2\n<|endoftext|>\n",
                encoding="utf-8",
            )
            tokens_path = tmp / "all_tokens.npy"
            targets_path = tmp / "all_targets.npy"

            token_count = encode_file_with_targets(
                self.tokenizer,
                corpus_path,
                tokens_path,
                targets_path,
                tokens_dtype=np.dtype(np.uint16),
                targets_dtype=np.dtype(np.int32),
            )
            self.assertGreater(token_count, 0)
            self.assertTrue(tokens_path.exists())
            self.assertTrue(targets_path.exists())

            tokens = np.load(tokens_path, mmap_mode="r")
            targets = np.load(targets_path, mmap_mode="r")
            self.assertEqual(len(tokens), len(targets))
            self.assertIn(IGNORE_INDEX, targets)

            # Test split_tokens_to_files with targets
            train_tokens_path = tmp / "train_tokens.npy"
            val_tokens_path = tmp / "val_tokens.npy"
            train_targets_path = tmp / "train_targets.npy"
            val_targets_path = tmp / "val_targets.npy"

            train_count, val_count = split_tokens_to_files(
                tokens,
                train_tokens_path,
                val_tokens_path,
                validation_split=0.2,
                dtype=np.dtype(np.uint16),
                targets=targets,
                train_targets_path=train_targets_path,
                val_targets_path=val_targets_path,
                targets_dtype=np.dtype(np.int32),
            )
            self.assertEqual(train_count + val_count, token_count)
            train_targs = np.load(train_targets_path, mmap_mode="r")
            self.assertEqual(len(train_targs), train_count)

            # Release memmap file handles for clean Windows deletion
            del tokens
            del targets
            del train_targs

    def test_token_dataset_returns_targets_when_provided(self) -> None:
        tokens = list(range(10, 30))
        # Mask first 5 targets with IGNORE_INDEX
        targets = [IGNORE_INDEX if i < 5 else tokens[i] for i in range(len(tokens))]
        ds = TokenDataset(tokens=tokens, context_length=4, stride=1, targets=targets)
        x, y = ds[0]
        # x is tokens 0..3: [10, 11, 12, 13]
        # y is targets 1..4: [-100, -100, -100, -100]
        self.assertTrue(torch.equal(x, torch.tensor([10, 11, 12, 13])))
        self.assertTrue(torch.equal(y, torch.tensor([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX])))

        # Later slice where targets are unmasked
        x_later, y_later = ds[6]
        # targets 7..10 should have real token IDs
        self.assertFalse((y_later == IGNORE_INDEX).all())

    def test_train_model_with_targets_gradient_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_config = ModelConfig(
                vocab_size=200,
                context_length=8,
                layer_count=2,
                head_count=2,
                embedding_size=32,
            )
            training_config = TrainingConfig(
                output_dir=tmp / "run",
                device="cpu",
                epochs=1,
                batch_size=2,
                learning_rate=0.01,
                weight_decay=0.0,
                optimizer_name="adamw",
            )
            # Create synthetic sequence where prompt tokens are masked to -100
            seq_len = 32
            tokens = [i % 150 for i in range(seq_len)]
            targets = [IGNORE_INDEX if i < 16 else tokens[i] for i in range(seq_len)]

            result = train_model(
                model_config=model_config,
                training_config=training_config,
                train_tokens=tokens,
                val_tokens=tokens,
                pad_token_id=2,
                train_targets=targets,
                val_targets=targets,
            )
            self.assertTrue(result.checkpoint_path.exists())
            self.assertTrue(result.summary_path.exists())


    def test_find_completion_spans_with_eos_tag(self) -> None:
        text = "System: Sys\nUser: Prompt\nAssistant: Hello there!\n<eos>"
        spans = find_completion_spans(text)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertIn("Hello there!", text[start:end])
        self.assertTrue(text[start:end].endswith("<eos>"))

    def test_instruction_dataset_discrete_boundaries(self) -> None:
        from engine.target_masking import InstructionDataset, collate_instruction_batch
        # Tokens: Sample 1 (10, 11, EOS=1), Sample 2 (20, 21, 22, EOS=1)
        tokens = np.array([10, 11, 1, 20, 21, 22, 1], dtype=np.int64)
        targets = np.array([-100, 11, 1, -100, -100, 22, 1], dtype=np.int64)
        ds = InstructionDataset(tokens, targets=targets, context_length=16, eos_token_id=1)
        self.assertEqual(len(ds), 2)
        x0, y0 = ds[0]
        # x0 should be [10, 11], y0 should be [11, 1]
        self.assertTrue(torch.equal(x0, torch.tensor([10, 11])))
        self.assertTrue(torch.equal(y0, torch.tensor([11, 1])))

        x1, y1 = ds[1]
        self.assertTrue(torch.equal(x1, torch.tensor([20, 21, 22])))
        self.assertTrue(torch.equal(y1, torch.tensor([-100, 22, 1])))

        # Test batch collation with padding
        batch_x, batch_y = collate_instruction_batch([(x0, y0), (x1, y1)], pad_token_id=0)
        self.assertEqual(batch_x.shape, (2, 3))
        self.assertEqual(batch_y.shape, (2, 3))
        # First item padded to length 3 with pad_token_id=0 for input and IGNORE_INDEX for target
        self.assertEqual(batch_x[0].tolist(), [10, 11, 0])
        self.assertEqual(batch_y[0].tolist(), [11, 1, IGNORE_INDEX])
        self.assertEqual(batch_x[1].tolist(), [20, 21, 22])
        self.assertEqual(batch_y[1].tolist(), [-100, 22, 1])

    def test_checkpoint_records_fine_tune_base(self) -> None:
        from engine.model import MicroGPT
        from engine.training_checkpoint import save_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_config = ModelConfig(vocab_size=100, context_length=8, layer_count=1, head_count=1, embedding_size=16)
            training_config = TrainingConfig(
                output_dir=tmp,
                training_mode="fine_tune",
                fine_tune_from_checkpoint=tmp / "base.pt",
            )
            model = MicroGPT(model_config)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            from torch.amp import GradScaler
            scaler = GradScaler("cpu", enabled=False)
            ckpt_path = tmp / "ckpt.pt"
            save_checkpoint(ckpt_path, model, opt, opt, scaler, model_config, training_config, 1, 1, 1.0, 1.0)
            loaded = torch.load(ckpt_path, map_location="cpu")
            self.assertEqual(loaded.get("fine_tune_base_checkpoint"), str(tmp / "base.pt"))


if __name__ == "__main__":
    unittest.main()

