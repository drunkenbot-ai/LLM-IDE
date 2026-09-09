"""Unit tests for code fine-tuning workflow, dataset loading, and target masking."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from engine.config import DatasetConfig
from engine.data_core import (
    load_jsonl_documents,
    load_structured_json_documents,
)
from engine.dataset_build import build_dataset
from engine.target_masking import IGNORE_INDEX, find_completion_spans
import interface.app  # noqa: F401
from interface.screens.fine_tuning_screen import FineTuningScreenMixin
from interface.screens.training_screen import TrainingScreenMixin


class _Combo:
    """Minimal combo-box test double."""

    def __init__(self, text: str) -> None:
        self._text = text

    def currentText(self) -> str:
        return self._text

    def findText(self, text: str) -> int:
        return 0

    def setCurrentIndex(self, index: int) -> None:
        pass


class _ValueHolder:
    """Minimal value spinbox test double."""

    def __init__(self, val: float | int = 0) -> None:
        self._val = val

    def value(self) -> float | int:
        return self._val

    def setValue(self, val: float | int) -> None:
        self._val = val

    def setText(self, text: str) -> None:
        pass

    def maximum(self) -> float | int:
        return 128


class _CheckedHolder:
    """Minimal checkbox test double."""

    def __init__(self, checked: bool = False) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        self._checked = checked


class TestCodeFineTune(unittest.TestCase):
    def test_load_jsonl_documents_preserves_indentation_and_newlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "code.jsonl"
            rec = {
                "messages": [
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": "Write Python code to solve problem."},
                    {
                        "role": "assistant",
                        "content": "```python\ndef solve(data):\n    if not data:\n        return []\n    return [x * 2 for x in data]\n```\nExplanation: Doubled.",
                    },
                ]
            }
            source.write_text(json.dumps(rec) + "\n", encoding="utf-8")

            docs = load_jsonl_documents(source)
            self.assertEqual(len(docs), 1)
            text = docs[0].text

            # Ensure newlines and indentation were not destroyed by clean_text
            self.assertIn("\ndef solve(data):\n", text)
            self.assertIn("    if not data:\n", text)
            self.assertIn("        return []\n", text)

    def test_load_structured_json_documents_supports_code_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "bash.jsonl"
            rec = {
                "messages": [
                    {"role": "system", "content": "You are a bash assistant."},
                    {"role": "user", "content": "Write a bash function."},
                    {
                        "role": "assistant",
                        "content": "```sh\ncheck_file() {\n  local f=\"$1\"\n  test -f \"$f\"\n}\n```",
                    },
                ]
            }
            source.write_text(json.dumps(rec) + "\n", encoding="utf-8")

            docs = load_structured_json_documents(source, "code")
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].kind, "code")
            self.assertIn("check_file() {\n  local f=", docs[0].text)

    def test_code_fine_tune_target_masking_spans(self) -> None:
        text = (
            "System: You are a helpful bash programming assistant.\n"
            "User: Write Bash code to parse input.\n"
            "Assistant: ```sh\n"
            "count_bash() {\n"
            "  local path=\"$1\"\n"
            "  grep -c \"ERROR\" \"$path\"\n"
            "}\n"
            "```\n"
            "Explanation: Done.\n"
            "<eos>"
        )
        spans = find_completion_spans(text)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        completion_text = text[start:end]

        # Completion target must start with the code block
        self.assertTrue(completion_text.startswith("```sh\ncount_bash()"))
        # System prompt and User query must NOT be in completion target
        self.assertNotIn("System:", completion_text)
        self.assertNotIn("User:", completion_text)
        self.assertNotIn("Write Bash code", completion_text)

    def test_build_dataset_with_code_stage_creates_target_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_dir = tmp / "input"
            in_dir.mkdir()
            code_file = in_dir / "code.jsonl"
            rec = {
                "messages": [
                    {"role": "system", "content": "Coding assistant."},
                    {"role": "user", "content": "Return 42."},
                    {"role": "assistant", "content": "```python\ndef answer():\n    return 42\n```"},
                ]
            }
            code_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            out_dir = tmp / "output"

            config = DatasetConfig(
                input_dir=in_dir,
                output_dir=out_dir,
                vocab_size=128,
                dataset_stage="code",
                code_dataset_path=code_file,
                code_dataset_paths=[code_file],
                max_workers=0,
            )
            result = build_dataset(config)
            self.assertTrue(result.output_dir.exists())

            train_tokens_path = out_dir / "train_tokens.npy"
            train_targets_path = out_dir / "train_targets.npy"

            self.assertTrue(train_tokens_path.exists())
            self.assertTrue(train_targets_path.exists())

            targets = np.load(train_targets_path)
            masked_count = int(np.sum(targets == IGNORE_INDEX))
            unmasked_count = int(np.sum(targets != IGNORE_INDEX))

            self.assertGreater(masked_count, 0)
            self.assertGreater(unmasked_count, 0)

    def test_fine_tuning_screen_code_defaults(self) -> None:
        window = FineTuningScreenMixin()
        window.training_mode = _Combo("Code fine-tune")
        self.assertEqual(window._training_mode_value(), "fine_tune")
        self.assertEqual(window._training_stage_value(), "code")

        # Mock controls for apply_recommended_fine_tune_settings
        window.peft_method = _Combo("LoRA adapters")
        window.lora_targets = _Combo("Attention + MLP")
        window.lora_rank = _ValueHolder(8)
        window.lora_alpha = _ValueHolder(16.0)
        window.lora_dropout = _ValueHolder(0.05)
        window.learning_rate = _ValueHolder(0.0001)
        window.max_grad_norm = _ValueHolder(1.0)
        window.weight_decay = _ValueHolder(0.01)
        window.epochs = _ValueHolder(3)
        window.scheduler_name = _Combo("cosine")
        window.fine_tune_preview = _ValueHolder(0)
        window.fine_tune_checkpoint = type("MockCP", (), {"text": lambda s: ""})()

        def _set_combo_text(combo: _Combo, text: str) -> None:
            combo._text = text

        def _set_combo_by_data(combo: _Combo, data: str, mapping: dict) -> None:
            pass

        window._set_combo_text = _set_combo_text
        window._set_combo_by_data = _set_combo_by_data
        window._update_training_mode_controls = lambda: None

        window.apply_recommended_fine_tune_settings()

        self.assertEqual(window.lora_targets.currentText(), "Attention + MLP")
        self.assertEqual(window.lora_rank.value(), 16)
        self.assertEqual(window.lora_alpha.value(), 32.0)
        self.assertEqual(window.learning_rate.value(), 0.00005)

    def test_training_screen_profile_code_fine_tune(self) -> None:
        window = TrainingScreenMixin()
        window.training_profile = _Combo("Code fine-tune")
        window.optimizer_name = _Combo("AdamW")
        window.scheduler_name = _Combo("Cosine decay")
        window.learning_rate = _ValueHolder(0.001)
        window.weight_decay = _ValueHolder(0.01)
        window.min_lr_ratio = _ValueHolder(0.1)
        window.polynomial_power = _ValueHolder(1.0)
        window.max_grad_norm = _ValueHolder(1.0)
        window.precision = _Combo("FP32")
        window.use_amp = _CheckedHolder(False)
        window.activation_checkpointing = _CheckedHolder(True)
        window.batch_size = _ValueHolder(8)
        window.gradient_accumulation = _ValueHolder(2)
        window.data_loader_workers = _ValueHolder(2)
        window.warmup_steps = _ValueHolder(100)
        window.dropout = _ValueHolder(0.1)
        window.early_stopping_patience = _ValueHolder(5)
        window.epochs = _ValueHolder(5)
        window.eval_batches = _ValueHolder(50)
        window.context_length = _ValueHolder(512)
        window.n_embd = _ValueHolder(256)
        window.n_layer = _ValueHolder(6)
        window.sample_stride = _ValueHolder(128)
        window.eval_interval = _ValueHolder(50)
        window.max_eval_batches = _ValueHolder(50)
        window.save_interval = _ValueHolder(500)
        window.training_mode = _Combo("Pretrain from scratch")
        window.peft_method = _Combo("Full fine-tune")
        window.lora_rank = _ValueHolder(8)
        window.lora_alpha = _ValueHolder(16.0)
        window.lora_dropout = _ValueHolder(0.05)
        window.lora_targets = _Combo("Attention projections")

        def _set_combo_text(combo: _Combo, text: str) -> None:
            combo._text = text

        window._set_combo_text = _set_combo_text
        window._update_training_mode_controls = lambda: None
        window.refresh_model_estimate = lambda: None
        window.training_log = []

        window.apply_training_profile()

        self.assertEqual(window.training_mode.currentText(), "Code fine-tune")
        self.assertEqual(window.peft_method.currentText(), "LoRA adapters")
        self.assertEqual(window.lora_targets.currentText(), "Attention + MLP")
        self.assertEqual(window.lora_rank.value(), 16)
        self.assertEqual(window.lora_alpha.value(), 32.0)
        self.assertGreater(window.batch_size.value(), 0)
        self.assertGreater(window.gradient_accumulation.value(), 0)


if __name__ == "__main__":
    unittest.main()
