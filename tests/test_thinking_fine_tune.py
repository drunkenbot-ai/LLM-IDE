"""Unit tests for thinking fine-tuning workflow, dataset loading, target masking, and UI controls."""

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


class _MockWindow(FineTuningScreenMixin, TrainingScreenMixin):
    def __init__(self, training_mode: str = "Thinking fine-tune") -> None:
        self.training_mode = _Combo(training_mode)
        self.peft_method = _Combo("LoRA adapters")
        self.lora_targets = _Combo("Attention projections")
        self.lora_rank = _ValueHolder(8)
        self.lora_alpha = _ValueHolder(16.0)
        self.lora_dropout = _ValueHolder(0.05)
        self.learning_rate = _ValueHolder(0.0001)
        self.max_grad_norm = _ValueHolder(1.0)
        self.epochs = _ValueHolder(5)
        self.fine_tune_checkpoint = _ValueHolder()
        self.fine_tune_check_button = _CheckedHolder()
        self.fine_tune_output_dir = _Combo("")

    def _set_combo_text(self, combo: Any, text: str) -> None:
        combo._text = text

    def refresh_fine_tune_workflow(self) -> None:
        pass

    def _refresh_fine_tune_default_output(self) -> None:
        pass


class TestThinkingFineTune(unittest.TestCase):
    def test_load_structured_json_documents_supports_thinking_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "thinking.jsonl"
            rec = {
                "messages": [
                    {"role": "user", "content": "Calculate 15% of 80."},
                    {
                        "role": "assistant",
                        "content": "<think>\n15% of 80 is 0.15 * 80 = 12.\n</think>\n15% of 80 is 12.",
                    },
                ]
            }
            source.write_text(json.dumps(rec) + "\n", encoding="utf-8")

            docs = load_structured_json_documents(source, "thinking")
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].kind, "thinking")
            self.assertIn("<think>", docs[0].text)
            self.assertIn("15% of 80 is 12.", docs[0].text)

    def test_thinking_target_masking_isolates_thought_and_answer(self) -> None:
        text = (
            "User: What is 2 + 2?\n"
            "Assistant: <think>\n"
            "We are asked to calculate the sum of 2 and 2.\n"
            "2 + 2 = 4.\n"
            "</think>\n"
            "The sum of 2 and 2 is 4.\n"
            "<eos>"
        )
        spans = find_completion_spans(text)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        completion_text = text[start:end]

        # Completion target must start with the thought block
        self.assertTrue(completion_text.startswith("<think>"))
        self.assertIn("</think>", completion_text)
        self.assertIn("The sum of 2 and 2 is 4.", completion_text)
        # Prompt must NOT be in completion target
        self.assertNotIn("User: What is 2 + 2?", completion_text)

    def test_build_dataset_with_thinking_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_dir = tmp / "input"
            in_dir.mkdir()
            think_file = in_dir / "thinking.jsonl"
            rec = {
                "messages": [
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "<think>\nGravity attracts masses.\n</think>\nGravity is a force."},
                ]
            }
            think_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            out_dir = tmp / "output"

            config = DatasetConfig(
                input_dir=in_dir,
                output_dir=out_dir,
                vocab_size=128,
                dataset_stage="thinking",
                thinking_dataset_path=think_file,
                thinking_dataset_paths=[think_file],
                max_workers=0,
            )
            result = build_dataset(config)
            self.assertTrue(result.output_dir.exists())
            self.assertTrue((out_dir / "train_tokens.npy").exists())
            self.assertTrue((out_dir / "train_targets.npy").exists())

            # Check masking
            targets = np.load(out_dir / "train_targets.npy")
            self.assertTrue(np.any(targets == IGNORE_INDEX))
            self.assertTrue(np.any(targets != IGNORE_INDEX))

    def test_ui_thinking_fine_tune_controls(self) -> None:
        window = _MockWindow("Thinking fine-tune")
        self.assertEqual(window._training_mode_value(), "fine_tune")
        self.assertEqual(window._training_stage_value(), "thinking")

        window.weight_decay = _ValueHolder(0.01)
        window.scheduler_name = _Combo("cosine")
        window.fine_tune_preview = _ValueHolder(0)
        window.fine_tune_checkpoint = type("MockCP", (), {"text": lambda s: ""})()
        window._set_combo_by_data = lambda combo, data, mapping: None
        window._sync_architecture_from_fine_tune_base = lambda: False
        window._update_training_mode_controls = lambda: None

        window.apply_recommended_fine_tune_settings()
        self.assertEqual(window.lora_targets.currentText(), "Attention + MLP")
        self.assertEqual(window.lora_rank.value(), 16)
        self.assertEqual(window.lora_alpha.value(), 32.0)
        self.assertEqual(window.learning_rate.value(), 0.00003)
        self.assertEqual(window.epochs.value(), 3)


if __name__ == "__main__":
    unittest.main()
