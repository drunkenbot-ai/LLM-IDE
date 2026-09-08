from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure interface.app is imported first to populate shared mixin globals
from interface import app as _interface_app  # noqa: F401
from interface.core.project_state import ProjectStateMixin
from interface.core.project_state_apply import ProjectStateApplyMixin

from engine.config import DatasetConfig, ModelConfig, TrainingConfig
from engine.data import Document
from engine.dataset_corpus import _StreamingCorpusBuilder
from engine.dataset_mixture import (
    MAX_REPETITIVE_UNIT_RATIO,
    MIN_UNIQUE_UNITS_FOR_DIVERSITY,
    _filter_repetitive_documents,
)
from engine.tokenizer import (
    EOS_TOKEN,
    encode_file_to_bin,
    load_token_memmap,
    token_dtype_for_vocab,
    train_tokenizer,
)
from engine.training import TokenDataset, train_model


class DiversityFilterTests(unittest.TestCase):
    """Tests for low-diversity filtering behavior."""

    def test_diversity_filter_accepts_large_code_and_books(self) -> None:
        """Legitimate files with repeated syntax but many unique units must be accepted."""
        # Simulate a 1MB-style code file: common boilerplate repeated 10 times,
        # but 150 completely unique function definitions and logic blocks.
        boilerplate = [
            "// Standard project license header and notice block",
            "import std.collections.unordered_map;",
            "import std.concurrent.atomic_ref;",
            "if (error_code != 0) { return error_code; }",
        ]
        unique_lines = [
            f"void process_record_item_id_{i}(int param_{i}) {{ execute_job({i}); }}"
            for i in range(150)
        ]
        lines = unique_lines + (boilerplate * 10)
        content = "\n".join(lines)
        doc = Document(path=Path("code_sample.cpp"), text=content, kind="code")

        accepted, report = _filter_repetitive_documents([doc])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(report["removed_documents"], 0)

    def test_diversity_filter_rejects_synthetic_padding(self) -> None:
        """Tiny templates repeated over and over to pad corpus size must be excluded."""
        template_sentences = [
            "The quick brown fox jumps over the lazy dog repeatedly.",
            "Synthetic data generation fills space without any real knowledge.",
            "This template unit repeats across the document to pad character length.",
        ]
        text = " ".join(template_sentences * 40)
        doc = Document(path=Path("synthetic_padding.txt"), text=text, kind="prose")

        accepted, report = _filter_repetitive_documents([doc])
        self.assertEqual(len(accepted), 0)
        self.assertEqual(report["removed_documents"], 1)

    def test_streaming_corpus_builder_respects_filter_toggle(self) -> None:
        """When filter_low_diversity is False, repetitive files are not excluded."""
        template = "This template repeats indefinitely without variation for testing."
        text = " ".join([template] * 80)
        doc = Document(path=Path("repetitive.txt"), text=text, kind="prose")

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "corpus.txt"
            builder = _StreamingCorpusBuilder(
                corpus_path=corpus_path,
                code_training_mode=False,
                generate_instruction_samples=False,
                reasoning_sample_mode="scaffold",
                filter_low_diversity=False,
            )
            builder.submit(doc)
            self.assertEqual(builder.stats.low_diversity_removed, 0)
            self.assertEqual(builder.stats.accepted_document_count, 1)
            builder.close()


class EarlyStoppingPatiencePersistenceTests(unittest.TestCase):
    """Tests for early stopping patience persistence in project state."""

    def test_early_stopping_patience_in_default_state(self) -> None:
        window = MagicMock()
        window.device.currentText.return_value = "cpu"
        window.use_amp_default = False
        state = ProjectStateMixin._default_project_state(window)

        self.assertIn("early_stopping", state["training"])
        self.assertTrue(state["training"]["early_stopping"])
        self.assertIn("early_stopping_patience", state["training"])
        self.assertEqual(state["training"]["early_stopping_patience"], 3)

    def test_early_stopping_patience_snapshot_and_apply(self) -> None:
        window = MagicMock()
        window.theme_name = "dark"
        window.persisted_training_process = {}
        window.early_stopping.isChecked.return_value = True
        window.early_stopping_patience.value.return_value = 7

        snap = ProjectStateMixin._project_state_dict(window, "test", Path("."))
        self.assertEqual(snap["training"]["early_stopping_patience"], 7)

        # Apply should restore patience value
        test_state = {
            "training": {
                "early_stopping": True,
                "early_stopping_patience": 5,
            }
        }
        ProjectStateApplyMixin._apply_project_state(window, test_state)
        window.early_stopping_patience.setValue.assert_called_with(5)


class TokenizerCorpusEncodingTests(unittest.TestCase):
    """Tests that tokenizing multi-line text does not corrupt lines with <bos>/<eos>."""

    def test_encode_file_to_bin_no_per_line_special_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corpus_file = temp_path / "corpus.txt"
            # Two documents, each with two lines, separated by <eos>
            corpus_text = (
                f"Line one of document alpha\n"
                f"Line two of document alpha\n"
                f"{EOS_TOKEN}\n"
                f"Line one of document beta\n"
                f"Line two of document beta\n"
                f"{EOS_TOKEN}\n"
            )
            corpus_file.write_text(corpus_text, encoding="utf-8")
            tok_path = temp_path / "tokenizer.json"
            tok = train_tokenizer(corpus_file, tok_path, vocab_size=60)
            eos_id = tok.token_to_id(EOS_TOKEN)
            bos_id = tok.token_to_id("<bos>")

            bin_file = temp_path / "tokens.bin"
            dtype = token_dtype_for_vocab(60)
            encode_file_to_bin(tok, corpus_file, bin_file, dtype=dtype)

            tokens = list(load_token_memmap(bin_file, dtype=dtype))

            # eos_id should only appear at the document boundaries (2 times total)
            eos_indices = [i for i, t in enumerate(tokens) if t == eos_id]
            self.assertEqual(len(eos_indices), 2)

            # bos_id should NOT appear before every line
            bos_indices = [i for i, t in enumerate(tokens) if t == bos_id]
            self.assertEqual(len(bos_indices), 0)


class ValidationLoaderStrideTests(unittest.TestCase):
    """Tests for non-overlapping validation loader stride."""

    def test_validation_stride_equals_context_length(self) -> None:
        context_length = 64
        # 257 tokens -> 4 non-overlapping windows of length 64 with stride 64
        tokens = list(range(257))
        val_dataset = TokenDataset(tokens, context_length=context_length, stride=context_length)
        self.assertEqual(len(val_dataset), 4)

        # First window covers tokens 0..63 (input) -> 1..64 (target)
        x0, y0 = val_dataset[0]
        self.assertEqual(x0.tolist(), list(range(0, 64)))
        self.assertEqual(y0.tolist(), list(range(1, 65)))

        # Second window covers tokens 64..127 (input) -> 65..128 (target)
        x1, y1 = val_dataset[1]
        self.assertEqual(x1.tolist(), list(range(64, 128)))
        self.assertEqual(y1.tolist(), list(range(65, 129)))


class TrainingDiagnosticsAndTelemetryTests(unittest.TestCase):
    """Tests for preflight warnings and telemetry emission enhancements."""

    def test_head_dim_not_divisible_by_eight_detected(self) -> None:
        model_config = ModelConfig(
            vocab_size=32,
            context_length=16,
            embedding_size=560,
            head_count=8,
            layer_count=2,
        )
        head_dim = model_config.embedding_size // model_config.head_count
        self.assertEqual(head_dim, 70)
        self.assertNotEqual(head_dim % 8, 0)

    def test_milestone_step_emits_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_config = ModelConfig(
                vocab_size=16,
                context_length=8,
                embedding_size=16,
                head_count=2,
                layer_count=1,
                dropout=0.0,
            )
            training_config = TrainingConfig(
                output_dir=Path(tmp_dir),
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                sample_stride=8,
                warmup_steps=0,
                eval_interval=0,
                save_interval=0,
                use_amp=False,
                precision="fp32",
                device="cpu",
                resume=False,
                early_stopping=False,
            )
            events: list[dict] = []
            train_model(
                model_config,
                training_config,
                [index % 16 for index in range(32)],
                [],
                pad_token_id=-1,
                progress=events.append,
            )
            step_events = [e for e in events if e.get("event_type") == "step"]
            self.assertGreaterEqual(len(step_events), 1)
            self.assertIn("Step 1", step_events[0]["message"])


if __name__ == "__main__":
    unittest.main()

