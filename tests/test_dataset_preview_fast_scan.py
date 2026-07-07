from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_trainer.config import DatasetConfig
from llm_trainer.dataset_preview import scan_dataset_preview


class DatasetPreviewFastScanTests(unittest.TestCase):
    def test_strict_duplicate_verification_rechecks_fast_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Equal-size files force a fast-scan collision when sample bytes are zero.
            (input_dir / "a.txt").write_text("aaaaaaaaaabbbbbbbbbb", encoding="utf-8")
            (input_dir / "b.txt").write_text("ccccccccccdddddddddd", encoding="utf-8")

            fast_only_config = DatasetConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                fast_scan_mode=True,
                fast_scan_sample_bytes=0,
                strict_duplicate_verification=False,
            )
            fast_only = scan_dataset_preview(fast_only_config, sample_limit=2)
            self.assertGreaterEqual(fast_only.duplicate_count, 2)

            strict_config = DatasetConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                fast_scan_mode=True,
                fast_scan_sample_bytes=0,
                strict_duplicate_verification=True,
            )
            strict = scan_dataset_preview(strict_config, sample_limit=2)
            self.assertEqual(strict.duplicate_count, 0)


if __name__ == "__main__":
    unittest.main()
