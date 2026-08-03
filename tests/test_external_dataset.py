from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from llm_trainer.external_dataset import (
    DatasetManifest,
    install_categories,
    is_newer_version,
    parse_version,
)


class ExternalDatasetTests(unittest.TestCase):
    def test_manifest_rejects_unsafe_archive_name(self) -> None:
        with self.assertRaises(ValueError):
            DatasetManifest.from_json({
                "dataset_id": "owner/repo",
                "version": "1.0.0",
                "categories": [{
                    "name": "base",
                    "archive": "../base.zip",
                    "size_bytes": 1,
                    "file_count": 1,
                    "sha256": "0" * 64,
                }],
            })

    def test_version_comparison(self) -> None:
        self.assertEqual(parse_version("2.10.0"), (2, 10, 0))
        self.assertTrue(is_newer_version("2.0.0", "1.9.9"))
        self.assertFalse(is_newer_version("1.0.1", "1.0.1"))

    def test_install_verifies_and_extracts_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "base-1.0.0.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("base_training/example.txt", "example")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest = DatasetManifest.from_json({
                "dataset_id": "owner/repo",
                "version": "1.0.0",
                "categories": [{
                    "name": "base_training",
                    "archive": archive.name,
                    "size_bytes": archive.stat().st_size,
                    "file_count": 1,
                    "sha256": digest,
                }],
            })

            def copy_archive(_url: str, destination: Path, _progress: object) -> None:
                destination.write_bytes(archive.read_bytes())

            destination = root / "installed"
            with patch("llm_trainer.external_dataset._download", side_effect=copy_archive):
                install_categories(manifest, destination, manifest_url="https://example.test/release/manifest.json")

            self.assertEqual((destination / "base_training" / "example.txt").read_text(), "example")
            self.assertEqual((destination / "version.txt").read_text(), "1.0.0\n")


if __name__ == "__main__":
    unittest.main()
