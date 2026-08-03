from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from engine.external_dataset import (
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
            with patch("engine.external_dataset._download", side_effect=copy_archive):
                install_categories(manifest, destination, manifest_url="https://example.test/release/manifest.json")

            self.assertEqual((destination / "base_training" / "example.txt").read_text(), "example")
            self.assertEqual((destination / "version.txt").read_text(), "1.0.0\n")

    def test_install_skips_existing_category_at_same_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "installed"
            category_dir = destination / "base_training"
            category_dir.mkdir(parents=True)
            (category_dir / "existing.txt").write_text("keep", encoding="utf-8")
            (destination / "version.txt").write_text("1.0.0\n", encoding="utf-8")
            manifest = DatasetManifest.from_json({
                "dataset_id": "owner/repo",
                "version": "1.0.0",
                "categories": [{
                    "name": "base_training",
                    "archive": "base.zip",
                    "size_bytes": 1,
                    "file_count": 1,
                    "sha256": "0" * 64,
                }],
            })

            with patch("engine.external_dataset._download") as download:
                install_categories(manifest, destination, categories=["base_training"])

            download.assert_not_called()
            self.assertEqual((category_dir / "existing.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
