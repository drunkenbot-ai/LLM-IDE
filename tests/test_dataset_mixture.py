from __future__ import annotations

import unittest
from pathlib import Path

from llm_trainer.data import Document
from llm_trainer.dataset_mixture import _apply_dataset_mixture, _deduplicate_documents


class DatasetMixtureTests(unittest.TestCase):
    def test_deduplicate_documents_removes_exact_duplicates(self) -> None:
        docs = [
            Document(path=Path("a.txt"), text="Hello world", kind="prose"),
            Document(path=Path("b.txt"), text="Hello   world", kind="prose"),
            Document(path=Path("c.py"), text="print('x')", kind="code", language="python"),
        ]
        deduped, report = _deduplicate_documents(docs)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(report["removed_documents"], 1)

    def test_apply_dataset_mixture_prefers_weighted_code_family(self) -> None:
        docs = [
            Document(path=Path("notes.txt"), text="general prose sample", kind="prose"),
            Document(path=Path("main.py"), text="def run():\n    return 1\n", kind="code", language="python"),
        ]
        selected, report = _apply_dataset_mixture(
            docs,
            weights={"source_code": 100.0, "local_prose": 0.0},
            progress=None,
        )
        self.assertTrue(report["applied"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].kind, "code")


if __name__ == "__main__":
    unittest.main()
