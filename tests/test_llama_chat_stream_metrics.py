from __future__ import annotations

import unittest
from threading import Lock
from typing import Any, Iterator

from llm_trainer.llama_chat import LlamaChatSession


class _FakeLlama:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def create_chat_completion(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
        if kwargs.get("stream"):
            for content in self._chunks:
                yield {"choices": [{"delta": {"content": content}}]}
            return
        raise AssertionError("Non-stream path is not expected in this test.")

    def tokenize(self, value: bytes) -> list[str]:
        text = value.decode("utf-8")
        return [item for item in text.split() if item]


class LlamaChatStreamMetricsTests(unittest.TestCase):
    def _build_session(self, chunks: list[str]) -> LlamaChatSession:
        session = LlamaChatSession.__new__(LlamaChatSession)
        session._lock = Lock()
        session._messages = []
        session._llm = _FakeLlama(chunks)
        return session

    def test_stream_progress_reports_token_metrics_not_chunk_metrics(self) -> None:
        session = self._build_session(["hello world", " and more words"])
        events: list[dict[str, Any]] = []

        result = session.generate_stream(
            prompt="hi",
            progress=lambda event: events.append(dict(event)),
            thinking_enabled=False,
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["chunk_count"], 1)
        self.assertEqual(events[1]["chunk_count"], 2)
        self.assertEqual(events[0]["token_count"], 2)
        self.assertEqual(events[1]["token_count"], 5)
        self.assertGreater(float(events[1]["tokens_per_second"]), 0.0)
        self.assertEqual(int(result["token_count"]), 5)
        self.assertEqual(result["reply"], "hello world and more words")


if __name__ == "__main__":
    unittest.main()
