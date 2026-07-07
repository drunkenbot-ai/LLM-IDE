from __future__ import annotations

import unittest

import torch

from llm_trainer.config import ModelConfig
from llm_trainer.model import MicroGPT


def _build_model(position_encoding: str) -> MicroGPT:
    config = ModelConfig(
        vocab_size=64,
        context_length=8,
        embedding_size=32,
        head_count=4,
        layer_count=2,
        dropout=0.0,
        position_encoding=position_encoding,
        attention_backend="manual",
    )
    config.validate()
    model = MicroGPT(config)
    model.eval()
    return model


class GenerationKvCacheRegressionTests(unittest.TestCase):
    def test_cached_generation_does_not_fallback_to_full_forward_after_prefill(self) -> None:
        torch.manual_seed(123)
        model = _build_model(position_encoding="learned")
        prompt = torch.randint(0, model.config.vocab_size, (1, model.config.context_length - 1))

        call_log: list[tuple[str, int]] = []
        original_forward_with_cache = model.forward_with_cache
        original_forward = model.forward

        def wrapped_forward_with_cache(idx: torch.Tensor, *args: object, **kwargs: object):
            call_log.append(("fwc", int(idx.size(1))))
            return original_forward_with_cache(idx, *args, **kwargs)

        def wrapped_forward(idx: torch.Tensor, *args: object, **kwargs: object):
            call_log.append(("fwd", int(idx.size(1))))
            return original_forward(idx, *args, **kwargs)

        model.forward_with_cache = wrapped_forward_with_cache
        model.forward = wrapped_forward

        model.generate(
            prompt,
            max_new_tokens=4,
            temperature=1.0,
            top_k=1,
            use_kv_cache=True,
        )

        self.assertFalse(any(kind == "fwd" for kind, _ in call_log), call_log)
        self.assertEqual(call_log[0], ("fwc", model.config.context_length - 1))
        self.assertTrue(all(token_count == 1 for _, token_count in call_log[1:]), call_log)

    def test_cached_and_non_cached_generation_match_past_context_boundary(self) -> None:
        for position_encoding in ("learned", "rope"):
            for prompt_length in (1, 7, 8, 9, 12):
                torch.manual_seed(123)
                model = _build_model(position_encoding=position_encoding)
                prompt = torch.randint(0, model.config.vocab_size, (1, prompt_length))

                cached = model.generate(
                    prompt.clone(),
                    max_new_tokens=6,
                    temperature=1.0,
                    top_k=1,
                    use_kv_cache=True,
                )
                non_cached = model.generate(
                    prompt.clone(),
                    max_new_tokens=6,
                    temperature=1.0,
                    top_k=1,
                    use_kv_cache=False,
                )
                self.assertTrue(
                    torch.equal(cached, non_cached),
                    msg=f"mismatch for {position_encoding=} {prompt_length=}",
                )


if __name__ == "__main__":
    unittest.main()
