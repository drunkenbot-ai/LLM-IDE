from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import numpy.lib.format as npy_format
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
MAX_TOKENIZER_TRAINING_CHARS = 25_000_000
MAX_TOKENIZER_LINE_CHARS = 8_192
# Tokens are streamed to disk in fixed-size batches rather than accumulated
# into one giant Python list, so peak RAM during encoding stays roughly
# constant regardless of corpus size.
ENCODE_FLUSH_TOKEN_COUNT = 200_000
# Chunk size (in tokens) used when streaming raw token bytes into the final
# .npy file. Keeps the .bin -> .npy conversion step's RAM use flat too.
NPY_CONVERT_CHUNK_TOKENS = 1_000_000


def train_tokenizer(
    corpus_path: Path,
    output_path: Path,
    vocab_size: int = 8000,
    min_frequency: int = 2,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer.

    Args:
        corpus_path: Text corpus used for tokenizer training.
        output_path: Destination tokenizer JSON path.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum token frequency for BPE merges.
        should_stop: Optional callback returning true when training should stop.

    Returns:
        Trained tokenizer instance.
    """

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(_iter_corpus_lines(corpus_path, should_stop), trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def _iter_corpus_lines(corpus_path: Path, should_stop: Optional[Callable[[], bool]]) -> Iterator[str]:
    """Yield corpus lines and check for cancellation between chunks.

    Args:
        corpus_path: Text corpus path.
        should_stop: Optional cancellation callback.

    Yields:
        Corpus text lines.

    Raises:
        RuntimeError: If cancellation is requested.
    """

    emitted_chars = 0
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if should_stop and should_stop():
                raise RuntimeError("Dataset preparation stopped by user.")
            for start in range(0, len(line), MAX_TOKENIZER_LINE_CHARS):
                chunk = line[start : start + MAX_TOKENIZER_LINE_CHARS]
                if not chunk:
                    continue
                remaining = MAX_TOKENIZER_TRAINING_CHARS - emitted_chars
                if remaining <= 0:
                    return
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                emitted_chars += len(chunk)
                yield chunk


def load_tokenizer(path: Path) -> Tokenizer:
    """Load a tokenizer from disk.

    Args:
        path: Tokenizer JSON path.

    Returns:
        Loaded tokenizer.
    """

    return Tokenizer.from_file(str(path))


def token_id(tokenizer: Tokenizer, token: str) -> int:
    """Return the integer ID for a required special token.

    Args:
        tokenizer: Tokenizer to query.
        token: Token string to find.

    Returns:
        Token ID.

    Raises:
        ValueError: If the tokenizer does not contain the token.
    """

    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"Tokenizer is missing required token: {token}")
    return value


def missing_training_special_tokens(tokenizer: Tokenizer) -> list[str]:
    """Return special tokens missing from a tokenizer.

    Args:
        tokenizer: Tokenizer to inspect.

    Returns:
        Missing special token strings.
    """

    return [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]


def validate_training_tokenizer(tokenizer: Tokenizer) -> None:
    """Validate that a tokenizer can be used by the MicroGPT trainer.

    Args:
        tokenizer: Tokenizer to validate.

    Raises:
        ValueError: If required special tokens are missing.
    """

    missing = missing_training_special_tokens(tokenizer)
    if missing:
        raise ValueError(
            "Tokenizer is not compatible with Micro LLM Creator training. "
            f"Missing required special token(s): {', '.join(missing)}. "
            "Use a tokenizer created by this app, or import a tokenizer containing "
            "<pad>, <unk>, <bos>, and <eos>."
        )


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode text into token IDs.

    Args:
        tokenizer: Tokenizer used for encoding.
        text: Text to encode.

    Returns:
        List of token IDs.
    """

    token_ids: list[int] = []
    for start in range(0, len(text), MAX_TOKENIZER_LINE_CHARS):
        chunk = text[start : start + MAX_TOKENIZER_LINE_CHARS]
        if chunk:
            token_ids.extend(tokenizer.encode(chunk).ids)
    return token_ids


def encode_file(
    tokenizer: Tokenizer,
    corpus_path: Path,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[int]:
    """Encode a corpus file into token IDs without loading it all at once.

    Args:
        tokenizer: Tokenizer used for encoding.
        corpus_path: Text corpus path.
        should_stop: Optional callback returning true when encoding should stop.

    Returns:
        Token IDs for the corpus.

    Raises:
        RuntimeError: If cancellation is requested.
    """

    token_ids: list[int] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if should_stop and should_stop():
                raise RuntimeError("Dataset preparation stopped by user.")
            for start in range(0, len(line), MAX_TOKENIZER_LINE_CHARS):
                chunk = line[start : start + MAX_TOKENIZER_LINE_CHARS]
                if chunk:
                    token_ids.extend(tokenizer.encode(chunk).ids)
    return token_ids


def token_dtype_for_vocab(vocab_size: int) -> np.dtype:
    """Pick the smallest unsigned integer dtype that can hold every token ID.

    Args:
        vocab_size: Tokenizer vocabulary size.

    Returns:
        ``uint16`` for the (overwhelmingly common) case of a vocab under
        65,536, otherwise ``uint32``.
    """

    return np.dtype(np.uint16) if vocab_size <= 65_535 else np.dtype(np.uint32)


def encode_file_to_bin(
    tokenizer: Tokenizer,
    corpus_path: Path,
    output_path: Path,
    dtype: np.dtype,
    should_stop: Optional[Callable[[], bool]] = None,
) -> int:
    """Encode a corpus file straight to a flat, headerless binary token file.

    Unlike ``encode_file``, this never accumulates the whole corpus as a
    Python list of ints in memory. Token IDs are buffered in small batches
    and flushed to disk as raw fixed-width integers (``dtype``), so peak RAM
    stays roughly constant no matter how large the corpus is. This is an
    intermediate format (no shape/dtype header) -- see ``encode_file_to_npy``
    for the version that produces a directly loadable ``.npy`` file.

    Args:
        tokenizer: Tokenizer used for encoding.
        corpus_path: Text corpus path.
        output_path: Destination raw ``.bin`` path for the encoded token stream.
        dtype: Integer dtype to store each token ID as (see
            ``token_dtype_for_vocab``).
        should_stop: Optional callback returning true when encoding should stop.

    Returns:
        Total number of tokens written.

    Raises:
        RuntimeError: If cancellation is requested.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    total_tokens = 0
    with corpus_path.open("r", encoding="utf-8") as source, output_path.open("wb") as sink:
        for line in source:
            if should_stop and should_stop():
                raise RuntimeError("Dataset preparation stopped by user.")
            for start in range(0, len(line), MAX_TOKENIZER_LINE_CHARS):
                chunk = line[start : start + MAX_TOKENIZER_LINE_CHARS]
                if not chunk:
                    continue
                buffer.extend(tokenizer.encode(chunk).ids)
                if len(buffer) >= ENCODE_FLUSH_TOKEN_COUNT:
                    np.asarray(buffer, dtype=dtype).tofile(sink)
                    total_tokens += len(buffer)
                    buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(sink)
            total_tokens += len(buffer)
    return total_tokens


def token_count_in_bin(path: Path, dtype: np.dtype) -> int:
    """Return how many tokens a raw ``.bin`` file holds, without loading it.

    Args:
        path: Raw token ``.bin`` file path.
        dtype: Integer dtype the tokens were stored as.

    Returns:
        Number of tokens in the file.
    """

    itemsize = np.dtype(dtype).itemsize
    return path.stat().st_size // itemsize


def load_token_memmap(path: Path, dtype: np.dtype) -> np.memmap:
    """Open a raw token ``.bin`` file as a read-only memory-mapped array.

    Args:
        path: Raw token ``.bin`` file path.
        dtype: Integer dtype the tokens were stored as.

    Returns:
        Read-only memory-mapped array of token IDs.
    """

    return np.memmap(path, dtype=dtype, mode="r")


def convert_bin_to_npy(bin_path: Path, npy_path: Path, dtype: np.dtype, token_count: int) -> None:
    """Convert a flat raw token ``.bin`` file into a self-describing ``.npy`` file.

    Only a small, fixed-size ``.npy`` header (built from ``dtype`` and
    ``token_count``, which ``encode_file_to_bin`` already gave us for free)
    needs to be known upfront. The token data itself is streamed across in
    fixed-size chunks via plain file reads/writes -- the full token array is
    never materialized in memory during conversion, regardless of how large
    the file is. The result is byte-for-byte a normal ``.npy`` file, loadable
    with ``numpy.load(path, mmap_mode="r")`` like any other.

    Args:
        bin_path: Source raw ``.bin`` file (as written by ``encode_file_to_bin``).
        npy_path: Destination ``.npy`` file path.
        dtype: Integer dtype the tokens were stored as.
        token_count: Number of tokens in ``bin_path``.
    """

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "descr": npy_format.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": (token_count,),
    }
    chunk_bytes = NPY_CONVERT_CHUNK_TOKENS * np.dtype(dtype).itemsize
    with bin_path.open("rb") as source, npy_path.open("wb") as sink:
        npy_format.write_array_header_1_0(sink, header)
        while True:
            block = source.read(chunk_bytes)
            if not block:
                break
            sink.write(block)


def encode_file_to_npy(
    tokenizer: Tokenizer,
    corpus_path: Path,
    output_path: Path,
    dtype: np.dtype,
    should_stop: Optional[Callable[[], bool]] = None,
) -> int:
    """Encode a corpus file straight to a memory-map-friendly ``.npy`` file.

    Combines ``encode_file_to_bin`` (single-pass streaming encode, constant
    RAM) with ``convert_bin_to_npy`` (header + streamed byte copy, also
    constant RAM) so the whole corpus is never held in memory as a Python
    list or a full array at any point, while still producing a standard
    ``.npy`` file that ``numpy.load(path, mmap_mode="r")`` opens directly.

    Args:
        tokenizer: Tokenizer used for encoding.
        corpus_path: Text corpus path.
        output_path: Destination ``.npy`` path for the encoded token stream.
        dtype: Integer dtype to store each token ID as (see
            ``token_dtype_for_vocab``).
        should_stop: Optional callback returning true when encoding should stop.

    Returns:
        Total number of tokens written.

    Raises:
        RuntimeError: If cancellation is requested.
    """

    temp_bin_path = output_path.with_suffix(output_path.suffix + ".raw_tmp")
    try:
        token_count = encode_file_to_bin(tokenizer, corpus_path, temp_bin_path, dtype, should_stop=should_stop)
        convert_bin_to_npy(temp_bin_path, output_path, dtype, token_count)
    finally:
        temp_bin_path.unlink(missing_ok=True)
    return token_count