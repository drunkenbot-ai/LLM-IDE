from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List


def remove_sections(text):

    for section in REMOVE_SECTIONS:

        pattern = (
            rf"\n{section}\n.*"
        )

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

    return text


def clean_text(text):
    import re

    # ---------------------------------------------------------
    # Remove CSS
    # ---------------------------------------------------------
    text = re.sub(
        r"\.mw-parser-output.*?(?=The |\# |\n[A-Z])",
        "",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"@media.*?(?=The |\# |\n[A-Z])",
        "",
        text,
        flags=re.DOTALL,
    )

    # ---------------------------------------------------------
    # Remove references like [1], [23], [a]
    # ---------------------------------------------------------
    text = re.sub(r"\[[^\]]+\]", "", text)

    # ---------------------------------------------------------
    # Remove edit markers
    # ---------------------------------------------------------
    text = text.replace("[edit]", "")

    # ---------------------------------------------------------
    # Collapse whitespace first
    # ---------------------------------------------------------
    text = re.sub(r"\s+", " ", text).strip()

    # ---------------------------------------------------------
    # Remove everything before the first real paragraph.
    # Most Wikipedia pages begin with
    #
    # "The ..."
    # "A ..."
    # "An ..."
    #
    # This removes infoboxes/navigation.
    # ---------------------------------------------------------
    m = re.search(r"\b(The|A|An)\b.+", text)

    if m:
        text = text[m.start():]

    # ---------------------------------------------------------
    # Sentence splitting
    # ---------------------------------------------------------
    text = re.sub(
        r"([.!?])\s+",
        r"\1\n",
        text
    )

    # ---------------------------------------------------------
    # Rebuild paragraphs
    # ---------------------------------------------------------
    paragraph_starters = (
        "The ",
        "In ",
        "On ",
        "At ",
        "After ",
        "Before ",
        "During ",
        "By ",
        "Following ",
        "Meanwhile ",
        "However ",
        "Although ",
        "Later ",
        "Since ",
        "From ",
        "As ",
        "When ",
        "While ",
    )

    paragraphs = []
    current = ""

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if current == "":
            current = line
            continue

        if line.startswith(paragraph_starters):
            paragraphs.append(current.strip())
            current = line
        else:
            current += " " + line

    if current:
        paragraphs.append(current.strip())

    # ---------------------------------------------------------
    # Remove obvious junk paragraphs
    # ---------------------------------------------------------
    cleaned = []

    junk_words = (
        "Belligerents",
        "Campaign",
        "Atlantic Theater",
        "West Indies",
        "Result",
        "Date",
        "Location",
        "Combatants",
        "Casualties",
        "Commander",
        "References",
        "External links",
        "Bibliography",
        "Further reading",
        "See also",
    )

    for p in paragraphs:

        if len(p) < 40:
            continue

        if any(word in p for word in junk_words):
            continue

        cleaned.append(p)

    return "\n\n".join(cleaned)


def chunk_text(text, words_per_chunk):

    words = text.split()

    chunks = []

    for i in range(0, len(words), words_per_chunk):

        chunks.append(
            " ".join(words[i:i + words_per_chunk])
        )

    return chunks


def process_file(file_path, output_dir):

    out = Path(output_dir) / file_path.name

    # Skip if already cleaned
    if out.exists():
        print(f"Skipping (already cleaned): {file_path.name}")
        return

    text = file_path.read_text(
        encoding="utf8",
        errors="ignore",
    )

    cleaned = clean_text(text)

    out.write_text(
        cleaned,
        encoding="utf8",
    )

    print(f"Cleaned: {file_path.name}")


def cleanup(INPUT_DIR, OUTPUT_DIR):

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    files = list(input_dir.glob("*.txt"))

    print(f"Found {len(files)} files")

    cleaned_count = 0
    skipped_count = 0

    for i, file in enumerate(files, 1):

        print(f"[{i}/{len(files)}] {file.name}")

        out = output_dir / file.name

        if out.exists():
            print("   -> Already cleaned, skipping.")
            skipped_count += 1
            continue

        process_file(file, output_dir)
        cleaned_count += 1

    print()
    print(f"Cleanup Done. Cleaned: {cleaned_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
