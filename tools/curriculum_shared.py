from __future__ import annotations

NAMES = [
    "Mina", "Ravi", "Lena", "Omar", "Sara", "Tara", "Jin", "Ada",
    "Leo", "Nia", "Sam", "Priya",
]

MODIFIERS = ["carefully", "quickly", "calmly", "clearly", "patiently"]

CLOSERS = [
    "What did you notice?",
    "Why do you think that happens?",
    "How could you check this?",
    "What would change if one detail changed?",
    "Explain it back in your own words.",
]

CODE_LANGUAGE_SPECS = {
    "python": {
        "label": "Python",
        "ext": "py",
        "comment": "#",
        "types": ["list[int]", "dict[str, int]", "str", "tuple[int, int]", "set[str]"],
        "containers": ["list", "dictionary", "set", "tuple", "file"],
        "errors": ["IndexError", "KeyError", "TypeError", "ValueError", "NameError"],
    },
    "javascript": {
        "label": "JavaScript",
        "ext": "js",
        "comment": "//",
        "types": ["Array", "Object", "string", "number", "Promise"],
        "containers": ["array", "object", "map", "set", "DOM node"],
        "errors": ["TypeError", "ReferenceError", "RangeError", "SyntaxError", "Promise rejection"],
    },
    "typescript": {
        "label": "TypeScript",
        "ext": "ts",
        "comment": "//",
        "types": ["number[]", "Record<string, number>", "string", "Promise<void>", "ReadonlyArray<string>"],
        "containers": ["typed array", "record", "interface", "union", "generic"],
        "errors": ["type mismatch", "undefined value", "narrowing error", "implicit any", "async error"],
    },
    "java": {
        "label": "Java",
        "ext": "java",
        "comment": "//",
        "types": ["List<Integer>", "Map<String, Integer>", "String", "Optional<String>", "Set<String>"],
        "containers": ["ArrayList", "HashMap", "HashSet", "class", "stream"],
        "errors": ["NullPointerException", "IndexOutOfBoundsException", "IllegalArgumentException", "ClassCastException", "IOException"],
    },
    "csharp": {
        "label": "C#",
        "ext": "cs",
        "comment": "//",
        "types": ["List<int>", "Dictionary<string, int>", "string", "Task", "IEnumerable<string>"],
        "containers": ["List", "Dictionary", "HashSet", "class", "LINQ query"],
        "errors": ["NullReferenceException", "IndexOutOfRangeException", "InvalidOperationException", "ArgumentException", "async deadlock"],
    },
    "cpp": {
        "label": "C++",
        "ext": "cpp",
        "comment": "//",
        "types": ["vector<int>", "unordered_map<string, int>", "string", "unique_ptr<Node>", "optional<int>"],
        "containers": ["vector", "unordered_map", "set", "struct", "iterator"],
        "errors": ["segmentation fault", "dangling pointer", "out_of_range", "memory leak", "undefined behavior"],
    },
    "rust": {
        "label": "Rust",
        "ext": "rs",
        "comment": "//",
        "types": ["Vec<i32>", "HashMap<String, i32>", "String", "Option<i32>", "Result<String, String>"],
        "containers": ["Vec", "HashMap", "slice", "struct", "iterator"],
        "errors": ["borrow checker error", "panic", "lifetime error", "unwrap failure", "type mismatch"],
    },
    "go": {
        "label": "Go",
        "ext": "go",
        "comment": "//",
        "types": ["[]int", "map[string]int", "string", "error", "chan int"],
        "containers": ["slice", "map", "struct", "goroutine", "channel"],
        "errors": ["nil pointer", "index out of range", "data race", "ignored error", "deadlock"],
    },
    "sql": {
        "label": "SQL",
        "ext": "sql",
        "comment": "--",
        "types": ["INTEGER", "TEXT", "TIMESTAMP", "BOOLEAN", "DECIMAL"],
        "containers": ["table", "index", "view", "join", "transaction"],
        "errors": ["missing index", "duplicate key", "bad join", "null value", "slow query"],
    },
    "bash": {
        "label": "Bash",
        "ext": "sh",
        "comment": "#",
        "types": ["string", "array", "exit code", "path", "environment variable"],
        "containers": ["loop", "function", "pipe", "process", "file"],
        "errors": ["missing quote", "bad path", "nonzero exit", "unset variable", "permission denied"],
    },
}

CODE_TASKS = [
    "parse input",
    "validate data",
    "filter a collection",
    "count repeated values",
    "read a file safely",
    "write a small helper",
    "handle an error",
    "sort records",
    "cache a result",
    "format output",
    "test an edge case",
    "split work into functions",
]

CODE_PATTERNS = [
    "loop",
    "function",
    "guard clause",
    "map lookup",
    "unit test",
    "small class",
    "command handler",
    "parser",
    "retry step",
    "cleanup step",
]


def mixed_radix_pick(index: int, *sizes: int) -> list[int]:
    """Decompose an index into independent per-axis picks.

    Unlike applying ``index % len(list)`` to several lists at once (which
    repeats after ``lcm`` of the list lengths -- often a tiny number), this
    treats ``index`` as a mixed-radix counter across every axis. The combined
    period is the *product* of all axis sizes, so a handful of modest lists
    (say four lists of 15-20 items) already yields a combinatorial space of
    tens of thousands of unique combinations before anything repeats.

    Args:
        index: Zero-based block index.
        *sizes: Length of each axis, in the same order picks are needed.

    Returns:
        One pick per axis, each in ``range(0, size)``.
    """

    picks = []
    remaining = index
    for size in sizes:
        size = max(1, size)
        picks.append(remaining % size)
        remaining //= size
    return picks

def combinatorial_period(*sizes: int) -> int:
    """Return the number of unique combinations `mixed_radix_pick` can produce."""

    period = 1
    for size in sizes:
        period *= max(1, size)
    return period

def write_blocks(path: Path, blocks: list[str], min_unique_ratio: float = 0.9) -> None:
    """Write plain-text corpus blocks, guarding against templated duplication.

    A generator that technically returns ``count`` blocks but only cycles
    through a handful of unique strings silently produces a dataset that is
    almost entirely duplicate data -- wasted disk, wasted training compute,
    and a validation split that can't mean anything because train and
    validation end up full of the same repeated content. This raises loudly
    instead of writing a file that *looks* like a real corpus but isn't.

    Args:
        path: Output text file.
        blocks: Corpus blocks.
        min_unique_ratio: Minimum allowed fraction of unique blocks. Raise
            the ratio for categories that should have high diversity; lower
            it only for content that is legitimately formulaic.

    Raises:
        ValueError: If the unique-block ratio falls below ``min_unique_ratio``.
    """

    if blocks:
        unique_ratio = len(set(blocks)) / len(blocks)
        if unique_ratio < min_unique_ratio:
            raise ValueError(
                f"{path}: only {unique_ratio:.1%} of {len(blocks)} blocks are unique "
                f"(minimum required: {min_unique_ratio:.0%}). Widen the source "
                "vocabulary/axes in the generator instead of shipping a "
                "duplicate-heavy file."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")
    temp_path.replace(path)

