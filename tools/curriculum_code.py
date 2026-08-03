from __future__ import annotations

from pathlib import Path
try:
    from .curriculum_shared import *
except ImportError:
    from curriculum_shared import *

def code_training_block(language: str, index: int) -> str:
    """Create one base-training code explanation block.

    Args:
        language: Key from CODE_LANGUAGE_SPECS.
        index: Unique deterministic block number.

    Returns:
        Plain text code teaching block.
    """

    spec = CODE_LANGUAGE_SPECS[language]
    task_i, pattern_i, type_i, container_i, error_i, name_i = mixed_radix_pick(
        index,
        len(CODE_TASKS),
        len(CODE_PATTERNS),
        len(spec["types"]),
        len(spec["containers"]),
        len(spec["errors"]),
        997,
    )
    label = spec["label"]
    comment = spec["comment"]
    task = CODE_TASKS[task_i]
    pattern = CODE_PATTERNS[pattern_i]
    type_name = spec["types"][type_i]
    container = spec["containers"][container_i]
    error = spec["errors"][error_i]
    unique = f"{language}_{name_i}_{index}"
    if language == "python":
        snippet = (
            f"def process_{unique}(items: list[int]) -> int:\n"
            f"    total = 0\n"
            f"    for value in items:\n"
            f"        if value >= 0:\n"
            f"            total += value\n"
            f"    return total\n\n"
            f"assert process_{unique}([1, -2, 3]) == 4"
        )
    elif language in {"javascript", "typescript"}:
        annotation = ": number[]" if language == "typescript" else ""
        return_type = ": number" if language == "typescript" else ""
        snippet = (
            f"function process_{unique}(items{annotation}){return_type} {{\n"
            f"  let total = 0;\n"
            f"  for (const value of items) {{\n"
            f"    if (value >= 0) total += value;\n"
            f"  }}\n"
            f"  return total;\n"
            f"}}\n\n"
            f"console.assert(process_{unique}([1, -2, 3]) === 4);"
        )
    elif language == "java":
        snippet = (
            f"static int process{unique.title().replace('_', '')}(java.util.List<Integer> items) {{\n"
            f"    int total = 0;\n"
            f"    for (int value : items) {{\n"
            f"        if (value >= 0) total += value;\n"
            f"    }}\n"
            f"    return total;\n"
            f"}}"
        )
    elif language == "csharp":
        snippet = (
            f"static int Process{unique.title().replace('_', '')}(IEnumerable<int> items) {{\n"
            f"    var total = 0;\n"
            f"    foreach (var value in items) {{\n"
            f"        if (value >= 0) total += value;\n"
            f"    }}\n"
            f"    return total;\n"
            f"}}"
        )
    elif language == "cpp":
        snippet = (
            f"int process_{unique}(const std::vector<int>& items) {{\n"
            f"    int total = 0;\n"
            f"    for (int value : items) {{\n"
            f"        if (value >= 0) total += value;\n"
            f"    }}\n"
            f"    return total;\n"
            f"}}"
        )
    elif language == "rust":
        snippet = (
            f"fn process_{unique}(items: &[i32]) -> i32 {{\n"
            f"    let mut total = 0;\n"
            f"    for value in items {{\n"
            f"        if *value >= 0 {{ total += *value; }}\n"
            f"    }}\n"
            f"    total\n"
            f"}}"
        )
    elif language == "go":
        snippet = (
            f"func process{unique.title().replace('_', '')}(items []int) int {{\n"
            f"    total := 0\n"
            f"    for _, value := range items {{\n"
            f"        if value >= 0 {{ total += value }}\n"
            f"    }}\n"
            f"    return total\n"
            f"}}"
        )
    elif language == "sql":
        snippet = (
            f"SELECT user_id, SUM(amount) AS total_{name_i}\n"
            f"FROM payments\n"
            f"WHERE amount >= 0\n"
            f"GROUP BY user_id\n"
            f"ORDER BY total_{name_i} DESC;"
        )
    else:
        snippet = (
            f"process_{unique}() {{\n"
            f"  local total=0\n"
            f"  for value in \"$@\"; do\n"
            f"    if [ \"$value\" -ge 0 ]; then total=$((total + value)); fi\n"
            f"  done\n"
            f"  printf '%s\\n' \"$total\"\n"
            f"}}"
        )
    return (
        f"{label} example {index}.\n"
        f"Goal: teach how to {task} with a {pattern}.\n"
        f"The example uses a {container} and a {type_name} value.\n"
        f"```{spec['ext']}\n{snippet}\n```\n"
        f"{comment} Read the code from top to bottom.\n"
        f"The function receives data, skips invalid values, and returns one clear result.\n"
        f"A common mistake in this topic is {error}.\n"
        f"Check the empty input case before trusting the code.\n"
        f"Keep names descriptive, keep steps small, and test one behavior at a time.\n"
    )


def code_fine_tune_block(language: str, index: int) -> str:
    """Create one code fine-tuning instruction block.

    Args:
        language: Key from CODE_LANGUAGE_SPECS.
        index: Unique deterministic block number.

    Returns:
        Instruction-style code fine-tuning block.
    """

    spec = CODE_LANGUAGE_SPECS[language]
    task_i, pattern_i, type_i, container_i, error_i, variant = mixed_radix_pick(
        index,
        len(CODE_TASKS),
        len(CODE_PATTERNS),
        len(spec["types"]),
        len(spec["containers"]),
        len(spec["errors"]),
        2003,
    )
    label = spec["label"]
    task = CODE_TASKS[task_i]
    pattern = CODE_PATTERNS[pattern_i]
    type_name = spec["types"][type_i]
    container = spec["containers"][container_i]
    error = spec["errors"][error_i]
    unique = f"{language}_{variant}_{index}"
    if language == "sql":
        response_code = (
            f"SELECT category, COUNT(*) AS count_{variant}\n"
            f"FROM events\n"
            f"WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'\n"
            f"GROUP BY category\n"
            f"ORDER BY count_{variant} DESC;"
        )
        bug_code = "SELECT category, COUNT(*) FROM events WHERE created_at >= CURRENT_DATE - INTERVAL '7 days';"
    elif language == "bash":
        response_code = (
            f"count_{unique}() {{\n"
            f"  local path=\"$1\"\n"
            f"  if [ ! -f \"$path\" ]; then return 1; fi\n"
            f"  grep -c \"ERROR\" \"$path\"\n"
            f"}}"
        )
        bug_code = "grep -c ERROR $path"
    elif language == "python":
        response_code = (
            f"def solve_{unique}(items: list[int], limit: int) -> list[int]:\n"
            f"    result: list[int] = []\n"
            f"    for value in items:\n"
            f"        if 0 <= value <= limit:\n"
            f"            result.append(value)\n"
            f"    return result\n\n"
            f"assert solve_{unique}([1, -1, 5], 3) == [1]"
        )
        bug_code = f"def solve_{unique}(items, limit):\n    return [x for x in items if x <= limit]"
    elif language in {"javascript", "typescript"}:
        annotation = ": number[]" if language == "typescript" else ""
        limit_annotation = ": number" if language == "typescript" else ""
        return_annotation = ": number[]" if language == "typescript" else ""
        response_code = (
            f"function solve_{unique}(items{annotation}, limit{limit_annotation}){return_annotation} {{\n"
            f"  const result = [];\n"
            f"  for (const value of items) {{\n"
            f"    if (value >= 0 && value <= limit) result.push(value);\n"
            f"  }}\n"
            f"  return result;\n"
            f"}}\n\n"
            f"console.assert(JSON.stringify(solve_{unique}([1, -1, 5], 3)) === JSON.stringify([1]));"
        )
        bug_code = f"function solve_{unique}(items, limit) {{ return items.filter(x => x <= limit); }}"
    elif language == "java":
        method = f"solve{unique.title().replace('_', '')}"
        response_code = (
            f"static java.util.List<Integer> {method}(java.util.List<Integer> items, int limit) {{\n"
            f"    java.util.List<Integer> result = new java.util.ArrayList<>();\n"
            f"    for (int value : items) {{\n"
            f"        if (value >= 0 && value <= limit) result.add(value);\n"
            f"    }}\n"
            f"    return result;\n"
            f"}}"
        )
        bug_code = f"static java.util.List<Integer> {method}(java.util.List<Integer> items, int limit) {{ return null; }}"
    elif language == "csharp":
        method = f"Solve{unique.title().replace('_', '')}"
        response_code = (
            f"static List<int> {method}(IEnumerable<int> items, int limit) {{\n"
            f"    var result = new List<int>();\n"
            f"    foreach (var value in items) {{\n"
            f"        if (value >= 0 && value <= limit) result.Add(value);\n"
            f"    }}\n"
            f"    return result;\n"
            f"}}"
        )
        bug_code = f"static List<int> {method}(IEnumerable<int> items, int limit) => items.Where(x => x <= limit).ToList();"
    elif language == "cpp":
        response_code = (
            f"std::vector<int> solve_{unique}(const std::vector<int>& items, int limit) {{\n"
            f"    std::vector<int> result;\n"
            f"    for (int value : items) {{\n"
            f"        if (value >= 0 && value <= limit) result.push_back(value);\n"
            f"    }}\n"
            f"    return result;\n"
            f"}}"
        )
        bug_code = f"std::vector<int> solve_{unique}(std::vector<int>& items, int limit) {{ return items; }}"
    elif language == "rust":
        response_code = (
            f"fn solve_{unique}(items: &[i32], limit: i32) -> Vec<i32> {{\n"
            f"    items.iter()\n"
            f"        .copied()\n"
            f"        .filter(|value| *value >= 0 && *value <= limit)\n"
            f"        .collect()\n"
            f"}}"
        )
        bug_code = f"fn solve_{unique}(items: Vec<i32>, limit: i32) -> Vec<i32> {{ items }}"
    elif language == "go":
        method = f"solve{unique.title().replace('_', '')}"
        response_code = (
            f"func {method}(items []int, limit int) []int {{\n"
            f"    result := make([]int, 0, len(items))\n"
            f"    for _, value := range items {{\n"
            f"        if value >= 0 && value <= limit {{ result = append(result, value) }}\n"
            f"    }}\n"
            f"    return result\n"
            f"}}"
        )
        bug_code = f"func {method}(items []int, limit int) []int {{ return items }}"
    return (
        f"Instruction: Write {label} code to {task}.\n"
        f"User context: Use a {pattern}. The input involves a {container}. The important type is {type_name}.\n"
        f"Response:\n"
        f"```{spec['ext']}\n{response_code}\n```\n"
        f"Explanation: The solution separates input handling from the core operation.\n"
        f"It names the result clearly and keeps each step small.\n"
        f"Edge case: empty input should return a safe default or a clear error.\n"
        f"Debugging example:\n"
        f"```{spec['ext']}\n{bug_code}\n```\n"
        f"The likely issue is {error}.\n"
        f"Fix: validate inputs, check boundaries, and test the smallest failing case first.\n"
        f"Final answer: use the shown pattern, then add tests for normal, empty, and invalid inputs.\n"
    )


def write_target_bytes(path: Path, block_factory, target_bytes: int) -> None:
    """Write generated corpus blocks until a file reaches the target size.

    Args:
        path: Output file path.
        block_factory: Callable accepting a block index and returning text.
        target_bytes: Minimum UTF-8 byte size to write.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        index = 0
        while handle.tell() < target_bytes:
            handle.write(block_factory(index))
            handle.write("\n\n")
            index += 1



