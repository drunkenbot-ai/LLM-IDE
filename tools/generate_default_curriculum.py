from __future__ import annotations

from pathlib import Path

try:
    from .curriculum_subjects import (
        code_blocks, computer_blocks, everyday_blocks, geography_history_blocks,
        language_blocks, math_blocks, reasoning_blocks, science_blocks,
        social_blocks,
    )
    from .curriculum_finetune import (
        conversation_fine_tune_blocks, instruction_fine_tune_blocks,
        programming_deep_blocks,
    )
    from .curriculum_code import (
        CODE_LANGUAGE_SPECS, code_fine_tune_block, code_training_block,
        write_target_bytes,
    )
    from .curriculum_shared import write_blocks
except ImportError:
    from curriculum_subjects import (
        code_blocks, computer_blocks, everyday_blocks, geography_history_blocks,
        language_blocks, math_blocks, reasoning_blocks, science_blocks,
        social_blocks,
    )
    from curriculum_finetune import (
        conversation_fine_tune_blocks, instruction_fine_tune_blocks,
        programming_deep_blocks,
    )
    from curriculum_code import (
        CODE_LANGUAGE_SPECS, code_fine_tune_block, code_training_block,
        write_target_bytes,
    )
    from curriculum_shared import write_blocks

ROOT = Path(__file__).resolve().parents[1] / "engine" / "default_data"

def main() -> None:
    """Generate the expanded default curriculum."""

    generators = {
        "language/language_curriculum_generated.txt": language_blocks,
        "mathematics/math_curriculum_generated.txt": math_blocks,
        "science/science_curriculum_generated.txt": science_blocks,
        "geography/geography_history_curriculum_generated.txt": geography_history_blocks,
        "reasoning/reasoning_curriculum_generated.txt": reasoning_blocks,
        "social_emotional/social_emotional_curriculum_generated.txt": social_blocks,
        "everyday/everyday_curriculum_generated.txt": everyday_blocks,
        "computer_science/computer_science_curriculum_generated.txt": computer_blocks,
        "code_training/code_explanation_curriculum_generated.txt": code_blocks,
    }
    # 48000 is a request ceiling, not a promise: each generator now caps
    # itself at its true combinatorial period (see mixed_radix_pick), so
    # categories built from smaller hand-written topic lists (science,
    # geography, social, everyday, computer_science, language) will
    # naturally produce fewer -- but 100% genuinely unique -- blocks instead
    # of silently repeating.
    for relative_path, generator in generators.items():
        write_blocks(ROOT / relative_path, generator(48000))

    programming_topics = [
        "python",
        "javascript_web",
        "java_csharp",
        "c_cpp_systems",
        "rust_go",
        "sql_shell",
        "algorithms",
        "debugging",
        "full_stack",
        "data_structures",
        "software_engineering",
        "mixed_language",
    ]
    for topic in programming_topics:
        write_blocks(
            ROOT / "programming_deep" / f"{topic}_curriculum_generated.txt",
            programming_deep_blocks(14000, topic),
            min_unique_ratio=0.4,
        )

    for language in CODE_LANGUAGE_SPECS:
        write_target_bytes(
            ROOT / "code_training" / f"{language}_code_training_1mb.txt",
            lambda index, language=language: code_training_block(language, index),
            1 * 1024 * 1024,
        )
        write_target_bytes(
            ROOT / "fine_tune_code" / f"{language}_code_finetune_10mb.txt",
            lambda index, language=language: code_fine_tune_block(language, index),
            10 * 1024 * 1024,
        )

    # Fine-tuning corpora were previously requested at 38000-48000 blocks per
    # topic while drawing from only 3 hand-written scenarios each -- 99.97%
    # duplicate content. Rather than pretend that much genuine dialogue
    # diversity exists, the scenario banks were expanded (~12 each) and the
    # requested count was brought down to what the combinatorial space can
    # actually back with real, non-duplicate examples.
    conversation_topics = [
        "daily_help",
        "learning_tutor",
        "coding_mentor",
        "empathy_support",
        "professional_chat",
    ]
    for topic in conversation_topics:
        write_blocks(
            ROOT / "fine_tune_conversation" / f"{topic}_conversation_generated.txt",
            conversation_fine_tune_blocks(6000, topic),
        )

    instruction_topics = [
        "writing_tasks",
        "reasoning_tasks",
        "coding_tasks",
        "classification_tasks",
        "format_following",
    ]
    for topic in instruction_topics:
        write_blocks(
            ROOT / "fine_tune_instruction" / f"{topic}_instruction_generated.txt",
            instruction_fine_tune_blocks(2000, topic),
        )


if __name__ == "__main__":
    main()
