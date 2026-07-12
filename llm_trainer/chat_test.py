
TOKENIZER_PATH = r"F:\Micro_LLM_Projects\orange\models\tokenizer.json"
MODEL_PATH = r"F:\Micro_LLM_Projects\orange\models\final_model.pt"



from pathlib import Path

from llm_trainer.microgpt_chat import load_microgpt_chat_session

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Point this to your MODEL FOLDER, not final_model.pt
# Example:
# E:\AI_Projects\Models\DrunkenBot


# "auto" = CUDA if available, otherwise CPU
DEVICE = "auto"

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading model...")

session = load_microgpt_chat_session(
    MODEL_PATH,
    device=DEVICE
)

print("Model loaded successfully!")
print(session.runtime_summary)
print()

print("Type 'exit' to quit.\n")

# ---------------------------------------------------------
# Chat Loop
# ---------------------------------------------------------

while True:

    user = input("You: ").strip()

    if not user:
        continue

    if user.lower() in ("exit", "quit"):
        break

    result = session.generate_stream(
        prompt=user,
        system_prompt="",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        reasoning_effort="Balanced",
        thinking_enabled=True,
    )

    print("\nDrunkenBot:")
    print(result["reply"])
    print()

    print(
        f"[{result['token_count']} tokens | "
        f"{result['tokens_per_second']:.2f} tok/s | "
        f"{result['elapsed_seconds']:.2f} sec]"
    )
    print("-" * 60)