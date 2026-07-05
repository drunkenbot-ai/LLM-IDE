from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "llm_trainer" / "default_data" / "generated_curriculum"


def write_blocks(path: Path, blocks: list[str]) -> None:
    """Write plain-text corpus blocks.

    Args:
        path: Output text file.
        blocks: Corpus blocks.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")
    temp_path.replace(path)


def language_blocks(count: int) -> list[str]:
    """Create language teaching blocks."""

    subjects = ["Mina", "Ravi", "Lena", "Omar", "Sara", "Tara"]
    verbs = ["reads", "writes", "asks", "answers", "walks", "listens"]
    objects = ["a book", "a note", "a question", "a sentence", "a story", "a message"]
    qualities = ["clear", "short", "kind", "useful", "simple", "careful"]
    blocks = []
    for index in range(count):
        subject = subjects[index % len(subjects)]
        verb = verbs[index % len(verbs)]
        obj = objects[index % len(objects)]
        quality = qualities[index % len(qualities)]
        blocks.append(
            f"{subject} {verb} {obj}.\n"
            f"The sentence has a subject.\n"
            f"The subject tells who acts.\n"
            f"The verb tells what happens.\n"
            f"The object receives the action.\n"
            f"A {quality} sentence is easy to read.\n"
            f"What does {subject} do?\n"
            f"{subject} {verb} {obj}.\n"
            f"Rewrite the idea with fewer words.\n"
            f"{subject} {verb}."
        )
    return blocks


def math_blocks(count: int) -> list[str]:
    """Create math teaching blocks."""

    blocks = []
    for index in range(1, count + 1):
        a = index % 18 + 2
        b = (index * 3) % 12 + 1
        total = a + b
        product = a * b
        blocks.append(
            f"A box has {a} pencils.\n"
            f"Another box has {b} pencils.\n"
            f"{a} plus {b} equals {total}.\n"
            f"Together the boxes have {total} pencils.\n"
            f"If there are {a} groups of {b}, multiply.\n"
            f"{a} times {b} equals {product}.\n"
            f"Addition joins amounts.\n"
            f"Multiplication joins equal groups.\n"
            f"Check the answer by counting carefully."
        )
    return blocks


def science_blocks(count: int) -> list[str]:
    """Create science teaching blocks."""

    topics = [
        ("plant", "roots take water from soil", "leaves use sunlight"),
        ("heart", "the heart pumps blood", "blood carries oxygen"),
        ("battery", "a battery stores energy", "a wire can carry electricity"),
        ("cloud", "warm air can hold water vapor", "cool air can form clouds"),
        ("magnet", "a magnet pulls some metals", "iron is attracted to magnets"),
        ("moon", "the Moon moves around Earth", "moonlight is reflected sunlight"),
    ]
    blocks = []
    for index in range(count):
        name, fact_one, fact_two = topics[index % len(topics)]
        blocks.append(
            f"A student studies a {name}.\n"
            f"The student observes carefully.\n"
            f"{fact_one.capitalize()}.\n"
            f"{fact_two.capitalize()}.\n"
            f"An observation tells what we notice.\n"
            f"A question asks why it happens.\n"
            f"A test can compare two cases.\n"
            f"Careful notes make science easier to repeat."
        )
    return blocks


def geography_history_blocks(count: int) -> list[str]:
    """Create geography and history teaching blocks."""

    places = [
        ("India", "New Delhi", "Asia", "the Himalayas"),
        ("Egypt", "Cairo", "Africa", "the Nile River"),
        ("Japan", "Tokyo", "Asia", "many islands"),
        ("France", "Paris", "Europe", "the Seine River"),
        ("Brazil", "Brasilia", "South America", "the Amazon region"),
    ]
    inventions = ["the wheel", "writing", "the compass", "the printing press", "the steam engine"]
    blocks = []
    for index in range(count):
        country, capital, continent, feature = places[index % len(places)]
        invention = inventions[index % len(inventions)]
        blocks.append(
            f"{country} is in {continent}.\n"
            f"The capital city is {capital}.\n"
            f"A map can show where {country} is.\n"
            f"One known feature is {feature}.\n"
            f"People in each place have culture.\n"
            f"Culture includes food, language, music, and customs.\n"
            f"History studies change over time.\n"
            f"An important invention was {invention}.\n"
            f"Inventions can change how people live."
        )
    return blocks


def reasoning_blocks(count: int) -> list[str]:
    """Create reasoning teaching blocks."""

    blocks = []
    for index in range(count):
        apples = index % 7 + 2
        extra = index % 5 + 1
        total = apples + extra
        blocks.append(
            f"Tom has {apples} apples.\n"
            f"He gets {extra} more apples.\n"
            f"To find the total, add.\n"
            f"{apples} plus {extra} equals {total}.\n"
            f"Tom has {total} apples.\n"
            f"If the number goes up, addition may help.\n"
            f"If the number goes down, subtraction may help.\n"
            f"Choose the operation from the story."
        )
    return blocks


def social_blocks(count: int) -> list[str]:
    """Create emotion and social reasoning blocks."""

    feelings = [
        ("sad", "her toy broke", "a friend helps her fix it"),
        ("proud", "he finished a hard task", "his practice helped"),
        ("worried", "the room is dark", "she turns on a light"),
        ("angry", "someone took his pencil", "he asks for it back calmly"),
        ("happy", "the class works together", "teamwork feels good"),
    ]
    blocks = []
    for index in range(count):
        feeling, cause, response = feelings[index % len(feelings)]
        blocks.append(
            f"Mina feels {feeling} because {cause}.\n"
            f"A feeling often has a cause.\n"
            f"The feeling can change.\n"
            f"Then {response}.\n"
            f"Kind words can help people feel safe.\n"
            f"Listening shows respect.\n"
            f"A good answer notices facts and feelings."
        )
    return blocks


def everyday_blocks(count: int) -> list[str]:
    """Create everyday knowledge and ethics blocks."""

    tasks = [
        ("cook rice", "wash the rice", "turn off the stove"),
        ("cross a road", "look both ways", "wait for vehicles to stop"),
        ("save money", "count income", "spend less than you earn"),
        ("clean a room", "put sharp things away", "wipe wet floors"),
        ("visit a doctor", "explain symptoms", "follow safe advice"),
    ]
    blocks = []
    for index in range(count):
        task, first, safe = tasks[index % len(tasks)]
        blocks.append(
            f"A person needs to {task}.\n"
            f"First, they {first}.\n"
            f"Good planning avoids mistakes.\n"
            f"They should {safe}.\n"
            f"Safety protects people.\n"
            f"Responsibility means doing what should be done.\n"
            f"Honesty and care help a community."
        )
    return blocks


def computer_blocks(count: int) -> list[str]:
    """Create computer science teaching blocks."""

    ideas = [
        ("variable", "stores a value", "x = 5"),
        ("loop", "repeats steps", "for item in items"),
        ("function", "groups steps", "def add(a, b)"),
        ("list", "keeps items in order", "numbers = [1, 2, 3]"),
        ("dictionary", "connects keys to values", "scores = {'Mina': 9}"),
        ("algorithm", "is a set of steps", "sort the numbers"),
    ]
    blocks = []
    for index in range(count):
        idea, meaning, example = ideas[index % len(ideas)]
        blocks.append(
            f"A {idea} {meaning}.\n"
            f"Example: {example}.\n"
            f"Code should be clear.\n"
            f"A programmer reads errors carefully.\n"
            f"Debugging means finding the cause.\n"
            f"Testing checks if code works.\n"
            f"Small steps make hard problems easier."
        )
    return blocks


def code_blocks(count: int) -> list[str]:
    """Create code explanation corpus blocks."""

    blocks = []
    for index in range(count):
        value = index % 9 + 1
        other = index % 5 + 2
        blocks.append(
            "Python example.\n"
            f"x = {value}\n"
            f"y = {other}\n"
            "print(x + y)\n"
            f"x stores {value}.\n"
            f"y stores {other}.\n"
            "The plus sign adds numbers.\n"
            f"The program prints {value + other}.\n"
            "This example teaches variables and addition."
        )
    return blocks


def programming_deep_blocks(count: int, topic: str) -> list[str]:
    """Create programming-focused corpus blocks.

    Args:
        count: Number of blocks to generate.
        topic: Programming topic name.

    Returns:
        Generated corpus blocks.
    """

    python_examples = [
        (
            "Python list filtering",
            "numbers = [1, 2, 3, 4, 5]\n"
            "even = []\n"
            "for number in numbers:\n"
            "    if number % 2 == 0:\n"
            "        even.append(number)\n"
            "print(even)",
            "The list stores numbers in order.\nThe loop checks each number.\nThe percent operator gives the remainder.\nA remainder of zero means the number is even.",
        ),
        (
            "Python function",
            "def area(width, height):\n"
            "    return width * height\n\n"
            "result = area(6, 4)\n"
            "print(result)",
            "The function receives width and height.\nThe return statement sends back the answer.\nThe area is twenty four.",
        ),
        (
            "Python dictionary",
            "scores = {'Mina': 8, 'Ravi': 9}\n"
            "scores['Lena'] = 7\n"
            "for name, score in scores.items():\n"
            "    print(name, score)",
            "A dictionary maps keys to values.\nThe key is a name.\nThe value is a score.\nThe items method gives key and value pairs.",
        ),
    ]
    javascript_examples = [
        (
            "JavaScript array map",
            "const prices = [10, 20, 30];\n"
            "const doubled = prices.map(price => price * 2);\n"
            "console.log(doubled);",
            "An array keeps values in order.\nThe map method creates a new array.\nThe arrow function runs once for each value.",
        ),
        (
            "JavaScript async function",
            "async function loadUser(id) {\n"
            "  const response = await fetch(`/users/${id}`);\n"
            "  return response.json();\n"
            "}",
            "The async keyword allows await.\nAwait pauses until the promise settles.\nThis is useful for network requests.",
        ),
    ]
    java_examples = [
        (
            "Java class",
            "class Counter {\n"
            "    private int value = 0;\n"
            "    void increment() {\n"
            "        value++;\n"
            "    }\n"
            "    int getValue() {\n"
            "        return value;\n"
            "    }\n"
            "}",
            "A class groups data and behavior.\nThe field stores the count.\nThe method changes the count.\nPrivate data is hidden from outside code.",
        ),
    ]
    cpp_examples = [
        (
            "C++ vector loop",
            "#include <iostream>\n"
            "#include <vector>\n\n"
            "int main() {\n"
            "    std::vector<int> values{1, 2, 3};\n"
            "    int total = 0;\n"
            "    for (int value : values) {\n"
            "        total += value;\n"
            "    }\n"
            "    std::cout << total << '\\n';\n"
            "}",
            "A vector stores many values.\nThe range loop visits each value.\nThe total variable accumulates the sum.",
        ),
        (
            "C pointer safety",
            "#include <stdio.h>\n\n"
            "int main(void) {\n"
            "    int value = 5;\n"
            "    int *ptr = &value;\n"
            "    printf(\"%d\\n\", *ptr);\n"
            "    return 0;\n"
            "}",
            "A pointer stores an address.\nThe address operator gets the address.\nThe star operator reads the value at the address.",
        ),
    ]
    rust_go_examples = [
        (
            "Rust ownership",
            "fn main() {\n"
            "    let name = String::from(\"Mina\");\n"
            "    print_name(&name);\n"
            "    println!(\"{}\", name);\n"
            "}\n\n"
            "fn print_name(value: &String) {\n"
            "    println!(\"{}\", value);\n"
            "}",
            "The ampersand borrows the string.\nBorrowing lets a function read without taking ownership.\nThe original value can still be used later.",
        ),
        (
            "Go error handling",
            "file, err := os.Open(\"data.txt\")\n"
            "if err != nil {\n"
            "    return err\n"
            "}\n"
            "defer file.Close()",
            "Go returns errors as values.\nThe code checks the error immediately.\nThe defer statement closes the file later.",
        ),
    ]
    sql_shell_examples = [
        (
            "SQL selection",
            "SELECT name, age\n"
            "FROM users\n"
            "WHERE age >= 18\n"
            "ORDER BY name;",
            "The SELECT clause chooses columns.\nThe FROM clause chooses a table.\nThe WHERE clause filters rows.\nThe ORDER BY clause sorts the result.",
        ),
        (
            "Bash pipeline",
            "cat access.log | grep ERROR | sort | uniq -c",
            "A pipeline sends output to the next command.\nGrep filters matching lines.\nSort groups similar lines.\nUniq counts repeated lines.",
        ),
        (
            "PowerShell pipeline",
            "Get-ChildItem -File | Where-Object { $_.Length -gt 1MB } | Select-Object Name, Length",
            "PowerShell passes objects through the pipeline.\nWhere-Object filters objects.\nSelect-Object chooses properties to display.",
        ),
    ]
    web_examples = [
        (
            "HTML form",
            "<form>\n"
            "  <label>Name</label>\n"
            "  <input name=\"name\" />\n"
            "  <button type=\"submit\">Save</button>\n"
            "</form>",
            "A form collects input.\nA label tells the user what to enter.\nA button submits the form.",
        ),
        (
            "CSS button",
            ".button {\n"
            "  background: #222;\n"
            "  color: white;\n"
            "  padding: 8px 12px;\n"
            "}\n"
            ".button:hover {\n"
            "  background: #444;\n"
            "}",
            "CSS changes how elements look.\nThe hover rule runs when the pointer is over the button.",
        ),
    ]
    algorithm_examples = [
        (
            "Binary search",
            "def binary_search(values, target):\n"
            "    low = 0\n"
            "    high = len(values) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if values[mid] == target:\n"
            "            return mid\n"
            "        if values[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1",
            "Binary search works on sorted data.\nEach step removes half of the remaining choices.\nThis makes it faster than checking every item.",
        ),
        (
            "Queue with list",
            "from collections import deque\n\n"
            "queue = deque()\n"
            "queue.append('first')\n"
            "queue.append('second')\n"
            "item = queue.popleft()\n"
            "print(item)",
            "A queue is first in, first out.\nAppend adds to the back.\nPopleft removes from the front.",
        ),
    ]
    debugging_examples = [
        (
            "Read the traceback",
            "Traceback says the error line.\nStart at the last line.\nFind the exception name.\nThen inspect the code near that line.\nA NameError often means a variable name is missing or misspelled.",
            "Debugging starts with evidence.\nDo not guess first.\nRead the error.\nReproduce the bug.\nChange one thing.\nRun the test again.",
        ),
        (
            "Off by one error",
            "for index in range(len(items)):\n"
            "    print(items[index])",
            "Indexes start at zero in many languages.\nThe last index is length minus one.\nAn off by one error reads before the start or after the end.",
        ),
    ]
    sets = {
        "python": python_examples,
        "javascript_web": javascript_examples + web_examples,
        "java_csharp": java_examples,
        "c_cpp_systems": cpp_examples,
        "rust_go": rust_go_examples,
        "sql_shell": sql_shell_examples,
        "algorithms": algorithm_examples,
        "debugging": debugging_examples,
        "full_stack": web_examples + sql_shell_examples + javascript_examples,
        "data_structures": algorithm_examples + python_examples,
        "software_engineering": debugging_examples + java_examples + rust_go_examples,
        "mixed_language": python_examples + javascript_examples + cpp_examples + rust_go_examples + sql_shell_examples,
    }
    examples = sets[topic]
    blocks = []
    for index in range(count):
        title, code, explanation = examples[index % len(examples)]
        scenario = index % 11
        blocks.append(
            f"{title}.\n"
            f"Example number {index + 1}.\n"
            f"{code}\n"
            f"{explanation}\n"
            f"The programmer should name variables clearly.\n"
            f"The program should handle expected input.\n"
            f"The program should fail clearly when input is wrong.\n"
            f"A small test should check the normal case.\n"
            f"A second test should check an edge case.\n"
            f"If scenario {scenario} changes, update the test first.\n"
            f"Good code is readable, correct, and easy to change."
        )
    return blocks


def conversation_fine_tune_blocks(count: int, topic: str) -> list[str]:
    """Create conversation fine-tuning corpus blocks.

    Args:
        count: Number of blocks to generate.
        topic: Conversation scenario group.

    Returns:
        Conversation training blocks.
    """

    scenarios = {
        "daily_help": [
            ("planning a day", "I have school and chores today.", "Make a short list. Do the urgent task first."),
            ("finding a lost key", "I lost my key.", "Check the last place you used it. Then check pockets and bags."),
            ("choosing food", "I want a simple dinner.", "Rice, vegetables, and eggs can make a quick meal."),
        ],
        "learning_tutor": [
            ("asking about fractions", "What is one half?", "One half means one part out of two equal parts."),
            ("grammar help", "When do I use went?", "Use went for the past. I went to school yesterday."),
            ("science question", "Why does ice melt?", "Ice melts when it gains heat. It becomes liquid water."),
        ],
        "coding_mentor": [
            ("debugging Python", "My loop prints too many lines.", "Check the range. The stop value may be too large."),
            ("understanding variables", "What is a variable?", "A variable is a name that stores a value."),
            ("reading an error", "What does NameError mean?", "It usually means Python cannot find that variable name."),
        ],
        "empathy_support": [
            ("nervous before exam", "I feel nervous about my exam.", "That is normal. Study small parts and take breaks."),
            ("friend conflict", "My friend ignored me.", "Ask calmly what happened. Listen before deciding."),
            ("mistake at work", "I made a mistake.", "Own it, fix what you can, and learn the cause."),
        ],
        "professional_chat": [
            ("email rewrite", "Can you make this email polite?", "Yes. Keep it short, clear, and respectful."),
            ("meeting plan", "How should I run a meeting?", "Set a goal, list topics, and end with action items."),
            ("status update", "I need to report progress.", "Say what is done, what is blocked, and what comes next."),
        ],
    }
    items = scenarios[topic]
    blocks = []
    for index in range(count):
        title, user_text, assistant_text = items[index % len(items)]
        turn = index % 9
        blocks.append(
            f"Conversation: {title}.\n"
            f"User: {user_text}\n"
            f"Assistant: {assistant_text}\n"
            f"User: Can you explain simply?\n"
            f"Assistant: Yes. I will use short steps.\n"
            f"Assistant: First, understand the problem.\n"
            f"Assistant: Second, choose a small action.\n"
            f"Assistant: Third, check the result.\n"
            f"User: What should I avoid?\n"
            f"Assistant: Avoid guessing when facts are missing.\n"
            f"Assistant: Ask a clear question if needed.\n"
            f"User: Give me a final answer.\n"
            f"Assistant: The best next step is to act carefully and review the result.\n"
            f"This dialogue teaches helpful conversation turn {turn}."
        )
    return blocks


def instruction_fine_tune_blocks(count: int, topic: str) -> list[str]:
    """Create instruction fine-tuning corpus blocks.

    Args:
        count: Number of blocks to generate.
        topic: Instruction task group.

    Returns:
        Instruction training blocks.
    """

    tasks = {
        "writing_tasks": [
            ("Rewrite this sentence in simpler English.", "The child rapidly moved across the room.", "The child ran across the room."),
            ("Summarize this passage.", "Mina planted seeds. She watered them. After many days, leaves grew.", "Mina planted and cared for seeds until they grew leaves."),
            ("Make this polite.", "Send the report now.", "Please send the report when you have a moment."),
        ],
        "reasoning_tasks": [
            ("Solve the word problem.", "A box has 6 pens. Ravi adds 4 pens. How many pens are there?", "There are 10 pens."),
            ("Choose the safer action.", "A wire is broken. Should Tara touch it or call an adult?", "Tara should call an adult."),
            ("Find the cause.", "The lamp does not turn on. The bulb is loose.", "The loose bulb may be the cause."),
        ],
        "coding_tasks": [
            ("Write a Python function that adds two numbers.", "Use parameters a and b.", "def add(a, b):\n    return a + b"),
            ("Explain this code.", "print(len([1, 2, 3]))", "It creates a list with three items and prints its length, which is 3."),
            ("Fix the bug.", "for i in range(3):\nprint(i)", "Indent the print line inside the loop."),
        ],
        "classification_tasks": [
            ("Classify the sentence.", "The sky is cloudy today.", "Category: weather observation."),
            ("Classify the request.", "Can you help me debug this error?", "Category: coding help."),
            ("Classify the emotion.", "I am proud because I finished the project.", "Emotion: proud."),
        ],
        "format_following": [
            ("Answer with two bullet points.", "Give two safe cooking tips.", "- Wash your hands.\n- Turn off the stove after cooking."),
            ("Return only the number.", "What is 8 plus 5?", "13"),
            ("Use a short answer.", "Why do plants need light?", "Plants use light to make food."),
        ],
    }
    items = tasks[topic]
    blocks = []
    for index in range(count):
        instruction, input_text, output_text = items[index % len(items)]
        turn = index % 13
        blocks.append(
            f"Instruction: {instruction}\n"
            f"Input: {input_text}\n"
            f"Response: {output_text}\n"
            f"The response follows the instruction.\n"
            f"The response stays focused on the user request.\n"
            f"The response avoids extra unrelated text.\n"
            f"If information is missing, ask one clear question.\n"
            f"If the task is simple, answer directly.\n"
            f"If the task needs steps, use short ordered steps.\n"
            f"This instruction sample teaches format control {turn}."
        )
    return blocks


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
        )

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
            conversation_fine_tune_blocks(38000, topic),
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
            instruction_fine_tune_blocks(48000, topic),
        )


if __name__ == "__main__":
    main()
