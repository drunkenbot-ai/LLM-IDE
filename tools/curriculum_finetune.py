from __future__ import annotations

try:
    from .curriculum_shared import *
except ImportError:
    from curriculum_shared import *

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
            ("packing for a trip", "I don't know what to pack.", "List essentials first: documents, clothes, chargers."),
            ("running late", "I'm running late for an appointment.", "Call ahead if you can, then focus on getting there safely."),
            ("organizing a closet", "My closet is a mess.", "Sort into keep, donate, and discard piles first."),
            ("choosing a gift", "I need a gift for a friend.", "Think about what they enjoy doing in their free time."),
            ("scheduling chores", "I keep forgetting chores.", "Write them on a weekly checklist you can see each morning."),
            ("saving time", "Mornings feel rushed.", "Prepare clothes and bags the night before."),
            ("handling a spill", "I spilled water on my desk.", "Unplug electronics first, then blot the liquid with a cloth."),
            ("choosing a route", "Traffic is bad on my usual route.", "Check a map app for a lighter route before leaving."),
            ("staying hydrated", "I forget to drink water.", "Keep a bottle within sight as a visual reminder."),
        ],
        "learning_tutor": [
            ("asking about fractions", "What is one half?", "One half means one part out of two equal parts."),
            ("grammar help", "When do I use went?", "Use went for the past. I went to school yesterday."),
            ("science question", "Why does ice melt?", "Ice melts when it gains heat. It becomes liquid water."),
            ("history question", "Why do maps matter in history?", "Maps show where events happened and how borders changed."),
            ("vocabulary help", "What does 'concise' mean?", "Concise means expressing something clearly in few words."),
            ("multiplication help", "What is 7 times 8?", "7 times 8 is 56."),
            ("reading comprehension", "How do I find the main idea?", "Look at the first and last sentences of the paragraph."),
            ("study habits", "How should I study for a test?", "Break the material into small chunks and review daily."),
            ("essay structure", "How do I start an essay?", "Open with a clear sentence stating your main point."),
            ("geometry help", "What is a right angle?", "A right angle measures exactly ninety degrees."),
            ("spelling help", "How do I remember 'necessary'?", "Remember: one collar, two sleeves -- one C, two S's."),
            ("note taking", "How should I take notes?", "Write short phrases, not full sentences, and use bullet points."),
        ],
        "coding_mentor": [
            ("debugging Python", "My loop prints too many lines.", "Check the range. The stop value may be too large."),
            ("understanding variables", "What is a variable?", "A variable is a name that stores a value."),
            ("reading an error", "What does NameError mean?", "It usually means Python cannot find that variable name."),
            ("understanding functions", "Why use functions?", "Functions let you reuse steps without repeating code."),
            ("list vs tuple", "What's the difference between a list and a tuple?", "Lists can change after creation; tuples cannot."),
            ("indexing errors", "I got an IndexError.", "You likely tried to access a position beyond the list's length."),
            ("understanding loops", "When do I use a while loop?", "Use a while loop when you don't know how many times to repeat in advance."),
            ("git basics", "What does git commit do?", "It saves a snapshot of your staged changes to the project history."),
            ("testing code", "Why write tests?", "Tests catch mistakes early and confirm code still works after changes."),
            ("naming variables", "How should I name variables?", "Use short, descriptive names that explain what the value represents."),
            ("recursion confusion", "I don't understand recursion.", "A recursive function solves a small case, then calls itself on a smaller one."),
            ("reading stack traces", "The stack trace is confusing.", "Start from the bottom-most line pointing to your own code."),
        ],
        "empathy_support": [
            ("nervous before exam", "I feel nervous about my exam.", "That is normal. Study small parts and take breaks."),
            ("friend conflict", "My friend ignored me.", "Ask calmly what happened. Listen before deciding."),
            ("mistake at work", "I made a mistake.", "Own it, fix what you can, and learn the cause."),
            ("feeling overwhelmed", "I have too much to do.", "Pick one task, finish it, then move to the next."),
            ("disappointment", "I didn't get the result I wanted.", "It's okay to feel disappointed. Consider what to try differently."),
            ("homesickness", "I miss home.", "That feeling is common. Reach out to family when you can."),
            ("public speaking fear", "I'm scared to speak in front of others.", "Practice out loud a few times; familiarity reduces nerves."),
            ("comparison worry", "I feel behind compared to others.", "Everyone moves at a different pace. Focus on your own progress."),
            ("difficult feedback", "I got harsh feedback.", "Take a breath, look for the useful part, and set the rest aside."),
            ("change anxiety", "Things are changing and I feel unsettled.", "Focus on what stays the same and what you can control."),
            ("apologizing", "I need to apologize but don't know how.", "Be specific about what happened and how you'll do better."),
            ("burnout", "I feel exhausted from working nonstop.", "Rest is productive too. Consider a short, real break."),
        ],
        "professional_chat": [
            ("email rewrite", "Can you make this email polite?", "Yes. Keep it short, clear, and respectful."),
            ("meeting plan", "How should I run a meeting?", "Set a goal, list topics, and end with action items."),
            ("status update", "I need to report progress.", "Say what is done, what is blocked, and what comes next."),
            ("giving feedback", "How do I give feedback kindly?", "Be specific, focus on the work, and suggest a next step."),
            ("declining a request", "How do I say no politely?", "Thank them, explain briefly, and offer an alternative if possible."),
            ("negotiating a deadline", "I need more time on a project.", "Explain the reason and propose a new, realistic date early."),
            ("onboarding a teammate", "How do I help a new hire settle in?", "Share key contacts, documents, and a short first-week plan."),
            ("prioritizing tasks", "I have too many tasks today.", "Rank by deadline and impact, then start with the most urgent."),
            ("summarizing a call", "How do I summarize a meeting?", "List decisions made, owners, and deadlines in a few lines."),
            ("cold outreach", "How do I write a cold email?", "Keep it short, state the purpose, and make the ask clear."),
            ("handling conflict", "A coworker disagreed with my plan.", "Ask about their concern directly and look for common ground."),
            ("requesting resources", "How do I ask for more budget?", "Explain the need, the expected benefit, and the cost clearly."),
        ],
    }
    items = scenarios[topic]
    endings = [
        "The best next step is to act carefully and review the result.",
        "The best next step is to keep it simple and adjust later.",
        "The best next step is to ask for help if anything is unclear.",
        "The best next step is to write it down so it isn't forgotten.",
        "The best next step is to check in again after trying it.",
    ]
    period = combinatorial_period(len(items), 9, len(endings))
    blocks = []
    for index in range(min(count, period)):
        item_i, turn, ending_i = mixed_radix_pick(index, len(items), 9, len(endings))
        title, user_text, assistant_text = items[item_i]
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
            f"Assistant: {endings[ending_i]}\n"
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
            ("Shorten this sentence.", "Due to the fact that it was raining, we decided to stay inside.", "Because it was raining, we stayed inside."),
            ("Fix the grammar.", "She don't like the plan.", "She doesn't like the plan."),
            ("Make this more formal.", "Hey, can you send that file?", "Could you please send the file at your convenience?"),
            ("Combine these sentences.", "The dog barked. The dog ran to the door.", "The dog barked and ran to the door."),
            ("Add a stronger verb.", "The team did a good job on the project.", "The team excelled on the project."),
            ("Remove redundancy.", "In my opinion, I think the plan is good.", "I think the plan is good."),
            ("Write a topic sentence.", "Details about rainforests having high rainfall and diverse species.", "Rainforests are defined by heavy rainfall and remarkable species diversity."),
        ],
        "reasoning_tasks": [
            ("Solve the word problem.", "A box has 6 pens. Ravi adds 4 pens. How many pens are there?", "There are 10 pens."),
            ("Choose the safer action.", "A wire is broken. Should Tara touch it or call an adult?", "Tara should call an adult."),
            ("Find the cause.", "The lamp does not turn on. The bulb is loose.", "The loose bulb may be the cause."),
            ("Solve the word problem.", "Lena has 15 stickers and gives 6 away. How many are left?", "9 stickers are left."),
            ("Order the steps.", "Steps: pour water, boil water, add tea leaves, given out of order.", "Pour water, boil water, add tea leaves."),
            ("Spot the contradiction.", "The store is open every day. The store is closed on Sundays.", "These two statements contradict each other."),
            ("Draw a conclusion.", "All birds in the flock flew south. It is now winter here.", "The birds likely migrated for winter."),
            ("Find the missing step.", "Recipe skips from 'mix batter' to 'serve cake' with nothing baked.", "The recipe is missing a baking step."),
            ("Compare two options.", "Option A costs less but takes longer. Option B costs more but is faster.", "Choose based on whether time or cost matters more."),
            ("Explain the pattern.", "2, 4, 6, 8, ...", "The pattern adds 2 to get each next number."),
        ],
        "coding_tasks": [
            ("Write a Python function that adds two numbers.", "Use parameters a and b.", "def add(a, b):\n    return a + b"),
            ("Explain this code.", "print(len([1, 2, 3]))", "It creates a list with three items and prints its length, which is 3."),
            ("Fix the bug.", "for i in range(3):\nprint(i)", "Indent the print line inside the loop."),
            ("Write a function that returns the max of two numbers.", "Use parameters a and b.", "def maximum(a, b):\n    return a if a > b else b"),
            ("Explain this code.", "x = [n * n for n in range(5)]", "It builds a list of squares for numbers 0 through 4 using a list comprehension."),
            ("Fix the bug.", "def greet(name)\n    print('Hello ' + name)", "Add a colon after the function signature: def greet(name):"),
            ("Write a function that checks if a number is even.", "Use one parameter n.", "def is_even(n):\n    return n % 2 == 0"),
            ("Explain this code.", "total = sum([1, 2, 3])", "It adds up the numbers in the list, giving a total of 6."),
            ("Fix the bug.", "if x = 5:\n    print('five')", "Use == for comparison instead of =: if x == 5:"),
            ("Write a function that reverses a string.", "Use one parameter text.", "def reverse(text):\n    return text[::-1]"),
        ],
        "classification_tasks": [
            ("Classify the sentence.", "The sky is cloudy today.", "Category: weather observation."),
            ("Classify the request.", "Can you help me debug this error?", "Category: coding help."),
            ("Classify the emotion.", "I am proud because I finished the project.", "Emotion: proud."),
            ("Classify the sentence.", "Water boils at 100 degrees Celsius.", "Category: science fact."),
            ("Classify the request.", "Please summarize this article for me.", "Category: writing help."),
            ("Classify the emotion.", "I felt nervous before the interview.", "Emotion: nervous."),
            ("Classify the sentence.", "Paris is the capital of France.", "Category: geography fact."),
            ("Classify the request.", "Can you check my math homework?", "Category: math help."),
            ("Classify the emotion.", "I was relieved when the test was over.", "Emotion: relieved."),
            ("Classify the sentence.", "The stock market fell sharply today.", "Category: financial news."),
        ],
        "format_following": [
            ("Answer with two bullet points.", "Give two safe cooking tips.", "- Wash your hands.\n- Turn off the stove after cooking."),
            ("Return only the number.", "What is 8 plus 5?", "13"),
            ("Use a short answer.", "Why do plants need light?", "Plants use light to make food."),
            ("Answer with two bullet points.", "Give two tips for studying.", "- Take short breaks.\n- Review notes daily."),
            ("Return only the number.", "What is 12 minus 7?", "5"),
            ("Use a short answer.", "Why do we wear seatbelts?", "Seatbelts help prevent injury in a crash."),
            ("Answer with three bullet points.", "List three parts of a plant.", "- Roots\n- Stem\n- Leaves"),
            ("Return only the word.", "What do bees produce?", "Honey"),
            ("Use one sentence.", "What is gravity?", "Gravity is the force that pulls objects toward each other."),
            ("Answer in a single word.", "What gas do plants release during photosynthesis?", "Oxygen"),
        ],
    }
    items = tasks[topic]
    closers = [
        "This instruction sample teaches format control.",
        "This instruction sample teaches staying on topic.",
        "This instruction sample teaches concise responses.",
        "This instruction sample teaches following the exact request.",
    ]
    period = combinatorial_period(len(items), 13, len(closers))
    blocks = []
    for index in range(min(count, period)):
        item_i, turn, closer_i = mixed_radix_pick(index, len(items), 13, len(closers))
        instruction, input_text, output_text = items[item_i]
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
            f"{closers[closer_i]} (variant {turn})"
        )
    return blocks


