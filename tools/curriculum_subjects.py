from __future__ import annotations

try:
    from .curriculum_shared import *
except ImportError:
    from curriculum_shared import *

def language_blocks(count: int) -> list[str]:
    """Create language teaching blocks."""

    subjects = [
        "Mina", "Ravi", "Lena", "Omar", "Sara", "Tara", "Jin", "Ada",
        "Leo", "Nia", "Sam", "Priya", "Yuki", "Noah", "Ines", "Kofi",
    ]
    verbs = [
        "reads", "writes", "asks", "answers", "walks", "listens",
        "draws", "explains", "counts", "builds", "shares", "practices",
    ]
    objects = [
        "a book", "a note", "a question", "a sentence", "a story",
        "a message", "a poem", "a list", "a letter", "a riddle",
    ]
    qualities = [
        "clear", "short", "kind", "useful", "simple", "careful",
        "honest", "tidy", "friendly", "direct",
    ]
    templates = [
        (
            "{subject} {verb} {obj}.\n"
            "The sentence has a subject.\n"
            "The subject tells who acts.\n"
            "The verb tells what happens.\n"
            "The object receives the action.\n"
            "A {quality} sentence is easy to read.\n"
            "What does {subject} do?\n"
            "{subject} {verb} {obj}.\n"
            "Rewrite the idea with fewer words.\n"
            "{subject} {verb}."
        ),
        (
            "Today, {subject} {verb} {obj}.\n"
            "Notice the subject and the verb.\n"
            "{subject} is the subject; {verb} is the verb.\n"
            "{obj} is the object of the sentence.\n"
            "A {quality} sentence keeps its meaning obvious.\n"
            "Try shortening it: {subject} {verb}.\n"
            "Now expand it again with a detail of your own."
        ),
        (
            "Sentence practice: {subject} {verb} {obj}.\n"
            "Underline the subject, then the verb, then the object.\n"
            "A {quality} sentence usually has one clear idea.\n"
            "Ask: who is acting? {subject}.\n"
            "Ask: what do they do? {verb}.\n"
            "Ask: what receives it? {obj}."
        ),
    ]
    period = combinatorial_period(len(subjects), len(verbs), len(objects), len(qualities), len(templates))
    blocks = []
    for index in range(min(count, period)):
        s_i, v_i, o_i, q_i, t_i = mixed_radix_pick(
            index, len(subjects), len(verbs), len(objects), len(qualities), len(templates)
        )
        blocks.append(
            templates[t_i].format(
                subject=subjects[s_i], verb=verbs[v_i], obj=objects[o_i], quality=qualities[q_i]
            )
        )
    return blocks


def math_blocks(count: int) -> list[str]:
    """Create math teaching blocks."""

    nouns = [
        "pencils", "marbles", "stickers", "apples", "coins", "buttons",
        "stones", "cards", "stamps", "beads",
    ]
    templates = [
        (
            "A box has {a} {noun}.\n"
            "Another box has {b} {noun}.\n"
            "{a} plus {b} equals {total}.\n"
            "Together the boxes have {total} {noun}.\n"
            "If there are {a} groups of {b}, multiply.\n"
            "{a} times {b} equals {product}.\n"
            "Addition joins amounts.\n"
            "Multiplication joins equal groups.\n"
            "Check the answer by counting carefully."
        ),
        (
            "You start with {a} {noun}.\n"
            "You receive {b} more {noun}.\n"
            "How many {noun} in total? Add {a} and {b}.\n"
            "{a} + {b} = {total}.\n"
            "If instead you had {a} equal groups of {b} {noun} each, multiply.\n"
            "{a} x {b} = {product}.\n"
            "Addition combines amounts; multiplication combines equal groups."
        ),
        (
            "There are {a} {noun} in one pile and {b} {noun} in another.\n"
            "Combined, that is {total} {noun} ({a} + {b} = {total}).\n"
            "If you arranged {a} rows of {b} {noun}, the total by multiplication is {product}.\n"
            "Recount to double check: does {total} match, and does {product} match?"
        ),
    ]
    a_values = list(range(2, 302))
    b_values = list(range(1, 201))
    period = combinatorial_period(len(a_values), len(b_values), len(nouns), len(templates))
    blocks = []
    for index in range(min(count, period)):
        a_i, b_i, n_i, t_i = mixed_radix_pick(index, len(a_values), len(b_values), len(nouns), len(templates))
        a, b, noun = a_values[a_i], b_values[b_i], nouns[n_i]
        blocks.append(
            templates[t_i].format(a=a, b=b, noun=noun, total=a + b, product=a * b)
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
        ("volcano", "melted rock rises from below", "pressure can cause an eruption"),
        ("river", "water flows from high to low ground", "rivers carry sediment downstream"),
        ("seed", "a seed holds a tiny plant", "water and warmth help it sprout"),
        ("lightning", "charge can build up in clouds", "a spark jumps between charges"),
        ("skeleton", "bones support the body", "joints let bones move"),
        ("sound", "sound travels as vibrations", "vibrations move through air"),
        ("mirror", "a mirror reflects light", "the reflected image looks reversed"),
        ("compass", "a compass needle is a small magnet", "it points toward magnetic north"),
        ("insect", "many insects have six legs", "some insects go through metamorphosis"),
        ("ice", "water freezes at zero degrees Celsius", "ice is less dense than liquid water"),
        ("gravity", "gravity pulls objects toward Earth", "heavier objects still fall at the same rate"),
        ("photosynthesis", "plants use sunlight to make food", "the process also releases oxygen"),
        ("erosion", "wind and water wear down rock over time", "erosion can reshape landscapes slowly"),
        ("circuit", "a circuit needs a complete loop", "a broken loop stops the current"),
    ]
    period = combinatorial_period(len(topics), len(NAMES), len(MODIFIERS), len(CLOSERS))
    blocks = []
    for index in range(min(count, period)):
        topic_i, name_i, mod_i, close_i = mixed_radix_pick(
            index, len(topics), len(NAMES), len(MODIFIERS), len(CLOSERS)
        )
        name, fact_one, fact_two = topics[topic_i]
        student, modifier, closer = NAMES[name_i], MODIFIERS[mod_i], CLOSERS[close_i]
        blocks.append(
            f"{student} studies a {name}.\n"
            f"{student} observes {modifier}.\n"
            f"{fact_one.capitalize()}.\n"
            f"{fact_two.capitalize()}.\n"
            f"An observation tells what we notice.\n"
            f"A question asks why it happens.\n"
            f"A test can compare two cases.\n"
            f"{closer}"
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
        ("Kenya", "Nairobi", "Africa", "the Great Rift Valley"),
        ("Canada", "Ottawa", "North America", "vast northern forests"),
        ("Australia", "Canberra", "Oceania", "large desert interior"),
        ("Peru", "Lima", "South America", "the Andes mountains"),
        ("Norway", "Oslo", "Europe", "deep coastal fjords"),
        ("Vietnam", "Hanoi", "Asia", "the Mekong Delta"),
        ("Morocco", "Rabat", "Africa", "the Atlas Mountains"),
        ("Mexico", "Mexico City", "North America", "central highland valleys"),
        ("Turkey", "Ankara", "Europe/Asia", "the Bosphorus strait"),
        ("Chile", "Santiago", "South America", "the Atacama Desert"),
    ]
    inventions = [
        "the wheel", "writing", "the compass", "the printing press",
        "the steam engine", "the telescope", "the telegraph", "the light bulb",
    ]
    period = combinatorial_period(len(places), len(inventions), len(NAMES), len(CLOSERS))
    blocks = []
    for index in range(min(count, period)):
        place_i, inv_i, name_i, close_i = mixed_radix_pick(
            index, len(places), len(inventions), len(NAMES), len(CLOSERS)
        )
        country, capital, continent, feature = places[place_i]
        invention, student, closer = inventions[inv_i], NAMES[name_i], CLOSERS[close_i]
        blocks.append(
            f"{student} is learning about {country}.\n"
            f"{country} is in {continent}.\n"
            f"The capital city is {capital}.\n"
            f"A map can show where {country} is.\n"
            f"One known feature is {feature}.\n"
            f"People in each place have culture.\n"
            f"Culture includes food, language, music, and customs.\n"
            f"History studies change over time.\n"
            f"An important invention was {invention}.\n"
            f"{closer}"
        )
    return blocks


def reasoning_blocks(count: int) -> list[str]:
    """Create reasoning teaching blocks."""

    people = ["Tom", "Mina", "Ravi", "Lena", "Omar", "Sara", "Tara", "Jin"]
    items = ["apples", "marbles", "coins", "stickers", "pencils", "cards"]
    templates = [
        (
            "{person} has {a} {item}.\n"
            "{person} gets {b} more {item}.\n"
            "To find the total, add.\n"
            "{a} plus {b} equals {total}.\n"
            "{person} has {total} {item}.\n"
            "If the number goes up, addition may help.\n"
            "If the number goes down, subtraction may help.\n"
            "Choose the operation from the story."
        ),
        (
            "{person} starts with {a} {item} and gives away {b}.\n"
            "{a} minus {b} equals {diff}.\n"
            "{person} now has {diff} {item}.\n"
            "Watch the wording: 'gives away' signals subtraction.\n"
            "Reread the story before picking an operation."
        ),
    ]
    a_values = list(range(2, 101))
    b_values = list(range(1, 61))
    period = combinatorial_period(len(people), len(items), len(a_values), len(b_values), len(templates))
    blocks = []
    for index in range(min(count, period)):
        p_i, i_i, a_i, b_i, t_i = mixed_radix_pick(
            index, len(people), len(items), len(a_values), len(b_values), len(templates)
        )
        person, item, a, b = people[p_i], items[i_i], a_values[a_i], b_values[b_i]
        diff = max(a, b) - min(a, b)
        if diff == 0:
            diff = 1  # avoid a degenerate "gives away everything" sentence
        blocks.append(
            templates[t_i].format(person=person, item=item, a=a, b=b, total=a + b, diff=diff)
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
        ("nervous", "she has a test tomorrow", "a short review calms her down"),
        ("embarrassed", "he tripped in front of others", "a friend jokes kindly and moves on"),
        ("excited", "her team scored a goal", "she cheers for her teammates"),
        ("frustrated", "the puzzle piece will not fit", "he takes a short break and tries again"),
        ("lonely", "his friend moved away", "he writes a letter to stay in touch"),
        ("grateful", "a neighbor helped carry groceries", "she says thank you"),
        ("confused", "the instructions were unclear", "he asks a clarifying question"),
    ]
    period = combinatorial_period(len(feelings), len(NAMES), len(MODIFIERS), len(CLOSERS))
    blocks = []
    for index in range(min(count, period)):
        feel_i, name_i, mod_i, close_i = mixed_radix_pick(
            index, len(feelings), len(NAMES), len(MODIFIERS), len(CLOSERS)
        )
        feeling, cause, response = feelings[feel_i]
        student, modifier, closer = NAMES[name_i], MODIFIERS[mod_i], CLOSERS[close_i]
        blocks.append(
            f"{student} feels {feeling} because {cause}.\n"
            f"A feeling often has a cause.\n"
            f"The feeling can change.\n"
            f"Then {response}.\n"
            f"{student} handles it {modifier}.\n"
            f"Kind words can help people feel safe.\n"
            f"Listening shows respect.\n"
            f"{closer}"
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
        ("pack a bag", "list what is needed", "check the list before leaving"),
        ("plant a garden", "prepare the soil", "water on a regular schedule"),
        ("fix a flat tire", "find a safe spot to stop", "use the right tools carefully"),
        ("write a budget", "list all expenses", "compare expenses to income"),
        ("host a guest", "prepare a clean space", "ask about any needs in advance"),
        ("borrow an item", "ask permission first", "return it in good condition"),
        ("resolve a disagreement", "listen to the other side", "look for a fair compromise"),
    ]
    period = combinatorial_period(len(tasks), len(NAMES), len(MODIFIERS), len(CLOSERS))
    blocks = []
    for index in range(min(count, period)):
        task_i, name_i, mod_i, close_i = mixed_radix_pick(
            index, len(tasks), len(NAMES), len(MODIFIERS), len(CLOSERS)
        )
        task, first, safe = tasks[task_i]
        person, modifier, closer = NAMES[name_i], MODIFIERS[mod_i], CLOSERS[close_i]
        blocks.append(
            f"{person} needs to {task}.\n"
            f"First, {person} {modifier} {first}.\n"
            f"Good planning avoids mistakes.\n"
            f"{person} should {safe}.\n"
            f"Safety protects people.\n"
            f"Responsibility means doing what should be done.\n"
            f"Honesty and care help a community.\n"
            f"{closer}"
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
        ("conditional", "chooses a path based on a check", "if score > 50"),
        ("recursion", "calls itself on a smaller case", "factorial(n - 1)"),
        ("class", "groups data and behavior", "class Counter"),
        ("array index", "points to one item's position", "items[0]"),
        ("boolean", "is either true or false", "is_ready = True"),
        ("string", "stores text", "name = 'Mina'"),
    ]
    period = combinatorial_period(len(ideas), len(NAMES), len(MODIFIERS), len(CLOSERS))
    blocks = []
    for index in range(min(count, period)):
        idea_i, name_i, mod_i, close_i = mixed_radix_pick(
            index, len(ideas), len(NAMES), len(MODIFIERS), len(CLOSERS)
        )
        idea, meaning, example = ideas[idea_i]
        student, modifier, closer = NAMES[name_i], MODIFIERS[mod_i], CLOSERS[close_i]
        blocks.append(
            f"A {idea} {meaning}.\n"
            f"Example: {example}.\n"
            f"{student} reads the code {modifier}.\n"
            f"A programmer reads errors carefully.\n"
            f"Debugging means finding the cause.\n"
            f"Testing checks if code works.\n"
            f"Small steps make hard problems easier.\n"
            f"{closer}"
        )
    return blocks


def code_blocks(count: int) -> list[str]:
    """Create code explanation corpus blocks."""

    operations = [
        ("+", "adds", lambda x, y: x + y),
        ("-", "subtracts", lambda x, y: x - y),
        ("*", "multiplies", lambda x, y: x * y),
    ]
    period = combinatorial_period(300, 200, len(operations))
    blocks = []
    for index in range(min(count, period)):
        x_i, y_i, op_i = mixed_radix_pick(index, 300, 200, len(operations))
        value, other = x_i + 1, y_i + 1
        symbol, verb, func = operations[op_i]
        blocks.append(
            "Python example.\n"
            f"x = {value}\n"
            f"y = {other}\n"
            f"print(x {symbol} y)\n"
            f"x stores {value}.\n"
            f"y stores {other}.\n"
            f"The {symbol} sign {verb} the numbers.\n"
            f"The program prints {func(value, other)}.\n"
            "This example teaches variables and arithmetic."
        )
    return blocks


