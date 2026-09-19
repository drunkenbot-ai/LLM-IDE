"""High-quality synthetic thinking dataset generator with <think>...</think> Chain-of-Thought.

Generates diverse multi-domain reasoning records (Math, Logic, Coding, Planning, Science)
and streams them to JSONL files strictly partitioned under 30MB each (default 28MB rollover).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Generator

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah",
    "Ian", "Julia", "Kevin", "Liam", "Mina", "Noah", "Olivia", "Priya",
    "Quinn", "Ravi", "Sophia", "Thomas", "Uma", "Victor", "Wendy", "Xavier",
    "Yasmine", "Zachary", "Aarav", "Elena", "Marcus", "Chloe", "Leo", "Maya"
]

ITEMS = [
    "laptops", "tablets", "smartphones", "books", "notebooks", "solar panels",
    "water filters", "bicycles", "toolkits", "backpacks", "headsets", "cameras"
]

CITIES = [
    "Tokyo", "Paris", "Berlin", "Toronto", "Sydney", "Mumbai", "London", "New York",
    "Singapore", "Seoul", "Amsterdam", "Chicago", "San Francisco", "Austin", "Dublin"
]


# ==============================================================================
# 1. Math & Quantitative Generators
# ==============================================================================

def gen_work_rate(rng: random.Random) -> dict[str, str]:
    name1, name2 = rng.sample(NAMES, 2)
    item = rng.choice(ITEMS)
    t1 = rng.randint(4, 12)
    t2 = rng.randint(4, 12)
    
    # Combined rate = 1/t1 + 1/t2 = (t1+t2)/(t1*t2)
    combined_time = (t1 * t2) / (t1 + t2)
    
    prompt = (
        f"{name1} can assemble a batch of {item} in {t1} hours, while {name2} can assemble "
        f"the exact same batch in {t2} hours. If they work together at their constant rates, "
        f"how long will it take them to complete the batch? Express your answer as a mixed number "
        f"or decimal rounded to two decimal places."
    )
    
    thought = (
        f"<think>\n"
        f"1. Understand the goal: Find the time needed for {name1} and {name2} working together to assemble the batch of {item}.\n"
        f"2. Determine individual rates:\n"
        f"   - {name1}'s work rate: R1 = 1/{t1} of the batch per hour.\n"
        f"   - {name2}'s work rate: R2 = 1/{t2} of the batch per hour.\n"
        f"3. Combined work rate:\n"
        f"   - R_total = R1 + R2 = 1/{t1} + 1/{t2} = ({t1} + {t2}) / ({t1} * {t2}) = {t1 + t2} / {t1 * t2}.\n"
        f"4. Calculate the time required (T = 1 / R_total):\n"
        f"   - T = ({t1} * {t2}) / ({t1 + t2}) = {t1 * t2} / {t1 + t2}.\n"
        f"   - Numeric division: {t1 * t2} / {t1 + t2} = {combined_time:.4f} hours.\n"
        f"   - Rounding to two decimal places gives {combined_time:.2f} hours.\n"
        f"5. Sanity check: When two people work together, the combined time must be strictly less than the faster person's time ({min(t1, t2)}h). {combined_time:.2f} < {min(t1, t2)}, which is consistent.\n"
        f"</think>"
    )
    
    answer = (
        f"Working together, {name1} and {name2} will complete the batch in **{combined_time:.2f} hours** "
        f"(or **{t1 * t2}/{t1 + t2} hours**)."
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


def gen_speed_distance(rng: random.Random) -> dict[str, str]:
    c1, c2 = rng.sample(CITIES, 2)
    name = rng.choice(NAMES)
    d = rng.randint(120, 480)
    s1 = rng.randint(40, 80)
    s2 = rng.randint(30, 60)
    while s1 == s2:
        s2 = rng.randint(30, 60)
        
    t1 = d / s1
    t2 = d / s2
    total_dist = 2 * d
    total_time = t1 + t2
    avg_speed = total_dist / total_time
    
    prompt = (
        f"{name} drove from {c1} to {c2} (a one-way distance of {d} km) at an average speed of {s1} km/h. "
        f"On the return trip along the exact same route, heavy traffic slowed the average speed to {s2} km/h. "
        f"What was {name}'s average speed for the entire round trip?"
    )
    
    thought = (
        f"<think>\n"
        f"1. Identify the common pitfall: The average speed for a round trip is NOT the simple arithmetic mean of the two speeds ({s1} + {s2}) / 2 = {(s1+s2)/2:.1f} km/h. Average speed is defined strictly as total distance divided by total time.\n"
        f"2. Calculate total distance:\n"
        f"   - Outbound distance: {d} km\n"
        f"   - Inbound distance: {d} km\n"
        f"   - Total distance = {d} + {d} = {total_dist} km.\n"
        f"3. Calculate individual times:\n"
        f"   - Time outbound: t1 = {d} / {s1} = {t1:.4f} hours.\n"
        f"   - Time inbound: t2 = {d} / {s2} = {t2:.4f} hours.\n"
        f"   - Total time: T = t1 + t2 = {t1:.4f} + {t2:.4f} = {total_time:.4f} hours.\n"
        f"4. Calculate harmonic average speed:\n"
        f"   - V_avg = Total Distance / Total Time = {total_dist} / {total_time:.4f} = {avg_speed:.2f} km/h.\n"
        f"   - Using the harmonic formula: 2 * s1 * s2 / (s1 + s2) = 2 * {s1} * {s2} / ({s1} + {s2}) = {2 * s1 * s2} / {s1 + s2} = {avg_speed:.2f} km/h.\n"
        f"5. Sanity check: The harmonic mean is always less than the arithmetic mean ({(s1+s2)/2:.1f} km/h). {avg_speed:.2f} < {(s1+s2)/2:.1f}, confirming correctness.\n"
        f"</think>"
    )
    
    answer = (
        f"The average speed for the entire round trip was **{avg_speed:.2f} km/h**.\n\n"
        f"*Note*: Because average speed is total distance divided by total time, it corresponds to the harmonic mean of the two speeds rather than their arithmetic average."
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


def gen_compound_percentage(rng: random.Random) -> dict[str, str]:
    item = rng.choice(ITEMS)
    orig_price = rng.randint(100, 1000)
    p1 = rng.choice([10, 15, 20, 25, 30])
    p2 = rng.choice([10, 15, 20])
    
    price_after_p1 = orig_price * (1 - p1 / 100)
    final_price = price_after_p1 * (1 + p2 / 100)
    net_change = ((final_price - orig_price) / orig_price) * 100
    direction = "decrease" if net_change < 0 else "increase"
    
    prompt = (
        f"A store originally priced a batch of {item} at ${orig_price}. During a weekend promotion, "
        f"the price was discounted by {p1}%. The following week, the discounted price was increased by {p2}%. "
        f"What is the final price of the item, and what is the overall percentage change from the original price?"
    )
    
    thought = (
        f"<think>\n"
        f"1. Understand the sequential percentage changes:\n"
        f"   - Original price: P0 = ${orig_price}.\n"
        f"   - First step: {p1}% discount.\n"
        f"   - Second step: {p2}% markup on the new discounted price.\n"
        f"2. Calculate price after discount:\n"
        f"   - Discount amount = {orig_price} * ({p1}/100) = ${orig_price * p1 / 100:.2f}.\n"
        f"   - P1 = {orig_price} - {orig_price * p1 / 100:.2f} = ${price_after_p1:.2f}.\n"
        f"3. Calculate price after subsequent markup:\n"
        f"   - Markup applies to P1 (${price_after_p1:.2f}), NOT the original price.\n"
        f"   - Markup amount = {price_after_p1:.2f} * ({p2}/100) = ${price_after_p1 * p2 / 100:.2f}.\n"
        f"   - Final Price P2 = {price_after_p1:.2f} + {price_after_p1 * p2 / 100:.2f} = ${final_price:.2f}.\n"
        f"4. Calculate net overall percentage change:\n"
        f"   - Multiplier = (1 - {p1/100}) * (1 + {p2/100}) = {1 - p1/100:.2f} * {1 + p2/100:.2f} = {(1 - p1/100)*(1 + p2/100):.4f}.\n"
        f"   - Percentage change = ({final_price:.2f} - {orig_price}) / {orig_price} * 100 = {net_change:.2f}%.\n"
        f"5. Verify: Sequential percentages cannot be added directly (-{p1}% + {p2}% != {net_change:.2f}%).\n"
        f"</think>"
    )
    
    answer = (
        f"- **Final Price**: **${final_price:.2f}**\n"
        f"- **Overall Change**: A **{abs(net_change):.2f}% {direction}** relative to the original price."
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


def gen_geometry_volume(rng: random.Random) -> dict[str, str]:
    r = rng.randint(3, 15)
    h = rng.randint(5, 25)
    vol = math.pi * (r ** 2) * h
    surface_area = 2 * math.pi * r * h + 2 * math.pi * (r ** 2)
    
    prompt = (
        f"A cylindrical storage tank has an internal radius of {r} meters and a height of {h} meters. "
        f"Calculate both the total volume and total exterior surface area (including top and bottom) "
        f"of the cylinder. Use π ≈ 3.14159."
    )
    
    thought = (
        f"<think>\n"
        f"1. Identify the given dimensions:\n"
        f"   - Cylinder radius r = {r} m\n"
        f"   - Cylinder height h = {h} m\n"
        f"2. Volume formula:\n"
        f"   - V = π * r^2 * h\n"
        f"   - r^2 = {r}^2 = {r**2}\n"
        f"   - V = π * {r**2} * {h} = {r**2 * h}π\n"
        f"   - Numerically: {r**2 * h} * 3.14159265 = {vol:.2f} m³.\n"
        f"3. Total Surface Area formula:\n"
        f"   - A_total = Lateral Area + 2 * Base Area\n"
        f"   - Lateral Area = 2 * π * r * h = 2 * π * {r} * {h} = {2 * r * h}π\n"
        f"   - Base Area (both ends) = 2 * π * r^2 = 2 * π * {r**2} = {2 * r**2}π\n"
        f"   - Total Area = {2 * r * h}π + {2 * r**2}π = {2 * r * h + 2 * r**2}π\n"
        f"   - Numerically: {2 * r * h + 2 * r**2} * 3.14159265 = {surface_area:.2f} m².\n"
        f"4. Sanity check: Units are m³ for volume and m² for surface area. Calculations are verified.\n"
        f"</think>"
    )
    
    answer = (
        f"For a cylinder with radius {r} m and height {h} m:\n"
        f"- **Volume**: **{vol:.2f} m³** (or **{r**2 * h}π m³**)\n"
        f"- **Total Surface Area**: **{surface_area:.2f} m²** (or **{2 * r * h + 2 * r**2}π m²**)"
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


def gen_probability_puzzle(rng: random.Random) -> dict[str, str]:
    red = rng.randint(4, 10)
    blue = rng.randint(4, 10)
    green = rng.randint(2, 6)
    total = red + blue + green
    
    # Probability of drawing 2 red without replacement
    prob_both_red = (red / total) * ((red - 1) / (total - 1))
    # Probability of drawing 1 red and 1 blue in any order
    prob_red_blue = 2 * (red / total) * (blue / (total - 1))
    
    prompt = (
        f"A bag contains {red} red marbles, {blue} blue marbles, and {green} green marbles. "
        f"If two marbles are drawn at random without replacement, calculate:\n"
        f"1. The probability that both marbles are red.\n"
        f"2. The probability that one marble is red and the other is blue."
    )
    
    thought = (
        f"<think>\n"
        f"1. Count total items:\n"
        f"   - Red = {red}, Blue = {blue}, Green = {green}.\n"
        f"   - Total marbles N = {red} + {blue} + {green} = {total}.\n"
        f"2. Question 1: P(Both are red without replacement):\n"
        f"   - First draw is red: P(R1) = {red}/{total}.\n"
        f"   - Remaining marbles: {total - 1}. Remaining red marbles: {red - 1}.\n"
        f"   - Second draw is red: P(R2 | R1) = {red - 1}/{total - 1}.\n"
        f"   - P(Both Red) = ({red}/{total}) * ({red - 1}/{total - 1}) = {red * (red - 1)} / {total * (total - 1)} = {prob_both_red:.4f} (approx {prob_both_red * 100:.2f}%).\n"
        f"3. Question 2: P(One red and one blue in any order):\n"
        f"   - Order 1: Red then Blue -> ({red}/{total}) * ({blue}/{total - 1})\n"
        f"   - Order 2: Blue then Red -> ({blue}/{total}) * ({red}/{total - 1})\n"
        f"   - Since both outcomes are mutually exclusive: P = 2 * ({red} * {blue}) / ({total} * {total - 1}) = {2 * red * blue} / {total * (total - 1)} = {prob_red_blue:.4f} (approx {prob_red_blue * 100:.2f}%).\n"
        f"4. Sanity check: Individual probabilities are bounded between 0 and 1. Denominators reflect sampling without replacement.\n"
        f"</think>"
    )
    
    answer = (
        f"1. **Probability both are red**: **{red * (red - 1)}/{total * (total - 1)}** (≈ **{prob_both_red * 100:.2f}%**)\n"
        f"2. **Probability of one red and one blue**: **{2 * red * blue}/{total * (total - 1)}** (≈ **{prob_red_blue * 100:.2f}%**)"
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


# ==============================================================================
# 2. Logic, Deduction & Puzzle Generators
# ==============================================================================

def gen_knights_and_knaves(rng: random.Random) -> dict[str, str]:
    name1, name2 = rng.sample(NAMES, 2)
    scenarios = [
        (
            f"{name1} says: 'At least one of us is a knave.'",
            f"Let's test both hypotheses for {name1}:\n"
            f"Case A: Assume {name1} is a knave.\n"
            f"- If {name1} is a knave, their statement ('At least one of us is a knave') must be false.\n"
            f"- The negation of 'at least one is a knave' is 'neither of us is a knave' (both are knights).\n"
            f"- But if both are knights, then {name1} is a knight, which contradicts our assumption that {name1} is a knave.\n"
            f"- Therefore, Case A is impossible. {name1} cannot be a knave.\n"
            f"Case B: {name1} is a knight.\n"
            f"- Since {name1} is a knight, their statement is true: 'At least one of us is a knave.'\n"
            f"- Because {name1} is a knight, the knave must be {name2}.\n"
            f"Conclusion: {name1} is a knight and {name2} is a knave.",
            f"**{name1} is a Knight**, and **{name2} is a Knave**."
        ),
        (
            f"{name1} says: 'Both of us are knaves.'",
            f"Let's evaluate the statement 'Both of us are knaves':\n"
            f"Case A: Assume {name1} is a knight.\n"
            f"- Then the statement 'Both of us are knaves' must be true.\n"
            f"- That would mean {name1} is a knave, which directly contradicts {name1} being a knight.\n"
            f"- Therefore, {name1} cannot be a knight.\n"
            f"Case B: {name1} is a knave.\n"
            f"- If {name1} is a knave, the statement 'Both of us are knaves' must be false.\n"
            f"- Since {name1} is indeed a knave, for the statement to be false, {name2} must NOT be a knave.\n"
            f"- Therefore, {name2} must be a knight.\n"
            f"Conclusion: {name1} is a knave and {name2} is a knight.",
            f"**{name1} is a Knave**, and **{name2} is a Knight**."
        ),
    ]
    prompt_stmt, analysis, conclusion = rng.choice(scenarios)
    
    prompt = (
        f"On an island where every inhabitant is either a Knight (who always tells the truth) "
        f"or a Knave (who always lies), you encounter two people, {name1} and {name2}. "
        f"{prompt_stmt} What are the identities of {name1} and {name2}?"
    )
    
    thought = (
        f"<think>\n"
        f"1. Clarify the rules of the island:\n"
        f"   - Knights always tell the truth (T -> T).\n"
        f"   - Knaves always lie (K -> F).\n"
        f"2. Analyze the statement and systematically test cases:\n"
        f"{analysis}\n"
        f"3. Verify consistency across all constraints.\n"
        f"</think>"
    )
    
    answer = f"Based on logical deduction: {conclusion}"
    return {"prompt": prompt, "thought": thought, "answer": answer}


def gen_ordering_puzzle(rng: random.Random) -> dict[str, str]:
    people = rng.sample(NAMES, 4)
    p1, p2, p3, p4 = people
    # Let order be p1 < p2 < p3 < p4 (1st to 4th)
    prompt = (
        f"Four friends—{p1}, {p2}, {p3}, and {p4}—finished in the top four places of a race. "
        f"We know that:\n"
        f"1. {p1} finished before {p2}.\n"
        f"2. {p3} finished after {p2}.\n"
        f"3. {p4} finished immediately after {p3}.\n"
        f"Determine the exact finishing order from 1st to 4th place."
    )
    
    thought = (
        f"<think>\n"
        f"1. Break down the given relational constraints:\n"
        f"   - Clue 1: {p1} < {p2} (p1 finished before p2)\n"
        f"   - Clue 2: {p2} < {p3} (p3 finished after p2)\n"
        f"   - Clue 3: {p4} is immediately after {p3} (p3 directly followed by p4, i.e., position(p4) = position(p3) + 1)\n"
        f"2. Combine the relative sequence:\n"
        f"   - From Clue 1 and 2: {p1} ... {p2} ... {p3}\n"
        f"   - From Clue 3: {p3} and {p4} are adjacent in the order ({p3}, {p4}).\n"
        f"3. Total positions available: 1st, 2nd, 3rd, 4th.\n"
        f"   - Since {p1} precedes {p2}, {p2} precedes {p3}, and {p3} precedes {p4}:\n"
        f"   - 1st place: {p1}\n"
        f"   - 2nd place: {p2}\n"
        f"   - 3rd place: {p3}\n"
        f"   - 4th place: {p4}\n"
        f"4. Verify all clues against this order:\n"
        f"   - Is {p1} before {p2}? Yes (1st vs 2nd).\n"
        f"   - Is {p3} after {p2}? Yes (3rd vs 2nd).\n"
        f"   - Is {p4} immediately after {p3}? Yes (4th immediately follows 3rd).\n"
        f"</think>"
    )
    
    answer = (
        f"The finishing order from 1st to 4th place is:\n"
        f"1. **1st Place**: {p1}\n"
        f"2. **2nd Place**: {p2}\n"
        f"3. **3rd Place**: {p3}\n"
        f"4. **4th Place**: {p4}"
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


def gen_syllogism_logic(rng: random.Random) -> dict[str, str]:
    groups = [
        ("mammals", "warm-blooded animals", "whales", "whales are warm-blooded"),
        ("polygons", "closed figures", "triangles", "triangles are closed figures"),
        ("prime numbers greater than 2", "odd numbers", "17", "17 is an odd number"),
        ("software engineers", "problem solvers", "systems architects", "systems architects are problem solvers"),
    ]
    A, B, C, conclusion = rng.choice(groups)
    
    prompt = (
        f"Evaluate the logical validity of the following argument:\n"
        f"- Premise 1: All {A} are {B}.\n"
        f"- Premise 2: All {C} are {A}.\n"
        f"- Conclusion: Therefore, all {conclusion}.\n"
        f"Is this argument valid? Explain why or why not."
    )
    
    thought = (
        f"<think>\n"
        f"1. Identify the formal structure of the syllogism:\n"
        f"   - Let A = {A}\n"
        f"   - Let B = {B}\n"
        f"   - Let C = {C}\n"
        f"2. Express premises symbolically in set notation / predicate logic:\n"
        f"   - Major premise: ∀x (A(x) -> B(x)) [A ⊆ B]\n"
        f"   - Minor premise: ∀x (C(x) -> A(x)) [C ⊆ A]\n"
        f"3. Apply hypothetical syllogism / transitivity of inclusion:\n"
        f"   - If C ⊆ A and A ⊆ B, then by transitivity, C ⊆ B.\n"
        f"   - This means ∀x (C(x) -> B(x)).\n"
        f"4. Evaluate the conclusion: 'all {conclusion}'.\n"
        f"   - Since {C} is a subset of {B}, the conclusion follows necessarily from the premises.\n"
        f"5. Check soundness vs validity: Validity requires only that IF premises are true, conclusion MUST be true. This argument is formally valid (Modus Barbara).\n"
        f"</think>"
    )
    
    answer = (
        f"**Yes, the argument is logically valid.**\n\n"
        f"### Logical Explanation:\n"
        f"This follows the classic Barbara syllogism (transitivity of universal inclusion):\n"
        f"1. Since all **{C}** belong to **{A}**,\n"
        f"2. And all **{A}** belong to **{B}**,\n"
        f"3. It necessarily follows that all **{C}** must belong to **{B}** ({conclusion})."
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


# ==============================================================================
# 3. Coding & Algorithmic Generators
# ==============================================================================

def gen_code_complexity(rng: random.Random) -> dict[str, str]:
    problems = [
        (
            "Find if an array contains any duplicate elements.",
            "def contains_duplicate(nums):\n    seen = set()\n    for num in nums:\n        if num in seen:\n            return True\n        seen.add(num)\n    return False",
            "O(n) time and O(n) space",
            "1. Analyze the time complexity:\n"
            "   - Building and looking up elements in a hash set has an average time complexity of O(1).\n"
            "   - We iterate over the list of length n exactly once.\n"
            "   - Total time: n iterations * O(1) = O(n) on average.\n"
            "2. Analyze space complexity:\n"
            "   - The `seen` set stores up to n distinct elements in the worst case (when there are no duplicates).\n"
            "   - Total space: O(n).\n"
            "3. Alternative trade-offs:\n"
            "   - Brute force nested loop: O(n^2) time, O(1) space.\n"
            "   - Sorting approach: O(n log n) time, O(1) or O(n) space."
        ),
        (
            "Binary search on a sorted list.",
            "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
            "O(log n) time and O(1) space",
            "1. Analyze the time complexity:\n"
            "   - In each iteration of the while loop, the search interval [low, high] is halved: n, n/2, n/4, ..., 1.\n"
            "   - The maximum number of steps k satisfies n / 2^k = 1 => k = log2(n).\n"
            "   - Therefore, time complexity is O(log n).\n"
            "2. Analyze space complexity:\n"
            "   - Only pointers (low, high, mid) are maintained.\n"
            "   - No auxiliary data structures are allocated.\n"
            "   - Therefore, space complexity is O(1) auxiliary."
        ),
    ]
    task, code, comp, analysis = rng.choice(problems)
    
    prompt = (
        f"Analyze the time and space complexity of the following Python function:\n\n"
        f"```python\n{code}\n```\n\n"
        f"State the Big-O complexity and explain step-by-step why."
    )
    
    thought = (
        f"<think>\n"
        f"1. Deconstruct the function's execution pattern:\n"
        f"   - Task: {task}\n"
        f"{analysis}\n"
        f"4. Double check edge cases (e.g. empty input, single-element input).\n"
        f"</think>"
    )
    
    answer = (
        f"### Complexity Analysis\n"
        f"- **Time Complexity**: **{comp.split(' and ')[0]}**\n"
        f"- **Space Complexity**: **{comp.split(' and ')[1]}**\n\n"
        f"### Explanation:\n"
        f"{analysis.replace('1. ', '- ').replace('2. ', '- ').replace('3. ', '- ')}"
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


def gen_code_debugging(rng: random.Random) -> dict[str, str]:
    bugs = [
        (
            "Removing items from a list while iterating over it",
            "def remove_evens(numbers):\n    for x in numbers:\n        if x % 2 == 0:\n            numbers.remove(x)\n    return numbers\n\nprint(remove_evens([2, 4, 6, 8, 10]))",
            "Modifying a list in-place while iterating shifts element indices, causing subsequent items to be skipped.",
            "def remove_evens(numbers):\n    return [x for x in numbers if x % 2 != 0]",
            "1. Trace execution on input [2, 4, 6, 8, 10]:\n"
            "   - Step 0: Index 0 is 2. It is even, so numbers.remove(2) is called. The list is now [4, 6, 8, 10].\n"
            "   - Step 1: The iterator advances to Index 1. Index 1 in [4, 6, 8, 10] is now 6! The number 4 was skipped.\n"
            "   - 6 is even, so remove(6) is called. List is now [4, 8, 10].\n"
            "   - Step 2: Iterator advances to Index 2, which is 10. The number 8 was skipped.\n"
            "   - Return value is [4, 8] instead of []!\n"
            "2. Identify the fundamental cause: In-place mutation during forward index traversal.\n"
            "3. Best solution: List comprehension filtering or creating a new list."
        ),
        (
            "Default mutable argument in function definition",
            "def append_item(val, items=[]):\n    items.append(val)\n    return items\n\nprint(append_item(1))\nprint(append_item(2))",
            "Default arguments are evaluated once at function definition time, so the same list instance is shared across calls.",
            "def append_item(val, items=None):\n    if items is None:\n        items = []\n    items.append(val)\n    return items",
            "1. Analyze default argument evaluation:\n"
            "   - In Python, `def append_item(val, items=[])` evaluates `[]` once when the function is defined.\n"
            "   - The default argument is stored in `append_item.__defaults__`.\n"
            "2. Trace calls:\n"
            "   - `append_item(1)` appends 1 to the default list -> returns [1].\n"
            "   - `append_item(2)` appends 2 to the same list instance -> returns [1, 2] instead of [2]!\n"
            "3. Fix: Use `None` as default sentinel value and instantiate a new list inside the function."
        )
    ]
    title, code, issue, fix_code, analysis = rng.choice(bugs)
    
    prompt = (
        f"Identify the bug in the following Python snippet, explain why it happens, "
        f"and provide the corrected code:\n\n"
        f"```python\n{code}\n```"
    )
    
    thought = (
        f"<think>\n"
        f"1. Examine the code behavior and intent:\n"
        f"{analysis}\n"
        f"4. Confirm that the proposed fix resolves the unexpected state mutation.\n"
        f"</think>"
    )
    
    answer = (
        f"### Bug Explanation\n"
        f"**Issue**: {issue}\n\n"
        f"### Corrected Code\n"
        f"```python\n{fix_code}\n```"
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


# ==============================================================================
# 4. Common Sense, Strategy & Planning Generators
# ==============================================================================

def gen_strategy_decision(rng: random.Random) -> dict[str, str]:
    topics = [
        (
            "monolithic architecture vs microservices for a 3-person startup",
            "Monolith is strongly recommended for early-stage small teams",
            "1. Evaluate constraints: Team size = 3 developers. Startup phase = early MVP, uncertain product-market fit.\n"
            "2. Compare options:\n"
            "   - Microservices: Adds distributed tracing, network latency, independent CI/CD pipelines, inter-service API versioning, container orchestration (Kubernetes). High operational overhead.\n"
            "   - Monolith: Single codebase, single deployment pipeline, immediate function calls instead of RPCs, rapid schema refactoring.\n"
            "3. Trade-off analysis: For 3 people, managing 5+ microservices diverts 40%+ of engineering effort into DevOps and infrastructure rather than building product features.\n"
            "4. Conclusion: Build a modular monolith first. Decouple services only when domain boundaries and team sizes demand it."
        ),
        (
            "relational database (PostgreSQL) vs document store (MongoDB) for financial ledger transactions",
            "Relational database with ACID transactions (e.g. PostgreSQL) is required",
            "1. Core requirements: Financial ledger transactions require strict consistency, double-entry bookkeeping, auditability, and no data loss.\n"
            "2. Evaluate properties:\n"
            "   - Relational (PostgreSQL): Native multi-row ACID transactions with strict serializability, foreign key constraints preventing orphan records, check constraints for positive balances.\n"
            "   - Document store (MongoDB): Great for flexible schemas and rapid prototyping, but complex multi-document joins and transaction guarantees are harder to enforce declaratively.\n"
            "3. Risk assessment: In financial accounting, schema rigidity is a safety feature, not a hindrance.\n"
            "4. Conclusion: PostgreSQL is the optimal choice."
        )
    ]
    topic, recommendation, analysis = rng.choice(topics)
    
    prompt = (
        f"Provide an architectural evaluation: {topic}. "
        f"What should the team choose and why? Include practical trade-offs."
    )
    
    thought = (
        f"<think>\n"
        f"1. Context and constraints assessment:\n"
        f"{analysis}\n"
        f"4. Synthesize clear recommendation with pros, cons, and migration path.\n"
        f"</think>"
    )
    
    answer = (
        f"### Recommendation\n"
        f"**{recommendation}**.\n\n"
        f"### Key Decision Factors:\n"
        f"{analysis.replace('1. ', '- ').replace('2. ', '- ').replace('3. ', '- ').replace('4. ', '- ')}"
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


def gen_troubleshooting_reasoning(rng: random.Random) -> dict[str, str]:
    scenarios = [
        (
            "An API endpoint suddenly experiences p99 latency spikes from 50ms to 4,000ms while CPU usage remains low (<15%).",
            "I/O or database connection pool exhaustion / lock contention",
            "1. Analyze symptoms:\n"
            "   - p99 latency increased 80x (50ms -> 4000ms).\n"
            "   - CPU is idle (<15%).\n"
            "2. Deduce bottleneck location:\n"
            "   - Low CPU rules out compute-bound bottlenecks (e.g. heavy JSON serialization, regex backtracking, infinite loops).\n"
            "   - Requests are spending 95%+ of their time waiting in blocked states.\n"
            "3. Common culprits for blocked waiting:\n"
            "   - Database connection pool starvation (all connections checked out, new requests queue up waiting for a free connection).\n"
            "   - Database row-level locks or transaction deadlocks.\n"
            "   - External third-party HTTP API call timing out synchronously without an aggressive timeout setting.\n"
            "4. Formulate diagnostic checklist: Check pool saturation metrics, active locks (`pg_stat_activity`), and external HTTP calls."
        ),
        (
            "A web application reports high memory usage that steadily climbs until the process is terminated by the OS OOM killer, even under flat traffic.",
            "Memory leak caused by unbounded global caches, event listener retention, or unclosed file/socket descriptors",
            "1. Analyze the pattern: Steady linear memory accumulation without recovery under constant traffic = Classic memory leak.\n"
            "2. Garbage collection behavior: If GC runs but memory does not drop, references are still held in the root set.\n"
            "3. Common causes:\n"
            "   - Global dicts/caches growing indefinitely without an eviction policy (LRU / TTL).\n"
            "   - Retained closures or callbacks registered to long-lived event emitters.\n"
            "   - C-extension buffer allocations (e.g. image processing or native bindings) not freed.\n"
            "4. Mitigation: Heap snapshot comparison before and after 100 requests; implement bounded caches."
        )
    ]
    problem, cause, analysis = rng.choice(scenarios)
    
    prompt = (
        f"Troubleshoot the following production incident:\n"
        f"**Scenario**: {problem}\n"
        f"What is the most likely root cause, and how should the engineering team diagnose it?"
    )
    
    thought = (
        f"<think>\n"
        f"1. Diagnostic breakdown:\n"
        f"{analysis}\n"
        f"4. Propose immediate actionable verification steps.\n"
        f"</think>"
    )
    
    answer = (
        f"### Most Likely Root Cause\n"
        f"**{cause}**.\n\n"
        f"### Systematic Diagnostic Steps\n"
        f"{analysis.replace('1. ', '- ').replace('2. ', '- ').replace('3. ', '- ').replace('4. ', '- ')}"
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


# ==============================================================================
# 5. Scientific & Physical Reasoning
# ==============================================================================

def gen_physics_reasoning(rng: random.Random) -> dict[str, str]:
    problems = [
        (
            "Why does ice float on liquid water, and why is this property essential for aquatic life in freezing climates?",
            "1. Examine molecular structure of H2O:\n"
            "   - Most substances contract as they cool and freeze because molecular kinetic energy drops and molecules pack tighter.\n"
            "   - Water reaches its maximum density at approximately 4°C (39.2°F).\n"
            "   - Below 4°C, hydrogen bonding forces water molecules into an open hexagonal crystalline lattice.\n"
            "2. Density comparison:\n"
            "   - The crystalline lattice of ice has more empty space between molecules than liquid water.\n"
            "   - Density of ice (≈ 0.917 g/cm³) is lower than liquid water (≈ 1.00 g/cm³), so ice floats (buoyancy principle).\n"
            "3. Ecological impact:\n"
            "   - Because ice floats, lakes freeze from the top down, forming an insulating surface layer.\n"
            "   - Liquid water remains at 4°C at the bottom, allowing aquatic organisms to survive winter.\n"
            "   - If water froze from the bottom up, entire bodies of water would freeze solid, killing all marine life."
        ),
        (
            "If a cannon fires a ball horizontally and a second ball is dropped from the exact same height at the exact same instant (ignoring air resistance), which ball hits the ground first?",
            "1. Identify the physical forces acting on each ball:\n"
            "   - The only force acting on both balls after release is gravity (downward acceleration g ≈ 9.8 m/s²).\n"
            "2. Deconstruct horizontal and vertical motion independence:\n"
            "   - Horizontal velocity has no vertical component.\n"
            "   - Vertical motion equation: h = (1/2) * g * t^2.\n"
            "   - Both balls start with initial vertical velocity v_y0 = 0.\n"
            "   - Both balls fall from the identical height h under the identical vertical acceleration g.\n"
            "3. Calculate time of flight:\n"
            "   - t = sqrt(2h / g) for both balls.\n"
            "4. Conclusion: Both balls hit the flat ground at the exact same time, despite the fired ball landing far downrange."
        )
    ]
    prompt_q, analysis = rng.choice(problems)
    
    prompt = f"Answer the following physics question with a clear explanation: {prompt_q}"
    
    thought = (
        f"<think>\n"
        f"1. Break down physical principles involved:\n"
        f"{analysis}\n"
        f"3. Ensure the distinction between intuitive misconceptions and physical laws is explicit.\n"
        f"</think>"
    )
    
    answer = (
        f"### Physical Explanation\n"
        f"{analysis.replace('1. ', '- ').replace('2. ', '- ').replace('3. ', '- ')}"
    )
    return {"prompt": prompt, "thought": thought, "answer": answer}


ALL_GENERATORS = [
    gen_work_rate,
    gen_speed_distance,
    gen_compound_percentage,
    gen_geometry_volume,
    gen_probability_puzzle,
    gen_knights_and_knaves,
    gen_ordering_puzzle,
    gen_syllogism_logic,
    gen_code_complexity,
    gen_code_debugging,
    gen_strategy_decision,
    gen_troubleshooting_reasoning,
    gen_physics_reasoning,
]


# ==============================================================================
# Streaming Dataset Writer with Size Partitioning (<30MB)
# ==============================================================================

def generate_thinking_dataset(
    output_dir: Path,
    target_files: int = 2,
    max_bytes_per_file: int = 28 * 1024 * 1024, # 28MB rollover
    seed: int = 42,
    system_prompt: str = (
        "You are a helpful, deep-thinking assistant. When presented with complex problems, "
        "first think through the solution step-by-step inside <think>...</think> tags, "
        "then provide your final clear answer."
    ),
) -> list[Path]:
    """Generate partitioned synthetic thinking dataset files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    created_files: list[Path] = []
    
    for file_idx in range(1, target_files + 1):
        file_path = output_dir / f"thinking_reasoning_part_{file_idx:03d}.jsonl"
        print(f"Generating {file_path.name} (target: ~{max_bytes_per_file / (1024*1024):.1f} MB)...")
        
        bytes_written = 0
        record_count = 0
        
        with open(file_path, "w", encoding="utf-8", newline="\n") as handle:
            while bytes_written < max_bytes_per_file:
                gen_fn = rng.choice(ALL_GENERATORS)
                item = gen_fn(rng)
                
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": f"{item['thought']}\n{item['answer']}"},
                    ]
                }
                
                line = json.dumps(record, ensure_ascii=False) + "\n"
                handle.write(line)
                bytes_written += len(line.encode("utf-8"))
                record_count += 1
                
                if record_count % 2500 == 0:
                    current_mb = bytes_written / (1024 * 1024)
                    print(f"  {file_path.name}: {record_count:,} records written ({current_mb:.2f} MB)...")
        
        final_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"Finished {file_path.name}: {record_count:,} records, {final_mb:.2f} MB (capped under 30MB).\n")
        created_files.append(file_path)
        
    return created_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic <think> reasoning dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"E:\AI_Projects\dataset\fine_tune_thinking",
        help="Target output directory for partitioned JSONL files.",
    )
    parser.add_argument(
        "--target-files",
        type=int,
        default=2,
        help="Number of partitioned files to generate.",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=28.0,
        help="Maximum MB per file before rollover (must be <= 29.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for deterministic generation.",
    )
    args = parser.parse_args()
    
    max_bytes = int(min(args.max_mb, 29.0) * 1024 * 1024)
    out_dir = Path(args.output_dir)
    
    print(f"=== Synthetic Thinking Dataset Generator ===")
    print(f"Output directory : {out_dir}")
    print(f"Target files     : {args.target_files}")
    print(f"Max size/file    : {max_bytes / (1024*1024):.1f} MB (strictly under 30MB)")
    print(f"Seed             : {args.seed}\n")
    
    generate_thinking_dataset(
        output_dir=out_dir,
        target_files=args.target_files,
        max_bytes_per_file=max_bytes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
