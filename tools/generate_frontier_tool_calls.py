"""Frontier Tool Calling & Function Execution Dataset Generator (BFCL Standard).

Generates high-capacity, production-grade function calling datasets across:
1. Realistic Production APIs (SQL, Files/OS, Web Search, Cloud DevOps, Financial, Communication, Analytics)
2. Complete Multi-Turn Execution Loops:
   - User Request -> Assistant Tool Call -> Tool Return Result -> Assistant Final Synthesis
3. Parallel Tool Execution (Multiple simultaneous calls for comparison)
4. Negative Examples (Direct conversational responses without unnecessary tool hallucinations)
5. Missing Parameter Clarification (Asking users for mandatory arguments rather than guessing)

Enforces strict sizing constraints: files are partitioned to stay <= 28 MB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

MAX_FILE_BYTES = 28 * 1024 * 1024  # 28 MB ceiling


# ==============================================================================
# Production Tool Registry (OpenAI Tool Specification Format)
# ==============================================================================

TOOLS_REGISTRY = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute a read-only SQL query against the production PostgreSQL analytics cluster.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "The valid SQL SELECT statement to execute."},
                    "database": {"type": "string", "description": "Target database (e.g., 'warehouse', 'users', 'billing').", "default": "warehouse"},
                    "timeout_seconds": {"type": "integer", "description": "Query execution timeout limit.", "default": 30}
                },
                "required": ["sql"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": "Retrieve the column names, data types, and primary/foreign key relationships for a database table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "The name of the database table (e.g., 'orders', 'users', 'inventory')."}
                },
                "required": ["table_name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Perform a real-time web search for current events, technical documentation, or public data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The precise search query terms."},
                    "num_results": {"type": "integer", "description": "Number of top results to retrieve (1-10).", "default": 5},
                    "domain_filter": {"type": "string", "description": "Optional domain filter (e.g., 'github.com', 'docs.python.org')."}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url_content",
            "description": "Download and parse the text/markdown content from a specific web URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The complete HTTP/HTTPS URL to fetch."},
                    "extract_markdown": {"type": "boolean", "description": "Convert HTML to readable markdown.", "default": True}
                },
                "required": ["url"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content from a specified file path in the workspace filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or workspace-relative path to the file."},
                    "offset_line": {"type": "integer", "description": "Starting line number (1-indexed).", "default": 1},
                    "limit_lines": {"type": "integer", "description": "Maximum number of lines to read.", "default": 200}
                },
                "required": ["path"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite content to a file in the workspace filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path."},
                    "content": {"type": "string", "description": "Text content to write to the file."},
                    "overwrite": {"type": "boolean", "description": "Overwrite existing file if present.", "default": True}
                },
                "required": ["path", "content"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Fetch real-time stock price quote, volume, day high/low, and market cap for a given ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., 'AAPL', 'NVDA', 'MSFT')."},
                    "include_extended_hours": {"type": "boolean", "description": "Include pre-market or after-hours quotes.", "default": False}
                },
                "required": ["ticker"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_metrics",
            "description": "Retrieve real-time telemetry metrics (CPU %, Memory %, Error Rate, p99 Latency) for a cloud microservice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "Microservice identifier (e.g., 'auth-service', 'payment-gateway', 'search-api')."},
                    "time_window_minutes": {"type": "integer", "description": "Rolling time window in minutes.", "default": 15}
                },
                "required": ["service_name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email message to designated recipients via corporate SMTP gateway.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}, "description": "List of recipient email addresses."},
                    "subject": {"type": "string", "description": "Subject line of the email."},
                    "body": {"type": "string", "description": "Body text or HTML content."},
                    "cc": {"type": "array", "items": {"type": "string"}, "description": "Optional CC recipient email addresses."}
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Schedule a meeting or calendar event in Google Calendar / Outlook.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title or meeting topic."},
                    "start_time": {"type": "string", "description": "ISO 8601 formatted start datetime (e.g., '2026-09-20T10:00:00Z')."},
                    "duration_minutes": {"type": "integer", "description": "Meeting duration in minutes.", "default": 30},
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee email addresses."}
                },
                "required": ["title", "start_time"],
                "additionalProperties": False
            }
        }
    },
]


# ==============================================================================
# Dynamic Scenario Generators
# ==============================================================================

CITIES = ["New York", "London", "Tokyo", "Singapore", "Berlin", "San Francisco", "Sydney", "Toronto", "Mumbai", "Paris"]
COMPANIES = [
    ("NVDA", "Nvidia"), ("AAPL", "Apple"), ("MSFT", "Microsoft"),
    ("AMZN", "Amazon"), ("GOOGL", "Alphabet"), ("TSLA", "Tesla")
]
SERVICES = ["auth-service", "payment-api", "search-engine", "notification-worker", "order-processor"]


def generate_single_tool_loop(rng: random.Random) -> Dict[str, Any]:
    """Pattern A: User asks question -> Assistant calls single tool -> Tool returns data -> Assistant final synthesis."""
    scenarios = [
        (
            "SQL Query",
            ["query_database", "get_table_schema"],
            "Find the top 5 customers by total order amount in 2026.",
            "query_database",
            {"sql": "SELECT user_id, SUM(amount) AS total_spend FROM orders WHERE order_date >= '2026-01-01' GROUP BY user_id ORDER BY total_spend DESC LIMIT 5;"},
            {"columns": ["user_id", "total_spend"], "rows": [["usr_1092", 48250.0], ["usr_8831", 39120.5], ["usr_0029", 31000.0], ["usr_5419", 29840.2], ["usr_9901", 27400.0]]},
            "Here are the top 5 customers by total spending in 2026 based on the analytics database:\n\n"
            "1. **User `usr_1092`**: $48,250.00\n"
            "2. **User `usr_8831`**: $39,120.50\n"
            "3. **User `usr_0029`**: $31,000.00\n"
            "4. **User `usr_5419`**: $29,840.20\n"
            "5. **User `usr_9901`**: $27,400.00"
        ),
        (
            "Stock Quote",
            ["get_stock_quote", "search_web"],
            "Can you check the current stock price and trading volume for Nvidia?",
            "get_stock_quote",
            {"ticker": "NVDA"},
            {"ticker": "NVDA", "price": 128.45, "change_percent": 3.42, "day_high": 129.80, "day_low": 124.15, "volume": 58240190, "market_cap_billions": 3160.5},
            "Nvidia (`NVDA`) is currently trading at **$128.45** (+3.42% on the session).\n\n"
            "- **Day Range**: $124.15 - $129.80\n"
            "- **Volume**: 58.24M shares\n"
            "- **Market Cap**: $3.16 Trillion"
        ),
        (
            "Cloud Service Telemetry",
            ["get_service_metrics", "read_file"],
            "Check the health and error rates for our payment-api microservice over the last 15 minutes.",
            "get_service_metrics",
            {"service_name": "payment-api", "time_window_minutes": 15},
            {"service": "payment-api", "cpu_percent": 42.5, "memory_percent": 61.2, "error_rate_pct": 0.04, "p99_latency_ms": 118, "active_replicas": 8},
            "The `payment-api` microservice is operating within healthy parameters over the last 15 minutes:\n\n"
            "- **CPU Utilization**: 42.5%\n"
            "- **Memory Usage**: 61.2%\n"
            "- **Error Rate**: 0.04% (well below the 0.5% SLA threshold)\n"
            "- **p99 Latency**: 118 ms\n"
            "- **Healthy Replicas**: 8 of 8 active"
        ),
    ]

    _, tool_names, user_query, call_name, call_args, tool_result, final_resp = rng.choice(scenarios)
    active_tools = [t for t in TOOLS_REGISTRY if t["function"]["name"] in tool_names]
    call_id = f"call_{rng.randint(10000000, 99999999)}"

    return {
        "tools": active_tools,
        "messages": [
            {"role": "user", "content": user_query},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call_name, "arguments": json.dumps(call_args)}
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": call_name,
                "content": json.dumps(tool_result)
            },
            {"role": "assistant", "content": final_resp}
        ]
    }


def generate_parallel_tool_loop(rng: random.Random) -> Dict[str, Any]:
    """Pattern B: User asks comparative question -> Assistant emits 2 parallel tool calls -> Both results returned -> Synthesizes comparison."""
    comp1, comp2 = rng.sample(COMPANIES, 2)
    t1, name1 = comp1
    t2, name2 = comp2

    call_id_1 = f"call_{rng.randint(10000000, 49999999)}"
    call_id_2 = f"call_{rng.randint(50000000, 99999999)}"

    p1 = round(rng.uniform(100.0, 400.0), 2)
    p2 = round(rng.uniform(100.0, 400.0), 2)
    chg1 = round(rng.uniform(-2.5, 4.5), 2)
    chg2 = round(rng.uniform(-2.5, 4.5), 2)

    user_query = f"Compare the current stock price and performance between {name1} and {name2}."

    active_tools = [t for t in TOOLS_REGISTRY if t["function"]["name"] in ["get_stock_quote", "search_web"]]

    return {
        "tools": active_tools,
        "messages": [
            {"role": "user", "content": user_query},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id_1,
                        "type": "function",
                        "function": {"name": "get_stock_quote", "arguments": json.dumps({"ticker": t1})}
                    },
                    {
                        "id": call_id_2,
                        "type": "function",
                        "function": {"name": "get_stock_quote", "arguments": json.dumps({"ticker": t2})}
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": call_id_1,
                "name": "get_stock_quote",
                "content": json.dumps({"ticker": t1, "price": p1, "change_percent": chg1, "market": "NASDAQ"})
            },
            {
                "role": "tool",
                "tool_call_id": call_id_2,
                "name": "get_stock_quote",
                "content": json.dumps({"ticker": t2, "price": p2, "change_percent": chg2, "market": "NASDAQ"})
            },
            {
                "role": "assistant",
                "content": (
                    f"### Stock Comparison: {name1} ({t1}) vs. {name2} ({t2})\n\n"
                    f"| Asset | Current Price ($) | Daily Change (%) | Market |\n"
                    f"| :--- | :--- | :--- | :--- |\n"
                    f"| **{name1} ({t1})** | ${p1:.2f} | {chg1:+.2f}% | NASDAQ |\n"
                    f"| **{name2} ({t2})** | ${p2:.2f} | {chg2:+.2f}% | NASDAQ |\n\n"
                    f"{name1 if chg1 > chg2 else name2} is outperforming on today's session by {abs(chg1 - chg2):.2f}%."
                )
            }
        ]
    }


def generate_negative_tool_case(rng: random.Random) -> Dict[str, Any]:
    """Pattern C: Tools are provided, but user asks a general knowledge/conceptual question that should be answered directly."""
    scenarios = [
        (
            "What is the difference between an inner join and an outer join in SQL?",
            "### Inner Join vs. Outer Join in SQL\n\n"
            "- **`INNER JOIN`**: Returns only the rows where there is a match in **both** joined tables. If a row in Table A has no corresponding match in Table B, it is excluded.\n"
            "- **`OUTER JOIN`**:\n"
            "  - **`LEFT (OUTER) JOIN`**: Returns all rows from Table A, plus matching rows from Table B (unmatched columns in B are filled with `NULL`).\n"
            "  - **`RIGHT (OUTER) JOIN`**: Returns all rows from Table B, plus matching rows from Table A.\n"
            "  - **`FULL (OUTER) JOIN`**: Returns all rows when there is a match in either table."
        ),
        (
            "Explain what the CAP theorem states regarding distributed systems.",
            "The **CAP Theorem** (Brewer's theorem) states that a distributed data store can simultaneously provide at most two of the following three guarantees:\n\n"
            "1. **Consistency (C)**: Every read receives the most recent write or an error.\n"
            "2. **Availability (A)**: Every non-failing node returns a non-error response, without guaranteeing it contains the most recent write.\n"
            "3. **Partition Tolerance (P)**: The system continues to operate despite an arbitrary number of dropped or delayed network packets.\n\n"
            "Because network partitions are inevitable in physical distributed networks, architectures must choose between **CP** (Consistency + Partition Tolerance) or **AP** (Availability + Partition Tolerance)."
        ),
        (
            "How does the Python Global Interpreter Lock (GIL) affect multithreading vs. multiprocessing?",
            "The **Global Interpreter Lock (GIL)** in CPython is a mutex that prevents multiple native threads from executing Python bytecode simultaneously.\n\n"
            "- **Multithreading**: Only one thread executes Python bytecode at a time. Ideal for **I/O-bound** tasks (network requests, disk reads), but cannot scale **CPU-bound** tasks across multiple CPU cores.\n"
            "- **Multiprocessing**: Spawns separate OS processes, each with its own Python interpreter and memory space, bypassing the GIL to achieve true parallel CPU execution across multiple cores."
        ),
    ]

    query, direct_answer = rng.choice(scenarios)
    active_tools = rng.sample(TOOLS_REGISTRY, 4)

    return {
        "tools": active_tools,
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": direct_answer}
        ]
    }


def generate_clarification_tool_case(rng: random.Random) -> Dict[str, Any]:
    """Pattern D: User asks an action that requires mandatory parameters, model asks for clarification rather than hallucinating."""
    scenarios = [
        (
            ["send_email"],
            "Can you send an email to the dev team about the deployment schedule?",
            "I would be glad to send that email, but I need a few details first:\n\n"
            "1. What is the recipient email address for the dev team?\n"
            "2. What subject line and specific message body should I include regarding the deployment schedule?"
        ),
        (
            ["create_calendar_event"],
            "Schedule a quick sync with Priya tomorrow.",
            "To schedule the meeting, please provide:\n\n"
            "1. What time tomorrow should the meeting start?\n"
            "2. What duration should I set (e.g. 15, 30, or 60 minutes)?\n"
            "3. What is Priya's email address so I can invite her?"
        ),
        (
            ["write_file"],
            "Write the deployment script to disk.",
            "Please specify:\n\n"
            "1. What destination file path should I write to (e.g. `scripts/deploy.sh`)?\n"
            "2. What exact content or commands should the deployment script contain?"
        ),
    ]

    tool_names, prompt, clarification = rng.choice(scenarios)
    active_tools = [t for t in TOOLS_REGISTRY if t["function"]["name"] in tool_names]

    return {
        "tools": active_tools,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": clarification}
        ]
    }


def generate_tool_batch(count: int, seed: int = 42) -> Generator[Dict[str, Any], None, None]:
    rng = random.Random(seed)
    generators = [
        generate_single_tool_loop,
        generate_single_tool_loop,
        generate_parallel_tool_loop,
        generate_negative_tool_case,
        generate_clarification_tool_case,
    ]
    for _ in range(count):
        gen = rng.choice(generators)
        yield gen(rng)


def write_partitioned_tool_dataset(
    output_dir: Path,
    target_count: int,
    max_file_mb: float = 28.0,
    seed: int = 42,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_file_mb * 1024 * 1024)

    written_files: List[Path] = []
    part_idx = 1
    cur_file = output_dir / f"clean_tool_call_part_{part_idx:03d}.jsonl"
    cur_bytes = 0
    cur_records = 0
    cur_fp = open(cur_file, "w", encoding="utf-8")
    written_files.append(cur_file)

    print(f"Generating {target_count:,} production-grade tool calling records to {output_dir}...")

    for idx, rec in enumerate(generate_tool_batch(target_count, seed=seed), start=1):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if cur_bytes + line_bytes > max_bytes and cur_records > 0:
            cur_fp.close()
            print(f"  Saved {cur_file.name}: {cur_records:,} records ({cur_bytes / (1024*1024):.2f} MB)")
            part_idx += 1
            cur_file = output_dir / f"clean_tool_call_part_{part_idx:03d}.jsonl"
            cur_fp = open(cur_file, "w", encoding="utf-8")
            written_files.append(cur_file)
            cur_bytes = 0
            cur_records = 0

        cur_fp.write(line)
        cur_bytes += line_bytes
        cur_records += 1

        if idx % 5000 == 0 or idx == target_count:
            print(f"  Progress: {idx:,} / {target_count:,} records processed...")

    cur_fp.close()
    print(f"  Saved {cur_file.name}: {cur_records:,} records ({cur_bytes / (1024*1024):.2f} MB)")
    print(f"Completed! Total partitions written: {len(written_files)}")
    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Frontier Tool Calling Dataset (BFCL Standard)")
    parser.add_argument("--output-dir", type=str, default="dataset/fine_tune_tool_call", help="Target output folder")
    parser.add_argument("--count", type=int, default=50000, help="Number of records to generate")
    parser.add_argument("--max-file-mb", type=float, default=28.0, help="Max MB per partition (default 28.0)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    write_partitioned_tool_dataset(out_path, args.count, max_file_mb=args.max_file_mb, seed=args.seed)


if __name__ == "__main__":
    main()
