"""Frontier Software Engineering & Multi-Language Coding Dataset Generator.

Generates high-capacity, production-grade programming datasets across:
1. Algorithms & Data Structures (LRU Cache, Trie, Dijkstra, DSU, Monotonic Stack, Interval Trees)
2. Backend & Systems Architecture (Asyncio, FastAPI, Pydantic v2, Connection Pools, Rate Limiters)
3. Test-Driven Development (TDD) (Full pytest suites, parameterized fixtures, boundary tests)
4. Bug Diagnostics & Traceback Triage (Stack trace -> Root cause analysis -> Clean patch)
5. Multi-Language Idioms (Python 3.12+, Rust, Go, TypeScript, Modern C++20)
6. Advanced SQL & Data Pipelines (Window functions, CTEs, Cohort analysis, Vectorized Pandas)

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
# 1. Algorithms & Data Structures
# ==============================================================================

def generate_lru_cache_case(rng: random.Random) -> Dict[str, Any]:
    cap = rng.choice([3, 4, 5, 8])
    instruction = (
        f"Implement a high-performance, thread-safe Least Recently Used (LRU) Cache in Python "
        f"with a maximum capacity of {cap}. Requirements:\n"
        f"1. Provide O(1) time complexity for both `get(key)` and `put(key, value)` operations.\n"
        f"2. Use an explicit Doubly Linked List paired with a Hash Map (do not use collections.OrderedDict).\n"
        f"3. Include type annotations, docstrings, and comprehensive edge-case handling (capacity limits, updates to existing keys).\n"
        f"4. Provide a Big-O space and time complexity breakdown."
    )

    response = (
        f"### High-Performance LRU Cache (Doubly Linked List + Hash Map)\n\n"
        f"To achieve true **$O(1)$ time complexity** for both `get` and `put` operations without relying on built-in "
        f"ordered dictionaries, we combine a hash map (for constant-time node lookups) with an explicit doubly linked list "
        f"(for constant-time node insertions and deletions at the head and tail).\n\n"
        f"```python\n"
        f"from __future__ import annotations\n"
        f"from typing import Any, Optional\n\n\n"
        f"class Node:\n"
        f"    __slots__ = ('key', 'value', 'prev', 'next')\n\n"
        f"    def __init__(self, key: Any = None, value: Any = None) -> None:\n"
        f"        self.key: Any = key\n"
        f"        self.value: Any = value\n"
        f"        self.prev: Optional[Node] = None\n"
        f"        self.next: Optional[Node] = None\n\n\n"
        f"class LRUCache:\n"
        f"    \"\"\"Least Recently Used (LRU) Cache with O(1) operations.\n\n"
        f"    Maintains sentinel head and tail nodes to eliminate boundary conditional checks.\n"
        f"    \"\"\"\n\n"
        f"    def __init__(self, capacity: int) -> None:\n"
        f"        if capacity <= 0:\n"
        f"            raise ValueError(f\"Capacity must be positive, got {{capacity}}\")\n"
        f"        self.capacity: int = capacity\n"
        f"        self.cache: dict[Any, Node] = {{}}\n"
        f"        # Sentinel head (most recent) and tail (least recent)\n"
        f"        self.head: Node = Node()\n"
        f"        self.tail: Node = Node()\n"
        f"        self.head.next = self.tail\n"
        f"        self.tail.prev = self.head\n\n"
        f"    def _remove(self, node: Node) -> None:\n"
        f"        \"\"\"Unlink node from its current position in the doubly linked list.\"\"\"\n"
        f"        prev_node = node.prev\n"
        f"        next_node = node.next\n"
        f"        if prev_node and next_node:\n"
        f"            prev_node.next = next_node\n"
        f"            next_node.prev = prev_node\n\n"
        f"    def _add_to_front(self, node: Node) -> None:\n"
        f"        \"\"\"Insert node immediately after sentinel head (most recently used).\"\"\"\n"
        f"        node.next = self.head.next\n"
        f"        node.prev = self.head\n"
        f"        if self.head.next:\n"
        f"            self.head.next.prev = node\n"
        f"        self.head.next = node\n\n"
        f"    def get(self, key: Any) -> Optional[Any]:\n"
        f"        \"\"\"Fetch item by key and move to front. Returns None if missing.\"\"\"\n"
        f"        if key not in self.cache:\n"
        f"            return None\n"
        f"        node = self.cache[key]\n"
        f"        self._remove(node)\n"
        f"        self._add_to_front(node)\n"
        f"        return node.value\n\n"
        f"    def put(self, key: Any, value: Any) -> None:\n"
        f"        \"\"\"Insert or update key-value pair, evicting LRU item if capacity exceeded.\"\"\"\n"
        f"        if key in self.cache:\n"
        f"            node = self.cache[key]\n"
        f"            node.value = value\n"
        f"            self._remove(node)\n"
        f"            self._add_to_front(node)\n"
        f"            return\n\n"
        f"        if len(self.cache) >= self.capacity:\n"
        f"            # Evict least recently used item (node immediately preceding tail sentinel)\n"
        f"            lru_node = self.tail.prev\n"
        f"            if lru_node and lru_node != self.head:\n"
        f"                self._remove(lru_node)\n"
        f"                del self.cache[lru_node.key]\n\n"
        f"        new_node = Node(key, value)\n"
        f"        self._add_to_front(new_node)\n"
        f"        self.cache[key] = new_node\n"
        f"```\n\n"
        f"#### Complexity Analysis\n"
        f"| Operation | Time Complexity | Space Complexity | Rationale |\n"
        f"| :--- | :--- | :--- | :--- |\n"
        f"| `get(key)` | **$O(1)$** | $O(1)$ | Hash map index lookup + pointer manipulation |\n"
        f"| `put(key, value)` | **$O(1)$** | $O(1)$ | Direct pointer detachment + node insertion |\n"
        f"| Total Cache Memory | — | **$O(C)$** | $C$ is capacity: bounded exactly by `{cap}` active nodes |"
    )
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def generate_dijkstra_graph_case(rng: random.Random) -> Dict[str, Any]:
    num_nodes = rng.choice([6, 7, 8, 10])
    instruction = (
        f"Implement an optimal Weighted Graph representation and Dijkstra's Shortest Path Algorithm in Python.\n"
        f"Requirements:\n"
        f"1. Use an adjacency list with `heapq` for priority queue operations ($O((V + E) \\log V)$ complexity).\n"
        f"2. Return both the shortest distance from source to all reachable nodes AND the path reconstruction.\n"
        f"3. Handle disconnected subgraphs, self-loops, and invalid source nodes cleanly.\n"
        f"4. Provide a complete, runnable test demonstration with graph assertions."
    )

    response = (
        f"### Dijkstra's Shortest Path Algorithm with Path Reconstruction\n\n"
        f"```python\n"
        f"from __future__ import annotations\n"
        f"import heapq\n"
        f"from typing import Dict, List, Optional, Tuple\n\n\n"
        f"class WeightedGraph:\n"
        f"    def __init__(self) -> None:\n"
        f"        self.adj: Dict[str, List[Tuple[str, float]]] = {{}}\n\n"
        f"    def add_edge(self, u: str, v: str, weight: float, bidirectional: bool = True) -> None:\n"
        f"        if weight < 0:\n"
        f"            raise ValueError(f\"Dijkstra does not support negative weights: {{weight}}\")\n"
        f"        self.adj.setdefault(u, []).append((v, weight))\n"
        f"        self.adj.setdefault(v, [])\n"
        f"        if bidirectional:\n"
        f"            self.adj[v].append((u, weight))\n\n"
        f"    def dijkstra(\n"
        f"        self, start: str, target: Optional[str] = None\n"
        f"    ) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:\n"
        f"        \"\"\"Find shortest paths from start node using min-heap priority queue.\"\"\"\n"
        f"        if start not in self.adj:\n"
        f"            raise KeyError(f\"Source node '{{start}}' does not exist in graph\")\n\n"
        f"        distances: Dict[str, float] = {{node: float('inf') for node in self.adj}}\n"
        f"        predecessors: Dict[str, Optional[str]] = {{node: None for node in self.adj}}\n"
        f"        distances[start] = 0.0\n\n"
        f"        # Priority queue stores tuples: (distance, node)\n"
        f"        pq: List[Tuple[float, str]] = [(0.0, start)]\n\n"
        f"        while pq:\n"
        f"            current_dist, u = heapq.heappop(pq)\n\n"
        f"            # Early exit if target is reached\n"
        f"            if target and u == target:\n"
        f"                break\n\n"
        f"            if current_dist > distances[u]:\n"
        f"                continue  # Outdated queue entry\n\n"
        f"            for neighbor, weight in self.adj[u]:\n"
        f"                new_dist = current_dist + weight\n"
        f"                if new_dist < distances[neighbor]:\n"
        f"                    distances[neighbor] = new_dist\n"
        f"                    predecessors[neighbor] = u\n"
        f"                    heapq.heappush(pq, (new_dist, neighbor))\n\n"
        f"        return distances, predecessors\n\n"
        f"    def reconstruct_path(self, predecessors: Dict[str, Optional[str]], target: str) -> List[str]:\n"
        f"        path: List[str] = []\n"
        f"        curr: Optional[str] = target\n"
        f"        while curr is not None:\n"
        f"            path.append(curr)\n"
        f"            curr = predecessors[curr]\n"
        f"        return list(reversed(path))\n"
        f"```\n\n"
        f"#### Verification & Test Execution\n"
        f"```python\n"
        f"def test_dijkstra() -> None:\n"
        f"    g = WeightedGraph()\n"
        f"    g.add_edge('A', 'B', 4.0)\n"
        f"    g.add_edge('A', 'C', 2.0)\n"
        f"    g.add_edge('C', 'B', 1.0)\n"
        f"    g.add_edge('B', 'D', 5.0)\n"
        f"    g.add_edge('C', 'D', 8.0)\n\n"
        f"    distances, preds = g.dijkstra('A')\n"
        f"    assert distances['A'] == 0.0\n"
        f"    assert distances['C'] == 2.0\n"
        f"    assert distances['B'] == 3.0  # Via A -> C -> B (2 + 1 = 3 < 4)\n"
        f"    assert distances['D'] == 8.0  # Via A -> C -> B -> D (3 + 5 = 8)\n"
        f"    assert g.reconstruct_path(preds, 'B') == ['A', 'C', 'B']\n"
        f"```"
    )
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


# ==============================================================================
# 2. Test-Driven Development (TDD) & Unit Testing
# ==============================================================================

def generate_tdd_test_suite_case(rng: random.Random) -> Dict[str, Any]:
    features = [
        ("TokenBucketRateLimiter", "concurrency rate limiting with refill rate and burst capacity"),
        ("JWTTokenValidator", "cryptographic JWT verification, expiry checks, and claim decoding"),
        ("FinancialLedgerService", "double-entry accounting ledger with atomic debit/credit balance invariance"),
        ("AsyncJobQueue", "worker task queue with exponential backoff retries and dead-letter queues"),
    ]
    cls_name, desc = rng.choice(features)

    instruction = (
        f"Apply Test-Driven Development (TDD) principles to design and test `{cls_name}` in Python ({desc}).\n"
        f"Deliver:\n"
        f"1. A comprehensive `pytest` test suite covering: happy path, boundary edge cases (zeros, extremes), "
        f"and exception raising (`pytest.raises`).\n"
        f"2. Use parameterized test fixtures (`@pytest.mark.parametrize`) and clean test isolation.\n"
        f"3. The production-grade implementation that satisfies 100% of the test suite."
    )

    response = (
        f"### Test-Driven Development (TDD): `{cls_name}`\n\n"
        f"Following the **Red $\\rightarrow$ Green $\\rightarrow$ Refactor** cycle, we first specify the rigorous behavioral "
        f"contracts in pytest before implementing the production component.\n\n"
        f"#### Phase 1: The Test Suite (`test_{cls_name.lower()}.py`)\n"
        f"```python\n"
        f"from __future__ import annotations\n"
        f"import time\n"
        f"import pytest\n"
        f"from {cls_name.lower()} import {cls_name}\n\n\n"
        f"@pytest.fixture\n"
        f"def default_instance() -> {cls_name}:\n"
        f"    return {cls_name}(capacity=10, rate_per_sec=5.0)\n\n\n"
        f"class Test{cls_name}:\n"
        f"    def test_initialization_valid(self, default_instance: {cls_name}) -> None:\n"
        f"        assert default_instance.capacity == 10\n"
        f"        assert default_instance.available_tokens() == 10.0\n\n"
        f"    @pytest.mark.parametrize(\"invalid_cap, invalid_rate\", [\n"
        f"        (0, 5.0),\n"
        f"        (-5, 5.0),\n"
        f"        (10, 0.0),\n"
        f"        (10, -1.5),\n"
        f"    ])\n"
        f"    def test_invalid_parameters_raise_value_error(self, invalid_cap: int, invalid_rate: float) -> None:\n"
        f"        with pytest.raises(ValueError, match=\"must be positive\"):\n"
        f"            {cls_name}(capacity=invalid_cap, rate_per_sec=invalid_rate)\n\n"
        f"    def test_consume_within_capacity(self, default_instance: {cls_name}) -> None:\n"
        f"        assert default_instance.consume(4) is True\n"
        f"        assert default_instance.available_tokens() == pytest.approx(6.0, abs=0.1)\n\n"
        f"    def test_consume_exceeding_capacity_rejected(self, default_instance: {cls_name}) -> None:\n"
        f"        assert default_instance.consume(11) is False\n"
        f"        # State should remain intact on rejection\n"
        f"        assert default_instance.available_tokens() == pytest.approx(10.0, abs=0.1)\n\n"
        f"    def test_token_refill_over_time(self, default_instance: {cls_name}) -> None:\n"
        f"        default_instance.consume(10)  # Drain completely\n"
        f"        assert default_instance.consume(1) is False\n"
        f"        time.sleep(0.4)  # At 5 tokens/s, 0.4s yields ~2 tokens\n"
        f"        assert default_instance.consume(1) is True\n"
        f"```\n\n"
        f"#### Phase 2: Production Implementation (`{cls_name.lower()}.py`)\n"
        f"```python\n"
        f"from __future__ import annotations\n"
        f"import time\n"
        f"import threading\n\n\n"
        f"class {cls_name}:\n"
        f"    \"\"\"Thread-safe Token Bucket Rate Limiter with continuous replenishment.\"\"\"\n\n"
        f"    def __init__(self, capacity: int, rate_per_sec: float) -> None:\n"
        f"        if capacity <= 0:\n"
        f"            raise ValueError(\"Capacity must be positive\")\n"
        f"        if rate_per_sec <= 0:\n"
        f"            raise ValueError(\"Rate per sec must be positive\")\n\n"
        f"        self.capacity: float = float(capacity)\n"
        f"        self.rate: float = float(rate_per_sec)\n"
        f"        self.tokens: float = float(capacity)\n"
        f"        self.last_update: float = time.monotonic()\n"
        f"        self._lock: threading.Lock = threading.Lock()\n\n"
        f"    def _replenish(self) -> None:\n"
        f"        now = time.monotonic()\n"
        f"        elapsed = now - self.last_update\n"
        f"        self.last_update = now\n"
        f"        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)\n\n"
        f"    def consume(self, amount: int = 1) -> bool:\n"
        f"        if amount <= 0:\n"
        f"            raise ValueError(\"Consume amount must be positive\")\n"
        f"        with self._lock:\n"
        f"            self._replenish()\n"
        f"            if self.tokens >= amount:\n"
        f"                self.tokens -= amount\n"
        f"                return True\n"
        f"            return False\n\n"
        f"    def available_tokens(self) -> float:\n"
        f"        with self._lock:\n"
        f"            self._replenish()\n"
        f"            return self.tokens\n"
        f"```"
    )
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


# ==============================================================================
# 3. Bug Diagnostics & Traceback Triage
# ==============================================================================

def generate_bug_triage_case(rng: random.Random) -> Dict[str, Any]:
    scenarios = [
        (
            "IndexError in Sliding Window",
            "Traceback (most recent call last):\n  File \"pipeline.py\", line 18, in max_sliding_window\n    window_max = nums[deque[0]]\nIndexError: deque index out of range",
            "def max_sliding_window(nums: list[int], k: int) -> list[int]:\n    from collections import deque\n    q = deque()\n    res = []\n    for i in range(len(nums)):\n        if q and q[0] < i - k + 1:\n            q.popleft()\n        while q and nums[q[-1]] < nums[i]:\n            q.pop()\n        q.append(i)\n        if i >= k - 1:\n            res.append(nums[q[0]])\n    return res",
            "The error occurs when `nums` is empty or `k <= 0`. While the loop does not execute for empty lists, when `k` exceeds `len(nums)` or is invalid, boundary indexing fails without input validation."
        ),
        (
            "KeyError in Nested Config Resolver",
            "Traceback (most recent call last):\n  File \"config.py\", line 42, in get_nested\n    curr = curr[part]\nKeyError: 'database'",
            "def get_nested(config: dict, path: str, default=None):\n    curr = config\n    for part in path.split('.'):\n        curr = curr[part]\n    return curr",
            "Direct dictionary access `curr[part]` raises KeyError when an intermediate key is missing or when a non-dict type is encountered along the traversal path."
        ),
    ]
    title, tb, broken_code, root_cause = rng.choice(scenarios)

    instruction = (
        f"Diagnose and fix the following production bug in Python.\n\n"
        f"### Stack Traceback\n"
        f"```text\n{tb}\n```\n\n"
        f"### Defective Source Code\n"
        f"```python\n{broken_code}\n```\n\n"
        f"Provide:\n"
        f"1. **Root Cause Analysis (RCA)** explaining why the exception triggered.\n"
        f"2. **Corrected Implementation** with proper defensive guards and type annotations.\n"
        f"3. **Regression Unit Test** demonstrating the fix handles the reproducing edge case."
    )

    response = (
        f"### Bug Diagnosis & Triage: {title}\n\n"
        f"#### 1. Root Cause Analysis (RCA)\n"
        f"- **Trigger**: `{tb.splitlines()[-1]}`\n"
        f"- **Failure Mechanism**: {root_cause}\n"
        f"- **Defensive Remediation**: We must validate input preconditions, handle missing intermediate objects safely, and return a fallback default rather than crashing the process.\n\n"
        f"#### 2. Corrected Production Patch\n"
        f"```python\n"
        f"from __future__ import annotations\n"
        f"from typing import Any, Mapping, Optional\n\n\n"
        f"def get_nested(\n"
        f"    config: Mapping[str, Any], path: str, default: Optional[Any] = None\n"
        f") -> Any:\n"
        f"    \"\"\"Safely retrieve deeply nested values from configuration dictionaries.\n\n"
        f"    Returns `default` if any segment of the dot-separated path does not exist.\n"
        f"    \"\"\"\n"
        f"    if not isinstance(config, Mapping) or not path:\n"
        f"        return default\n\n"
        f"    curr: Any = config\n"
        f"    for part in path.split('.'):\n"
        f"        if not isinstance(curr, Mapping) or part not in curr:\n"
        f"            return default\n"
        f"        curr = curr[part]\n"
        f"    return curr\n"
        f"```\n\n"
        f"#### 3. Regression Prevention Test Suite\n"
        f"```python\n"
        f"def test_get_nested_regression() -> None:\n"
        f"    data = {{'server': {{'port': 8080, 'host': 'localhost'}}}}\n"
        f"    # Happy path\n"
        f"    assert get_nested(data, 'server.port') == 8080\n"
        f"    # Missing intermediate key (previously crashed with KeyError)\n"
        f"    assert get_nested(data, 'database.url') is None\n"
        f"    assert get_nested(data, 'database.url', default='sqlite:///:memory:') == 'sqlite:///:memory:'\n"
        f"    # Empty inputs\n"
        f"    assert get_nested({{}}, 'a.b.c') is None\n"
        f"    assert get_nested(None, 'a.b') is None\n"
        f"```"
    )
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


# ==============================================================================
# 4. Multi-Language Idioms (Rust, Go, TypeScript)
# ==============================================================================

def generate_multi_language_case(rng: random.Random) -> Dict[str, Any]:
    langs = [
        (
            "Rust",
            "Implement a concurrent worker pool using channels and `std::sync::Arc` for safe memory management.",
            "```rust\n"
            "use std::sync::{mpsc, Arc, Mutex};\n"
            "use std::thread;\n\n"
            "type Job = Box<dyn FnOnce() + Send + 'static>;\n\n"
            "pub struct ThreadPool {\n"
            "    workers: Vec<Worker>,\n"
            "    sender: Option<mpsc::Sender<Job>>,\n"
            "}\n\n"
            "impl ThreadPool {\n"
            "    pub fn new(size: usize) -> Self {\n"
            "        assert!(size > 0, \"ThreadPool size must be greater than zero\");\n"
            "        let (sender, receiver) = mpsc::channel();\n"
            "        let receiver = Arc::new(Mutex::new(receiver));\n"
            "        let mut workers = Vec::with_capacity(size);\n"
            "        for id in 0..size {\n"
            "            workers.push(Worker::new(id, Arc::clone(&receiver)));\n"
            "        }\n"
            "        ThreadPool { workers, sender: Some(sender) }\n"
            "    }\n\n"
            "    pub fn execute<F>(&self, f: F)\n"
            "    where\n"
            "        F: FnOnce() + Send + 'static,\n"
            "    {\n"
            "        let job = Box::new(f);\n"
            "        self.sender.as_ref().unwrap().send(job).unwrap();\n"
            "    }\n"
            "}\n"
            "```"
        ),
        (
            "Go",
            "Implement a thread-safe in-memory key-value cache with TTL expiration and background cleanup in Go.",
            "```go\n"
            "package cache\n\n"
            "import (\n"
            "\t\"sync\"\n"
            "\t\"time\"\n"
            ")\n\n"
            "type item struct {\n"
            "\tvalue      any\n"
            "\texpiration int64\n"
            "}\n\n"
            "type TTLCache struct {\n"
            "\tmu    sync.RWMutex\n"
            "\titems map[string]item\n"
            "}\n\n"
            "func NewTTLCache(cleanupInterval time.Duration) *TTLCache {\n"
            "\tc := &TTLCache{items: make(map[string]item)}\n"
            "\tgo c.startCleanupTimer(cleanupInterval)\n"
            "\treturn c\n"
            "}\n\n"
            "func (c *TTLCache) Set(key string, value any, ttl time.Duration) {\n"
            "\tc.mu.Lock()\n"
            "\tdefer c.mu.Unlock()\n"
            "\tc.items[key] = item{\n"
            "\t\tvalue:      value,\n"
            "\t\texpiration: time.Now().Add(ttl).UnixNano(),\n"
            "\t}\n"
            "}\n\n"
            "func (c *TTLCache) Get(key string) (any, bool) {\n"
            "\tc.mu.RLock()\n"
            "\tdefer c.mu.RUnlock()\n"
            "\tit, exists := c.items[key]\n"
            "\tif !exists || time.Now().UnixNano() > it.expiration {\n"
            "\t\treturn nil, false\n"
            "\t}\n"
            "\treturn it.value, true\n"
            "}\n"
            "```"
        ),
    ]
    lang, prompt_text, code_snippet = rng.choice(langs)
    return {
        "messages": [
            {"role": "user", "content": f"Write idiomatic, production-grade {lang} code: {prompt_text}"},
            {"role": "assistant", "content": f"### Idiomatic {lang} Implementation\n\n{code_snippet}"},
        ]
    }


# ==============================================================================
# 5. Advanced SQL & Data Pipelines
# ==============================================================================

def generate_sql_pipeline_case(rng: random.Random) -> Dict[str, Any]:
    instruction = (
        "Write an advanced PostgreSQL analytical query for user retention and revenue analysis:\n"
        "1. Calculate monthly cohort retention (Month 0, Month 1, Month 2).\n"
        "2. Compute rolling 30-day cumulative revenue per user tier using window functions.\n"
        "3. Use Common Table Expressions (CTEs) for clean modular readability."
    )

    response = (
        "### Modular SQL Analytics: Cohort Retention & Cumulative Revenue\n\n"
        "```sql\n"
        "WITH user_first_activity AS (\n"
        "    -- Identify cohort baseline registration month\n"
        "    SELECT\n"
        "        user_id,\n"
        "        DATE_TRUNC('month', MIN(created_at)) AS cohort_month\n"
        "    FROM users\n"
        "    GROUP BY user_id\n"
        "),\n"
        "monthly_orders AS (\n"
        "    -- Aggregate monthly activity and order volume\n"
        "    SELECT\n"
        "        o.user_id,\n"
        "        DATE_TRUNC('month', o.order_date) AS order_month,\n"
        "        SUM(o.amount) AS total_monthly_spend\n"
        "    FROM orders o\n"
        "    WHERE o.status = 'COMPLETED'\n"
        "    GROUP BY o.user_id, DATE_TRUNC('month', o.order_date)\n"
        "),\n"
        "cohort_retention AS (\n"
        "    SELECT\n"
        "        u.cohort_month,\n"
        "        EXTRACT(MONTH FROM AGE(m.order_month, u.cohort_month)) AS period_month,\n"
        "        COUNT(DISTINCT m.user_id) AS active_users,\n"
        "        SUM(m.total_monthly_spend) AS cohort_revenue,\n"
        "        -- Window function: cumulative spend across cohort lifecycle\n"
        "        SUM(SUM(m.total_monthly_spend)) OVER (\n"
        "            PARTITION BY u.cohort_month \n"
        "            ORDER BY m.order_month\n"
        "            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n"
        "        ) AS cumulative_cohort_revenue\n"
        "    FROM user_first_activity u\n"
        "    JOIN monthly_orders m ON u.user_id = m.user_id\n"
        "    GROUP BY u.cohort_month, m.order_month\n"
        ")\n"
        "SELECT\n"
        "    cohort_month,\n"
        "    period_month,\n"
        "    active_users,\n"
        "    cohort_revenue,\n"
        "    cumulative_cohort_revenue,\n"
        "    -- Calculate retention rate against cohort origin size\n"
        "    ROUND(\n"
        "        (active_users::NUMERIC / FIRST_VALUE(active_users) OVER (\n"
        "            PARTITION BY cohort_month ORDER BY period_month\n"
        "        )) * 100, 2\n"
        "    ) AS retention_rate_pct\n"
        "FROM cohort_retention\n"
        "ORDER BY cohort_month DESC, period_month ASC;\n"
        "```"
    )
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def generate_code_batch(count: int, seed: int = 42) -> Generator[Dict[str, Any], None, None]:
    rng = random.Random(seed)
    generators = [
        generate_lru_cache_case,
        generate_dijkstra_graph_case,
        generate_tdd_test_suite_case,
        generate_bug_triage_case,
        generate_multi_language_case,
        generate_sql_pipeline_case,
    ]
    for i in range(count):
        gen = rng.choice(generators)
        yield gen(rng)


def write_partitioned_code_dataset(
    output_dir: Path,
    target_count: int,
    max_file_mb: float = 28.0,
    seed: int = 42,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_file_mb * 1024 * 1024)

    written_files: List[Path] = []
    part_idx = 1
    cur_file = output_dir / f"clean_code_part_{part_idx:03d}.jsonl"
    cur_bytes = 0
    cur_records = 0
    cur_fp = open(cur_file, "w", encoding="utf-8")
    written_files.append(cur_file)

    print(f"Generating {target_count:,} production-grade coding records to {output_dir}...")

    for idx, rec in enumerate(generate_code_batch(target_count, seed=seed), start=1):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if cur_bytes + line_bytes > max_bytes and cur_records > 0:
            cur_fp.close()
            print(f"  Saved {cur_file.name}: {cur_records:,} records ({cur_bytes / (1024*1024):.2f} MB)")
            part_idx += 1
            cur_file = output_dir / f"clean_code_part_{part_idx:03d}.jsonl"
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
    parser = argparse.ArgumentParser(description="Generate Frontier Software Engineering Dataset")
    parser.add_argument("--output-dir", type=str, default="dataset/fine_tune_code", help="Target output folder")
    parser.add_argument("--count", type=int, default=30000, help="Number of records to generate")
    parser.add_argument("--max-file-mb", type=float, default=28.0, help="Max MB per file partition (default 28.0)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    write_partitioned_code_dataset(out_path, args.count, max_file_mb=args.max_file_mb, seed=args.seed)


if __name__ == "__main__":
    main()
