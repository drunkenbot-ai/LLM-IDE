"""Frontier Code & Systems Architecture Base Pretraining Generator.

Generates high-capacity, diverse, production-grade source code and architectural
documents formatted for base autoregressive pre-training (next-token prediction).

Covers 4 core tiers:
1. Multi-Language Systems: Rust, Go, Modern C++ (C++20), Python 3.12+, TypeScript, SQL, Shell.
2. Distributed Systems & Infrastructure: Raft consensus, LSM-Tree, Consistent Hashing, Epoll, Circuit Breakers.
3. Advanced Data Structures & Algorithmic Rigor: Red-Black Trees, Segment Trees, Tarjan SCC, Trie.
4. Systems Engineering RFCs & Architecture Decision Records (ADRs).

Enforces strict partition sizing: files are partitioned to stay <= 28.0 MB (29,360,128 bytes).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

MAX_PARTITION_BYTES = 28 * 1024 * 1024  # 28 MB ceiling


# ==============================================================================
# Partitioned Writer
# ==============================================================================

class ShardedCodeWriter:
    """Writes JSONL base pretraining records into partitioned files under 28 MB."""

    def __init__(self, output_dir: Path, prefix: str = "base_code_part", max_bytes: int = MAX_PARTITION_BYTES) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.part_idx = 1
        self.cur_file = self.output_dir / f"{self.prefix}_{self.part_idx:03d}.jsonl"
        self.cur_fp = open(self.cur_file, "w", encoding="utf-8")
        self.cur_bytes = 0
        self.cur_records = 0
        self.total_records = 0
        self.total_bytes = 0
        self.written_files: List[Path] = [self.cur_file]

    def write_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        doc = {
            "text": text,
            "meta": meta or {}
        }
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        b_len = len(line.encode("utf-8"))

        if self.cur_bytes + b_len > self.max_bytes and self.cur_records > 0:
            self.cur_fp.close()
            mb = self.cur_bytes / (1024 * 1024)
            print(f"  Saved {self.cur_file.name}: {self.cur_records:,} records ({mb:.2f} MB)")
            self.part_idx += 1
            self.cur_file = self.output_dir / f"{self.prefix}_{self.part_idx:03d}.jsonl"
            self.cur_fp = open(self.cur_file, "w", encoding="utf-8")
            self.written_files.append(self.cur_file)
            self.cur_bytes = 0
            self.cur_records = 0

        self.cur_fp.write(line)
        self.cur_bytes += b_len
        self.cur_records += 1
        self.total_records += 1
        self.total_bytes += b_len

    def close(self) -> None:
        if not self.cur_fp.closed:
            self.cur_fp.close()
            if self.cur_records > 0:
                mb = self.cur_bytes / (1024 * 1024)
                print(f"  Saved {self.cur_file.name}: {self.cur_records:,} records ({mb:.2f} MB)")
            elif len(self.written_files) > 1 and self.cur_file.exists() and self.cur_file.stat().st_size == 0:
                self.cur_file.unlink()
                self.written_files.remove(self.cur_file)


# ==============================================================================
# Helper Heuristic Token Estimator
# ==============================================================================

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ==============================================================================
# Tier 1: Multi-Language Systems Source Code
# ==============================================================================

def gen_rust_tokio_actor_system(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    capacity = rng.choice([32, 64, 128, 256])
    actor_name = rng.choice(["TelemetryPipeline", "OrderMatchingEngine", "StateSyncCoordinator", "ReplicationStream"])
    text = f"""//! Module: {actor_name.lower()}_actor.rs
//! High-throughput asynchronous actor system implementing the actor model over Tokio mpsc channels.
//!
//! # Architecture & Safety Invariants:
//! 1. Zero-shared mutable state: All mutation occurs strictly within the actor loop thread.
//! 2. Bounded backpressure: Channel capacity is fixed at {capacity} to prevent memory exhaustion under burst loads.
//! 3. Graceful shutdown: Handles `oneshot` return channels with timeout drops.

use std::sync::atomic::{{AtomicU64, Ordering}};
use std::sync::Arc;
use tokio::sync::{{mpsc, oneshot}};
use tokio::time::{{timeout, Duration}};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ActorError {{
    #[error("Actor mailbox dropped or worker task terminated")]
    MailboxClosed,
    #[error("Operation timed out after {{0:?}}")]
    Timeout(Duration),
    #[error("Internal state corruption: {{0}}")]
    CorruptedState(String),
}}

#[derive(Debug)]
pub enum {actor_name}Message {{
    RecordEvent {{
        event_id: u64,
        payload: Vec<u8>,
        respond_to: oneshot::Sender<Result<u64, ActorError>>,
    }},
    GetSnapshot {{
        respond_to: oneshot::Sender<Vec<u8>>,
    }},
    Shutdown {{
        force: bool,
    }},
}}

#[derive(Clone)]
pub struct {actor_name}Handle {{
    sender: mpsc::Sender<{actor_name}Message>,
    active_requests: Arc<AtomicU64>,
}}

impl {actor_name}Handle {{
    pub fn new(capacity: usize) -> Self {{
        let (sender, receiver) = mpsc::channel(capacity);
        let active_requests = Arc::new(AtomicU64::new(0));

        let actor = {actor_name}Actor::new(receiver, Arc::clone(&active_requests));
        tokio::spawn(actor.run());

        Self {{
            sender,
            active_requests,
        }}
    }}

    pub async fn record_event(&self, event_id: u64, payload: Vec<u8>, timeout_dur: Duration) -> Result<u64, ActorError> {{
        let (respond_to, rx) = oneshot::channel();
        let msg = {actor_name}Message::RecordEvent {{
            event_id,
            payload,
            respond_to,
        }};

        self.sender.send(msg).await.map_err(|_| ActorError::MailboxClosed)?;
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        match timeout(timeout_dur, rx).await {{
            Ok(Ok(result)) => {{
                self.active_requests.fetch_sub(1, Ordering::Relaxed);
                result
            }}
            Ok(Err(_)) => {{
                self.active_requests.fetch_sub(1, Ordering::Relaxed);
                Err(ActorError::MailboxClosed)
            }}
            Err(_) => {{
                self.active_requests.fetch_sub(1, Ordering::Relaxed);
                Err(ActorError::Timeout(timeout_dur))
            }}
        }}
    }}
}}

struct {actor_name}Actor {{
    receiver: mpsc::Receiver<{actor_name}Message>,
    processed_count: u64,
    events_buffer: Vec<(u64, Vec<u8>)>,
    active_requests: Arc<AtomicU64>,
}}

impl {actor_name}Actor {{
    fn new(receiver: mpsc::Receiver<{actor_name}Message>, active_requests: Arc<AtomicU64>) -> Self {{
        Self {{
            receiver,
            processed_count: 0,
            events_buffer: Vec::with_capacity(1024),
            active_requests,
        }}
    }}

    async fn run(mut self) {{
        while let Some(msg) = self.receiver.recv().await {{
            match msg {{
                {actor_name}Message::RecordEvent {{ event_id, payload, respond_to }} => {{
                    self.processed_count += 1;
                    self.events_buffer.push((event_id, payload));
                    let _ = respond_to.send(Ok(self.processed_count));
                }}
                {actor_name}Message::GetSnapshot {{ respond_to }} => {{
                    let serialized = format!("events_count: {{}}", self.events_buffer.len()).into_bytes();
                    let _ = respond_to.send(serialized);
                }}
                {actor_name}Message::Shutdown {{ force }} => {{
                    if force {{
                        break;
                    }}
                    // Drain remaining messages before exiting
                    while let Ok(remaining_msg) = self.receiver.try_recv() {{
                        if let {actor_name}Message::RecordEvent {{ respond_to, .. }} = remaining_msg {{
                            let _ = respond_to.send(Err(ActorError::MailboxClosed));
                        }}
                    }}
                    break;
                }}
            }}
        }}
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[tokio::test]
    async fn test_actor_throughput_and_lifecycle() {{
        let handle = {actor_name}Handle::new({capacity});
        let res = handle.record_event(1001, vec![1, 2, 3, 4], Duration::from_millis(500)).await;
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), 1);
    }}
}}
"""
    meta = {
        "domain": "code_systems",
        "language": "rust",
        "component": "tokio_actor_model",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_go_worker_pool_pipeline(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    num_workers = rng.choice([4, 8, 16, 32])
    buffer_sz = rng.choice([128, 256, 512, 1024])
    text = f"""// Package pipeline implements a robust, lock-free concurrent worker pool pattern in Go.
// It leverages CSP channels, context-based cancellation, and atomic instrumentation.
package pipeline

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

var (
	ErrPoolClosed   = errors.New("worker pool is closed to new jobs")
	ErrJobTimeout   = errors.New("job execution exceeded context deadline")
	ErrPanicRecover = errors.New("worker caught unexpected panic")
)

// Task represents a discrete unit of computation with error propagation.
type Task[T any, R any] struct {{
	ID        string
	Payload   T
	Execute   func(ctx context.Context, input T) (R, error)
	ResultCh  chan Result[R]
	CreatedAt time.Time
}}

// Result encapsulates computation return values and telemetry.
type Result[R any] struct {{
	TaskID    string
	Data      R
	Err       error
	Duration  time.Duration
}}

// WorkerPool orchestrates a bounded set of goroutines processing jobs from a buffered channel.
type WorkerPool[T any, R any] struct {{
	workerCount int
	jobQueue    chan Task[T, R]
	wg          sync.WaitGroup
	ctx         context.Context
	cancel      context.CancelFunc
	activeJobs  atomic.Int64
	completed   atomic.Int64
	errorsTotal atomic.Int64
	closed      atomic.Bool
}}

// NewWorkerPool instantiates a worker pool with {num_workers} concurrent workers and a {buffer_sz}-depth queue.
func NewWorkerPool[T any, R any](parentCtx context.Context, workers int, queueDepth int) *WorkerPool[T, R] {{
	if workers <= 0 {{
		workers = {num_workers}
	}}
	if queueDepth <= 0 {{
		queueDepth = {buffer_sz}
	}}

	ctx, cancel := context.WithCancel(parentCtx)
	p := &WorkerPool[T, R]{{
		workerCount: workers,
		jobQueue:    make(chan Task[T, R], queueDepth),
		ctx:         ctx,
		cancel:      cancel,
	}}

	p.start()
	return p
}}

func (p *WorkerPool[T, R]) start() {{
	p.wg.Add(p.workerCount)
	for i := 0; i < p.workerCount; i++ {{
		go p.workerLoop(i)
	}}
}}

func (p *WorkerPool[T, R]) workerLoop(workerID int) {{
	defer p.wg.Done()

	for {{
		select {{
		case <-p.ctx.Done():
			return
		case task, ok := <-p.jobQueue:
			if !ok {{
				return
			}}
			p.activeJobs.Add(1)
			p.processTask(task)
			p.activeJobs.Add(-1)
		}}
	}}
}}

func (p *WorkerPool[T, R]) processTask(task Task[T, R]) {{
	start := time.Now()
	res := Result[R]{{TaskID: task.ID}}

	defer func() {{
		if r := recover(); r != nil {{
			p.errorsTotal.Add(1)
			res.Err = fmt.Errorf("%w: %v", ErrPanicRecover, r)
		}}
		res.Duration = time.Since(start)
		p.completed.Add(1)
		task.ResultCh <- res
		close(task.ResultCh)
	}}()

	data, err := task.Execute(p.ctx, task.Payload)
	if err != nil {{
		p.errorsTotal.Add(1)
		res.Err = err
		return
	}}

	res.Data = data
}}

// Submit enqueues a task for parallel execution with guaranteed channel return.
func (p *WorkerPool[T, R]) Submit(task Task[T, R]) error {{
	if p.closed.Load() {{
		return ErrPoolClosed
	}}

	select {{
	case <-p.ctx.Done():
		return p.ctx.Err()
	case p.jobQueue <- task:
		return nil
	}}
}}

// Shutdown signals workers to cease ingestion and gracefully drains active tasks.
func (p *WorkerPool[T, R]) Shutdown(timeoutDur time.Duration) error {{
	if !p.closed.CompareAndSwap(false, true) {{
		return nil
	}}

	close(p.jobQueue)
	drainDone := make(chan struct{{}})
	go func() {{
		p.wg.Wait()
		close(drainDone)
	}}()

	select {{
	case <-drainDone:
		p.cancel()
		return nil
	case <-time.After(timeoutDur):
		p.cancel()
		return fmt.Errorf("timed out after %v waiting for workers to drain", timeoutDur)
	}}
}}
"""
    meta = {
        "domain": "code_systems",
        "language": "go",
        "component": "concurrent_worker_pool",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_cpp_lockfree_ring_buffer(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    ring_sz = rng.choice([1024, 2048, 4096, 8192])
    text = f"""// Module: lockfree_ring_buffer.hpp
// High-performance Single-Producer Single-Consumer (SPSC) Lock-Free Ring Buffer.
// Conforms to modern C++20 standard using std::atomic, memory order fences, and cache-line padding.

#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>
#include <concepts>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

template <typename T>
concept TriviallyRelocatable = std::is_nothrow_move_constructible_v<T> && std::is_nothrow_destructible_v<T>;

template <TriviallyRelocatable T, std::size_t Capacity = {ring_sz}>
class LockFreeSPSCQueue {{
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be an exact power of two for mask optimization");

public:
    LockFreeSPSCQueue() : head_(0), tail_(0) {{
        storage_ = static_cast<Node*>(::operator new(sizeof(Node) * Capacity, std::align_val_t{{CACHE_LINE_SIZE}}));
    }}

    ~LockFreeSPSCQueue() noexcept {{
        T discard;
        while (pop(discard)) {{}}
        ::operator delete(storage_, std::align_val_t{{CACHE_LINE_SIZE}});
    }}

    LockFreeSPSCQueue(const LockFreeSPSCQueue&) = delete;
    LockFreeSPSCQueue& operator=(const LockFreeSPSCQueue&) = delete;
    LockFreeSPSCQueue(LockFreeSPSCQueue&&) noexcept = delete;
    LockFreeSPSCQueue& operator=(LockFreeSPSCQueue&&) noexcept = delete;

    [[nodiscard]] bool push(const T& item) {{
        return emplace(item);
    }}

    [[nodiscard]] bool push(T&& item) {{
        return emplace(std::move(item));
    }}

    template <typename... Args>
    [[nodiscard]] bool emplace(Args&&... args) {{
        const std::size_t current_tail = tail_.load(std::memory_order_relaxed);
        const std::size_t current_head = head_.load(std::memory_order_acquire);

        // Check if ring buffer is saturated
        if ((current_tail - current_head) >= Capacity) {{
            return false;
        }}

        const std::size_t idx = current_tail & BufferMask;
        ::new (static_cast<void*>(&storage_[idx].data)) T(std::forward<Args>(args)...);

        // Store tail with release semantics to ensure element construction completes before updating cursor
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }}

    [[nodiscard]] bool pop(T& val) noexcept {{
        const std::size_t current_head = head_.load(std::memory_order_relaxed);
        const std::size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {{
            return false; // Queue is empty
        }}

        const std::size_t idx = current_head & BufferMask;
        T* item_ptr = reinterpret_cast<T*>(&storage_[idx].data);
        val = std::move(*item_ptr);
        item_ptr->~T();

        // Release slot back to producer
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }}

    [[nodiscard]] std::size_t size() const noexcept {{
        const std::size_t current_head = head_.load(std::memory_order_relaxed);
        const std::size_t current_tail = tail_.load(std::memory_order_relaxed);
        return (current_tail >= current_head) ? (current_tail - current_head) : 0;
    }}

    [[nodiscard]] bool empty() const noexcept {{
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }}

private:
    static constexpr std::size_t BufferMask = Capacity - 1;

    struct Node {{
        alignas(alignof(T)) std::byte data[sizeof(T)];
    }};

    Node* storage_;

    // Separate head and tail onto distinct cache lines to completely eradicate false sharing
    alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> tail_;
}};
"""
    meta = {
        "domain": "code_systems",
        "language": "cpp",
        "component": "lockfree_spsc_queue",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_python_async_connection_pool(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    max_size = rng.choice([10, 20, 50, 100])
    timeout_s = rng.choice([2.0, 5.0, 10.0])
    text = f"""\"\"\"High-performance asynchronous connection pool implementation with circuit-breaking health checks.

Architecture:
- Protocol-driven resource lifecycle (PEP 544).
- FIFO queue with non-blocking acquire and exponential backoff under exhaustion.
- Reaping background task for dead or expired connections.
\"\"\"

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Generic, Optional, Protocol, TypeVar

logger = logging.getLogger("systems.connection_pool")

T = TypeVar("T", bound="PooledConnection")


class PooledConnection(Protocol):
    \"\"\"Protocol defining required interface for resources managed by the connection pool.\"\"\"

    @property
    def is_healthy(self) -> bool:
        ...

    async def ping(self) -> bool:
        ...

    async def close(self) -> None:
        ...


@dataclass
class ConnectionWrapper(Generic[T]):
    conn: T
    created_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    usage_count: int = 0


class ConnectionPool(Generic[T]):
    \"\"\"Thread-safe, asyncio-native connection pool managing up to {max_size} connections.\"\"\"

    def __init__(
        self,
        factory: Callable[[], Any],
        max_size: int = {max_size},
        acquire_timeout: float = {timeout_s},
        max_idle_seconds: float = 300.0,
        max_lifetime_seconds: float = 3600.0,
    ) -> None:
        self._factory = factory
        self._max_size = max_size
        self._acquire_timeout = acquire_timeout
        self._max_idle_seconds = max_idle_seconds
        self._max_lifetime_seconds = max_lifetime_seconds

        self._available: asyncio.Queue[ConnectionWrapper[T]] = asyncio.Queue()
        self._in_use: set[ConnectionWrapper[T]] = set()
        self._total_created = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self._reaper_task: Optional[asyncio.Task[None]] = None

    async def initialize(self) -> None:
        \"\"\"Starts the background cleanup task.\"\"\"
        self._reaper_task = asyncio.create_task(self._reap_idle_loop())

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[T]:
        \"\"\"Context manager to check out a healthy connection and release it on exit.\"\"\"
        if self._closed:
            raise RuntimeError("ConnectionPool has been closed")

        wrapper = await self._acquire_wrapper()
        try:
            yield wrapper.conn
            wrapper.last_used = time.monotonic()
            wrapper.usage_count += 1
        except Exception:
            # If an error occurred on the connection, verify health before returning
            if not await self._validate_health(wrapper.conn):
                await self._destroy_connection(wrapper)
                raise
            raise
        finally:
            if wrapper in self._in_use:
                self._in_use.remove(wrapper)
                if not self._closed and await self._validate_health(wrapper.conn):
                    await self._available.put(wrapper)
                else:
                    await self._destroy_connection(wrapper)

    async def _acquire_wrapper(self) -> ConnectionWrapper[T]:
        end_time = time.monotonic() + self._acquire_timeout
        while True:
            # 1. Try to pull an idle connection
            try:
                wrapper = self._available.get_nowait()
                if self._is_expired(wrapper) or not await self._validate_health(wrapper.conn):
                    await self._destroy_connection(wrapper)
                    continue
                self._in_use.add(wrapper)
                return wrapper
            except asyncio.QueueEmpty:
                pass

            # 2. If under limit, create a new connection
            async with self._lock:
                if (self._total_created) < self._max_size:
                    raw_conn = await self._factory()
                    wrapper = ConnectionWrapper(conn=raw_conn)
                    self._total_created += 1
                    self._in_use.add(wrapper)
                    return wrapper

            # 3. Wait for an available connection with timeout
            remaining = end_time - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"Exceeded {{self._acquire_timeout}}s waiting for pool connection")

            try:
                wrapper = await asyncio.wait_for(self._available.get(), timeout=remaining)
                if self._is_expired(wrapper) or not await self._validate_health(wrapper.conn):
                    await self._destroy_connection(wrapper)
                    continue
                self._in_use.add(wrapper)
                return wrapper
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(f"Pool exhausted: {{len(self._in_use)}} active connections")

    def _is_expired(self, wrapper: ConnectionWrapper[T]) -> bool:
        now = time.monotonic()
        idle_time = now - wrapper.last_used
        total_age = now - wrapper.created_at
        return idle_time > self._max_idle_seconds or total_age > self._max_lifetime_seconds

    async def _validate_health(self, conn: T) -> bool:
        try:
            return bool(conn.is_healthy and await conn.ping())
        except Exception:
            return False

    async def _destroy_connection(self, wrapper: ConnectionWrapper[T]) -> None:
        async with self._lock:
            self._total_created = max(0, self._total_created - 1)
            self._in_use.discard(wrapper)
            try:
                await wrapper.conn.close()
            except Exception as e:
                logger.warning("Error while closing connection: %s", e)

    async def _reap_idle_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(30.0)
            # Drain queue and check for expired connections
            checked: list[ConnectionWrapper[T]] = []
            while not self._available.empty():
                try:
                    w = self._available.get_nowait()
                    if self._is_expired(w):
                        await self._destroy_connection(w)
                    else:
                        checked.append(w)
                except asyncio.QueueEmpty:
                    break
            for w in checked:
                await self._available.put(w)

    async def close(self) -> None:
        self._closed = True
        if self._reaper_task:
            self._reaper_task.cancel()
        while not self._available.empty():
            w = await self._available.get()
            await self._destroy_connection(w)
        for w in list(self._in_use):
            await self._destroy_connection(w)
"""
    meta = {
        "domain": "code_systems",
        "language": "python",
        "component": "async_connection_pool",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_typescript_state_machine(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    machine_name = rng.choice(["DistributedTransactionCoordinator", "RaftNodeStateMachine", "OrderFulfillmentWorkflow"])
    text = f"""/**
 * @file {machine_name.lower()}.ts
 * @description Type-safe, event-driven state machine with exhaustive compile-time checking,
 * transitions validation, and async lifecycle hooks using TypeScript 5+ type-level programming.
 */

export type TransactionState = 
  | 'INITIALIZED'
  | 'PREPARING'
  | 'COMMITTED'
  | 'ABORTING'
  | 'ROLLBACK_COMPLETE'
  | 'FAILED';

export type TransactionEvent =
  | {{ type: 'PREPARE'; payload: {{ participantIds: string[]; timeoutMs: number }} }}
  | {{ type: 'PREPARE_SUCCESS'; payload: {{ voteCount: number }} }}
  | {{ type: 'PREPARE_FAIL'; payload: {{ reason: string }} }}
  | {{ type: 'COMMIT' }}
  | {{ type: 'COMMIT_ACK'; payload: {{ txHash: string }} }}
  | {{ type: 'ABORT'; payload: {{ errorCode: number }} }};

export interface StateContext {{
  txId: string;
  participants: string[];
  preparedVotes: number;
  failureReason?: string;
  txHash?: string;
  updatedAt: Date;
}}

type TransitionMap = {{
  INITIALIZED: 'PREPARING' | 'FAILED';
  PREPARING: 'COMMITTED' | 'ABORTING' | 'FAILED';
  COMMITTED: never;
  ABORTING: 'ROLLBACK_COMPLETE' | 'FAILED';
  ROLLBACK_COMPLETE: never;
  FAILED: never;
}};

export type ValidNextState<S extends TransactionState> = TransitionMap[S];

export interface StateTransitionResult<S extends TransactionState> {{
  success: boolean;
  previousState: TransactionState;
  currentState: S;
  context: Readonly<StateContext>;
  timestamp: number;
}}

export class {machine_name} {{
  private state: TransactionState = 'INITIALIZED';
  private context: StateContext;
  private readonly listeners: Map<TransactionState, Array<(ctx: StateContext) => Promise<void>>> = new Map();

  constructor(txId: string, participants: string[]) {{
    this.context = {{
      txId,
      participants: [...participants],
      preparedVotes: 0,
      updatedAt: new Date(),
    }};
  }}

  public getState(): TransactionState {{
    return this.state;
  }}

  public getContext(): Readonly<StateContext> {{
    return Object.freeze({{ ...this.context }});
  }}

  public on(state: TransactionState, hook: (ctx: StateContext) => Promise<void>): this {{
    const current = this.listeners.get(state) || [];
    current.push(hook);
    this.listeners.set(state, current);
    return this;
  }}

  public async transition(event: TransactionEvent): Promise<StateTransitionResult<TransactionState>> {{
    const prevState = this.state;

    switch (this.state) {{
      case 'INITIALIZED':
        if (event.type === 'PREPARE') {{
          this.state = 'PREPARING';
          this.context.updatedAt = new Date();
        }} else {{
          this.handleInvalidTransition(event);
        }}
        break;

      case 'PREPARING':
        if (event.type === 'PREPARE_SUCCESS') {{
          this.context.preparedVotes = event.payload.voteCount;
          if (this.context.preparedVotes >= this.context.participants.length) {{
            this.state = 'COMMITTED';
          }}
        }} else if (event.type === 'PREPARE_FAIL') {{
          this.context.failureReason = event.payload.reason;
          this.state = 'ABORTING';
        }} else {{
          this.handleInvalidTransition(event);
        }}
        break;

      case 'ABORTING':
        if (event.type === 'ABORT') {{
          this.state = 'ROLLBACK_COMPLETE';
          this.context.updatedAt = new Date();
        }} else {{
          this.handleInvalidTransition(event);
        }}
        break;

      default:
        throw new Error(`Terminal state reached: cannot dispatch event ${{event.type}} from state ${{this.state}}`);
    }}

    await this.notifyListeners(this.state);

    return {{
      success: true,
      previousState: prevState,
      currentState: this.state,
      context: this.getContext(),
      timestamp: Date.now(),
    }};
  }}

  private handleInvalidTransition(event: TransactionEvent): never {{
    throw new Error(`Invalid event type ${{event.type}} dispatched while in state ${{this.state}} for tx ${{this.context.txId}}`);
  }}

  private async notifyListeners(targetState: TransactionState): Promise<void> {{
    const hooks = this.listeners.get(targetState);
    if (!hooks || hooks.length === 0) return;

    for (const hook of hooks) {{
      await hook(this.context);
    }}
  }}
}}
"""
    meta = {
        "domain": "code_systems",
        "language": "typescript",
        "component": "typed_state_machine",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_advanced_sql_window_cte(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    metric = rng.choice(["retention_rate", "clv_cumulative", "drawdown_analysis", "fill_slippage_distribution"])
    text = f"""-- =============================================================================
-- SQL Module: Advanced Financial & Operational Analytics ({metric.upper()})
-- Database Engine: PostgreSQL 16+ / Snowflake / BigQuery
-- Features: Recursive CTEs, Multi-Frame Window Functions, Gap Filling & Time Slicing
-- =============================================================================

WITH RECURSIVE calendar_spine AS (
    -- Anchor member: Generate contiguous calendar dates to eliminate missing trading day anomalies
    SELECT DATE '2025-01-01' AS calendar_date
    UNION ALL
    SELECT (calendar_date + INTERVAL '1 day')::DATE
    FROM calendar_spine
    WHERE calendar_date < DATE '2025-12-31'
),

daily_orders_aggregated AS (
    SELECT
        o.account_id,
        o.symbol,
        o.order_date::DATE AS trade_date,
        COUNT(o.order_id) AS total_orders,
        SUM(o.filled_quantity) AS total_shares_traded,
        SUM(o.filled_quantity * o.execution_price) AS gross_notional_traded,
        AVG(o.slippage_bps) AS avg_slippage_bps,
        MAX(o.execution_price) - MIN(o.execution_price) AS intraday_dispersion
    FROM trades.orders o
    WHERE o.order_status = 'FILLED'
      AND o.order_date >= DATE '2025-01-01'
    GROUP BY o.account_id, o.symbol, o.order_date::DATE
),

spined_portfolio_metrics AS (
    SELECT
        cs.calendar_date,
        doa.account_id,
        doa.symbol,
        COALESCE(doa.total_orders, 0) AS total_orders,
        COALESCE(doa.gross_notional_traded, 0.0) AS daily_notional,
        -- Rolling 30-day cumulative volume using preceding frame clause
        SUM(COALESCE(doa.gross_notional_traded, 0.0)) OVER (
            PARTITION BY doa.account_id, doa.symbol
            ORDER BY cs.calendar_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS rolling_30d_volume,
        -- Window function rank for trading activity tiers
        DENSE_RANK() OVER (
            PARTITION BY cs.calendar_date
            ORDER BY COALESCE(doa.gross_notional_traded, 0.0) DESC
        ) AS daily_volume_rank
    FROM calendar_spine cs
    LEFT JOIN daily_orders_aggregated doa ON cs.calendar_date = doa.trade_date
),

drawdown_analysis AS (
    SELECT
        calendar_date,
        account_id,
        daily_notional,
        rolling_30d_volume,
        -- Calculate High Watermark (HWM)
        MAX(rolling_30d_volume) OVER (
            PARTITION BY account_id
            ORDER BY calendar_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS peak_rolling_volume,
        -- Compute percentage drawdown from historical peak
        CASE 
            WHEN MAX(rolling_30d_volume) OVER (
                PARTITION BY account_id 
                ORDER BY calendar_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) = 0 THEN 0.0
            ELSE ROUND(
                (1.0 - (rolling_30d_volume / NULLIF(MAX(rolling_30d_volume) OVER (
                    PARTITION BY account_id 
                    ORDER BY calendar_date 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ), 0))) * 100.0, 
                4
            )
        END AS drawdown_pct
    FROM spined_portfolio_metrics
)

SELECT
    calendar_date,
    account_id,
    rolling_30d_volume,
    peak_rolling_volume,
    drawdown_pct,
    -- Lead & Lag to detect sudden volatility shocks
    LAG(drawdown_pct, 1, 0.0) OVER (PARTITION BY account_id ORDER BY calendar_date) AS prev_day_drawdown,
    ROUND(
        drawdown_pct - LAG(drawdown_pct, 1, 0.0) OVER (PARTITION BY account_id ORDER BY calendar_date), 
        4
    ) AS daily_drawdown_delta,
    NTILE(10) OVER (PARTITION BY calendar_date ORDER BY drawdown_pct DESC) AS risk_decile
FROM drawdown_analysis
WHERE account_id IS NOT NULL
ORDER BY account_id, calendar_date;
"""
    meta = {
        "domain": "code_systems",
        "language": "sql",
        "component": "window_cte_financial_analytics",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


# ==============================================================================
# Tier 2: Distributed Systems & Infrastructure Architecture
# ==============================================================================

def gen_distributed_raft_consensus(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    cluster_sz = rng.choice([3, 5, 7])
    heartbeat_ms = rng.choice([50, 100, 150])
    text = f"""# Distributed Systems Specification & Implementation: Raft Consensus Protocol
## Cluster Topology: {cluster_sz}-Node Active Quorum
## Target Heartbeat Interval: {heartbeat_ms}ms | Election Timeout Range: {heartbeat_ms*2}ms - {heartbeat_ms*4}ms

```python
\"\"\"Production-grade Raft State Machine implementation in Python.

Implements:
1. Leader Election with randomized randomized election timeouts to eliminate split-vote livelocks.
2. Log Replication with index-term consistency checks.
3. Safety Invariant: Election Safety (at most one leader per term).
4. Safety Invariant: Leader Append-Only (leaders never overwrite or truncate their own entries).
\"\"\"

from __future__ import annotations

import asyncio
import enum
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("raft.consensus")


class NodeRole(enum.Enum):
    FOLLOWER = "FOLLOWER"
    CANDIDATE = "CANDIDATE"
    LEADER = "LEADER"


@dataclass
class LogEntry:
    term: int
    index: int
    command: Any


@dataclass
class AppendEntriesRequest:
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int


@dataclass
class AppendEntriesResponse:
    term: int
    success: bool
    match_index: int


@dataclass
class RequestVoteRequest:
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteResponse:
    term: int
    vote_granted: bool


class RaftNode:
    \"\"\"A single peer in a {cluster_sz}-node Raft consensus cluster.\"\"\"

    def __init__(self, node_id: str, peers: List[str]) -> None:
        self.node_id = node_id
        self.peers = [p for p in peers if p != node_id]
        self.majority = (len(peers) // 2) + 1

        # Persistent state on all servers
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = [LogEntry(term=0, index=0, command=None)]  # 1-indexed sentinel

        # Volatile state on all servers
        self.commit_index = 0
        self.last_applied = 0
        self.role = NodeRole.FOLLOWER

        # Volatile state on leaders (re-initialized after election)
        self.next_index: Dict[str, int] = {{}}
        self.match_index: Dict[str, int] = {{}}

        # Timers and background tasks
        self.last_heartbeat_time = time.monotonic()
        self.election_timeout = self._random_election_timeout()
        self._running = False
        self._lock = asyncio.Lock()

    def _random_election_timeout(self) -> float:
        return random.uniform({heartbeat_ms / 1000.0 * 2:.3f}, {heartbeat_ms / 1000.0 * 4:.3f})

    @property
    def last_log_index(self) -> int:
        return self.log[-1].index

    @property
    def last_log_term(self) -> int:
        return self.log[-1].term

    async def handle_request_vote(self, req: RequestVoteRequest) -> RequestVoteResponse:
        \"\"\"Receiver implementation for RequestVote RPC (§5.2, §5.4).\"\"\"
        async with self._lock:
            # 1. Reply false if term < currentTerm (§5.1)
            if req.term < self.current_term:
                return RequestVoteResponse(term=self.current_term, vote_granted=False)

            if req.term > self.current_term:
                self.current_term = req.term
                self.role = NodeRole.FOLLOWER
                self.voted_for = None

            # 2. If votedFor is null or candidateId, and candidate's log is at least as up-to-date as receiver's log
            can_vote = self.voted_for is None or self.voted_for == req.candidate_id
            log_ok = (req.last_log_term > self.last_log_term) or (
                req.last_log_term == self.last_log_term and req.last_log_index >= self.last_log_index
            )

            if can_vote and log_ok:
                self.voted_for = req.candidate_id
                self.last_heartbeat_time = time.monotonic()
                return RequestVoteResponse(term=self.current_term, vote_granted=True)

            return RequestVoteResponse(term=self.current_term, vote_granted=False)

    async def handle_append_entries(self, req: AppendEntriesRequest) -> AppendEntriesResponse:
        \"\"\"Receiver implementation for AppendEntries RPC (§5.2, §5.3).\"\"\"
        async with self._lock:
            # 1. Reply false if term < currentTerm
            if req.term < self.current_term:
                return AppendEntriesResponse(term=self.current_term, success=False, match_index=0)

            # Valid leader recognized
            if req.term > self.current_term or self.role == NodeRole.CANDIDATE:
                self.current_term = req.term
                self.role = NodeRole.FOLLOWER
                self.voted_for = None

            self.last_heartbeat_time = time.monotonic()

            # 2. Reply false if log doesn't contain an entry at prevLogIndex matching prevLogTerm (§5.3)
            if req.prev_log_index > len(self.log) - 1:
                return AppendEntriesResponse(term=self.current_term, success=False, match_index=len(self.log) - 1)

            if self.log[req.prev_log_index].term != req.prev_log_term:
                # Fast rollback: reject and report conflict
                return AppendEntriesResponse(term=self.current_term, success=False, match_index=req.prev_log_index - 1)

            # 3. If existing entry conflicts with new one, delete existing entry and all that follow
            insert_idx = req.prev_log_index + 1
            for entry in req.entries:
                if insert_idx < len(self.log):
                    if self.log[insert_idx].term != entry.term:
                        self.log = self.log[:insert_idx]
                        self.log.append(entry)
                else:
                    self.log.append(entry)
                insert_idx += 1

            # 4. If leaderCommit > commitIndex, set commitIndex = min(leaderCommit, index of last new entry)
            if req.leader_commit > self.commit_index:
                self.commit_index = min(req.leader_commit, self.last_log_index)

            return AppendEntriesResponse(term=self.current_term, success=True, match_index=self.last_log_index)
```
"""
    meta = {
        "domain": "systems_architecture",
        "language": "python",
        "component": "raft_consensus_specification",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_lsm_tree_storage_engine(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    memtable_flush_threshold = rng.choice([1000, 5000, 10000])
    text = f"""# Storage Engine Internals: Log-Structured Merge-Tree (LSM-Tree)
## Target Architecture: Write-Ahead Log (WAL) + In-Memory SkipList (MemTable) + Immutable SSTables + Leveled Compaction

```python
\"\"\"High-performance LSM-Tree Storage Engine.

Key Architectural Components:
1. Write-Ahead Log (WAL): Sequential append-only on-disk ledger guaranteeing zero data loss.
2. MemTable: In-memory ordered key-value store (SkipList) with O(log N) lookup and insertion.
3. SSTable (Sorted String Table): Immutable on-disk file format composed of block indices and Bloom filters.
4. Compaction: Background merging of overlapping SSTables into sorted tiers, eliminating tombstones.
\"\"\"

from __future__ import annotations

import bisect
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


class BloomFilter:
    \"\"\"Probabilistic set membership filter to avoid expensive disk I/O on SSTable lookups.\"\"\"

    def __init__(self, capacity: int = 10000, fp_rate: float = 0.01) -> None:
        self.capacity = capacity
        self.fp_rate = fp_rate
        # Calculate optimal size (m) and hash functions (k)
        self.size = int(-(capacity * math.log(fp_rate)) / (math.log(2) ** 2))
        self.num_hashes = int((self.size / capacity) * math.log(2))
        self.bitset = bytearray((self.size + 7) // 8)

    def add(self, key: str) -> None:
        for seed in range(self.num_hashes):
            digest = hashlib.md5(f"{{seed}}:{{key}}".encode("utf-8")).digest()
            bit_idx = int.from_bytes(digest[:4], "big") % self.size
            self.bitset[bit_idx // 8] |= (1 << (bit_idx % 8))

    def might_contain(self, key: str) -> bool:
        for seed in range(self.num_hashes):
            digest = hashlib.md5(f"{{seed}}:{{key}}".encode("utf-8")).digest()
            bit_idx = int.from_bytes(digest[:4], "big") % self.size
            if not (self.bitset[bit_idx // 8] & (1 << (bit_idx % 8))):
                return False
        return True


class MemTable:
    \"\"\"In-memory active partition holding ordered writes up to {memtable_flush_threshold} records.\"\"\"

    def __init__(self, max_records: int = {memtable_flush_threshold}) -> None:
        self.max_records = max_records
        self._data: Dict[str, Optional[str]] = {{}}

    def put(self, key: str, value: str) -> bool:
        self._data[key] = value
        return len(self._data) >= self.max_records

    def delete(self, key: str) -> bool:
        # Tombstone sentinel
        self._data[key] = None
        return len(self._data) >= self.max_records

    def get(self, key: str) -> Tuple[bool, Optional[str]]:
        if key in self._data:
            return True, self._data[key]
        return False, None

    def items(self) -> List[Tuple[str, Optional[str]]]:
        return sorted(self._data.items(), key=lambda kv: kv[0])

    def clear(self) -> None:
        self._data.clear()


class SSTable:
    \"\"\"Immutable sorted file on disk with sparse index and Bloom filter.\"\"\"

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.sparse_index: List[Tuple[str, int]] = []  # (Key, Byte Offset)
        self.bloom_filter = BloomFilter(capacity=10000)

    @classmethod
    def write_from_memtable(cls, filepath: Path, memtable: MemTable) -> SSTable:
        sstable = cls(filepath)
        with open(filepath, "wb") as fp:
            offset = 0
            for idx, (k, v) in enumerate(memtable.items()):
                val_bytes = v.encode("utf-8") if v is not None else b""
                is_tombstone = 1 if v is None else 0
                record = struct.pack(">II? ", len(k), len(val_bytes), is_tombstone) + k.encode("utf-8") + val_bytes
                
                # Sample sparse index every 64 records
                if idx % 64 == 0:
                    sstable.sparse_index.append((k, offset))

                sstable.bloom_filter.add(k)
                fp.write(record)
                offset += len(record)

        return sstable

    def get(self, target_key: str) -> Tuple[bool, Optional[str]]:
        if not self.bloom_filter.might_contain(target_key):
            return False, None

        if not self.sparse_index:
            return False, None

        # Binary search on sparse index to find enclosing offset bounds
        keys = [k for k, _ in self.sparse_index]
        idx = bisect.bisect_right(keys, target_key) - 1
        if idx < 0:
            idx = 0

        start_offset = self.sparse_index[idx][1]
        with open(self.filepath, "rb") as fp:
            fp.seek(start_offset)
            while True:
                header = fp.read(9)
                if len(header) < 9:
                    break
                k_len, v_len, is_tombstone = struct.unpack(">II? ", header)
                cur_key = fp.read(k_len).decode("utf-8")
                val_bytes = fp.read(v_len)

                if cur_key == target_key:
                    if is_tombstone:
                        return True, None
                    return True, val_bytes.decode("utf-8")

                if cur_key > target_key:
                    break

        return False, None
```
"""
    meta = {
        "domain": "systems_architecture",
        "language": "python",
        "component": "lsm_tree_storage_engine",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


# ==============================================================================
# Tier 3: Advanced Data Structures & Algorithmic Rigor
# ==============================================================================

def gen_red_black_tree_proof(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    text = """/* Module: red_black_tree.c
 * Self-Balancing Red-Black Binary Search Tree with Formal Invariant Validation.
 *
 * Invariants Guaranteed:
 * 1. Every node is either RED or BLACK.
 * 2. The root node is always BLACK.
 * 3. Every leaf (NIL sentinel) is BLACK.
 * 4. If a node is RED, both its children are BLACK (No two consecutive RED nodes).
 * 5. For each node, every path from the node to descendant leaves contains the same number of BLACK nodes (Black-Height).
 *
 * Time Complexity:
 * - Search: O(log N)
 * - Insert: O(log N) with at most 2 tree rotations
 * - Delete: O(log N) with at most 3 tree rotations
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

typedef enum {
    NODE_RED,
    NODE_BLACK
} NodeColor;

typedef struct RBNode {
    int key;
    void* value;
    NodeColor color;
    struct RBNode* parent;
    struct RBNode* left;
    struct RBNode* right;
} RBNode;

typedef struct RBTree {
    RBNode* root;
    RBNode* nil; // Sentinel node
    size_t size;
} RBTree;

static RBNode* create_node(RBTree* tree, int key, void* value) {
    RBNode* node = (RBNode*)malloc(sizeof(RBNode));
    node->key = key;
    node->value = value;
    node->color = NODE_RED;
    node->parent = tree->nil;
    node->left = tree->nil;
    node->right = tree->nil;
    return node;
}

RBTree* rbtree_create(void) {
    RBTree* tree = (RBTree*)malloc(sizeof(RBTree));
    tree->nil = (RBNode*)malloc(sizeof(RBNode));
    tree->nil->color = NODE_BLACK;
    tree->nil->parent = NULL;
    tree->nil->left = NULL;
    tree->nil->right = NULL;
    tree->root = tree->nil;
    tree->size = 0;
    return tree;
}

static void left_rotate(RBTree* tree, RBNode* x) {
    RBNode* y = x->right;
    x->right = y->left;

    if (y->left != tree->nil) {
        y->left->parent = x;
    }

    y->parent = x->parent;

    if (x->parent == tree->nil) {
        tree->root = y;
    } else if (x == x->parent->left) {
        x->parent->left = y;
    } else {
        x->parent->right = y;
    }

    y->left = x;
    x->parent = y;
}

static void right_rotate(RBTree* tree, RBNode* y) {
    RBNode* x = y->left;
    y->left = x->right;

    if (x->right != tree->nil) {
        x->right->parent = y;
    }

    x->parent = y->parent;

    if (y->parent == tree->nil) {
        tree->root = x;
    } else if (y == y->parent->right) {
        y->parent->right = x;
    } else {
        y->parent->left = x;
    }

    x->right = y;
    y->parent = x;
}

static void rbtree_insert_fixup(RBTree* tree, RBNode* z) {
    while (z->parent->color == NODE_RED) {
        if (z->parent == z->parent->parent->left) {
            RBNode* uncle = z->parent->parent->right;
            if (uncle->color == NODE_RED) {
                // Case 1: Uncle is RED -> recolor
                z->parent->color = NODE_BLACK;
                uncle->color = NODE_BLACK;
                z->parent->parent->color = NODE_RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->right) {
                    // Case 2: Uncle is BLACK and z is right child -> left rotate
                    z = z->parent;
                    left_rotate(tree, z);
                }
                // Case 3: Uncle is BLACK and z is left child -> right rotate and recolor
                z->parent->color = NODE_BLACK;
                z->parent->parent->color = NODE_RED;
                right_rotate(tree, z->parent->parent);
            }
        } else {
            // Symmetrical right-side cases
            RBNode* uncle = z->parent->parent->left;
            if (uncle->color == NODE_RED) {
                z->parent->color = NODE_BLACK;
                uncle->color = NODE_BLACK;
                z->parent->parent->color = NODE_RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->left) {
                    z = z->parent;
                    right_rotate(tree, z);
                }
                z->parent->color = NODE_BLACK;
                z->parent->parent->color = NODE_RED;
                left_rotate(tree, z->parent->parent);
            }
        }
    }
    tree->root->color = NODE_BLACK;
}

void rbtree_insert(RBTree* tree, int key, void* value) {
    RBNode* z = create_node(tree, key, value);
    RBNode* y = tree->nil;
    RBNode* x = tree->root;

    while (x != tree->nil) {
        y = x;
        if (z->key < x->key) {
            x = x->left;
        } else {
            x = x->right;
        }
    }

    z->parent = y;
    if (y == tree->nil) {
        tree->root = z;
    } else if (z->key < y->key) {
        y->left = z;
    } else {
        y->right = z;
    }

    tree->size++;
    rbtree_insert_fixup(tree, z);
}
"""
    meta = {
        "domain": "algorithms",
        "language": "c",
        "component": "red_black_tree_balancing",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_segment_tree_lazy_propagation(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    text = """\"\"\"Segment Tree with Lazy Propagation for Range Updates and Range Minimum/Sum Queries.

Complexity:
- Build: O(N)
- Range Update: O(log N)
- Range Query: O(log N)
- Space: O(4N)
\"\"\"

from __future__ import annotations

from typing import Callable, List, Optional


class LazySegmentTree:
    \"\"\"General-purpose lazy segment tree supporting range additions and range minimum queries.\"\"\"

    def __init__(self, data: List[int]) -> None:
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if self.n > 0:
            self._build(data, 1, 0, self.n - 1)

    def _build(self, data: List[int], node: int, start: int, end: int) -> None:
        if start == end:
            self.tree[node] = data[start]
            return

        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        self._build(data, left_child, start, mid)
        self._build(data, right_child, mid + 1, end)
        self.tree[node] = min(self.tree[left_child], self.tree[right_child])

    def _push_down(self, node: int, start: int, end: int) -> None:
        if self.lazy[node] != 0:
            val = self.lazy[node]
            left_child = 2 * node
            right_child = 2 * node + 1

            self.tree[left_child] += val
            self.lazy[left_child] += val

            self.tree[right_child] += val
            self.lazy[right_child] += val

            self.lazy[node] = 0

    def update_range(self, l: int, r: int, val: int) -> None:
        \"\"\"Adds val to all elements in range [l, r] inclusive in O(log N) time.\"\"\"
        self._update(1, 0, self.n - 1, l, r, val)

    def _update(self, node: int, start: int, end: int, l: int, r: int, val: int) -> None:
        if r < start or end < l:
            return

        if l <= start and end <= r:
            self.tree[node] += val
            self.lazy[node] += val
            return

        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._update(2 * node, start, mid, l, r, val)
        self._update(2 * node + 1, mid + 1, end, l, r, val)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query_range(self, l: int, r: int) -> int:
        \"\"\"Returns the minimum element in range [l, r] inclusive in O(log N) time.\"\"\"
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        if r < start or end < l:
            return float("inf")  # Identity element for min

        if l <= start and end <= r:
            return self.tree[node]

        self._push_down(node, start, end)
        mid = (start + end) // 2
        left_res = self._query(2 * node, start, mid, l, r)
        right_res = self._query(2 * node + 1, mid + 1, end, l, r)
        return min(left_res, right_res)
"""
    meta = {
        "domain": "algorithms",
        "language": "python",
        "component": "lazy_segment_tree",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


# ==============================================================================
# Tier 4: Systems Engineering RFCs & Architecture Decision Records (ADRs)
# ==============================================================================

def gen_systems_rfc_spec(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    rfc_id = rng.randint(1001, 8999)
    service_name = rng.choice(["GlobalEventMesh", "EdgeKeyVault", "DistributedTracingIngress", "ZeroTrustServiceMesh"])
    text = f"""# RFC-{rfc_id}: Architectural Specification for {service_name}
**Status**: APPROVED | **Author**: Systems Architecture Working Group | **Date**: 2026-03-15
**Target Tier**: Infrastructure Core | **Classification**: High Reliability Tier-0

## 1. Executive Summary & Problem Context
The current inter-service synchronization layer experiences unbounded tail latency under asymmetric network partitions. When downstream database replicas experience transient I/O saturations, head-of-line blocking propagates upstream, resulting in cascading thread exhaustion across core microservices.

This RFC defines the architectural blueprint for **{service_name}**, replacing synchronized blocking RPCs with an asynchronous, non-blocking actor mesh utilizing bounded ring-buffers and edge-evaluated circuit breakers.

## 2. System Design & Architectural Invariants

### 2.1 Component Topology
```text
+-------------------+      gRPC Multiplex      +--------------------------+
|  Ingress Gateway  | ======================> |  {service_name} Dispatcher  |
+-------------------+                         +--------------------------+
                                                          |
                      +-----------------------------------+-----------------------------------+
                      |                                   |                                   |
                      v                                   v                                   v
          +-----------------------+           +-----------------------+           +-----------------------+
          | Worker Shard #1 (CPU) |           | Worker Shard #2 (CPU) |           | Worker Shard #3 (CPU) |
          +-----------------------+           +-----------------------+           +-----------------------+
                      |                                   |                                   |
                      +-------------------> [ Ring Buffer ] <---------------------------------+
                                                  |
                                                  v
                                      +-----------------------+
                                      |  Storage Engine (LSM) |
                                      +-----------------------+
```

### 2.2 Formal Invariants
1. **Bounded Latency Guarantee**: P99.9 egress latency shall not exceed 15ms under steady-state load (50,000 req/sec per cluster).
2. **Crash-Consistency (WAL Guarantee)**: Every committed state transition must be flushed to the append-only write log (`O_DIRECT` or `fdatasync`) prior to emitting acknowledgment to client callers.
3. **Partition Tolerance**: During a split-brain condition, network nodes isolated from the quorum partition must transition to read-only degraded mode within 2 heartbeat timeouts.

## 3. Failure Modes & Threat Model

| Failure Scenario | Detection Mechanism | Mitigation & Recovery |
| :--- | :--- | :--- |
| Primary Leader Node Crash | Missed 3 consecutive heartbeats (300ms) | Automatic election trigger via Raft majority quorum |
| Downstream Buffer Saturation | Ingress queue depth exceeds 85% capacity | Exponential backpressure via HTTP 429 & Retry-After |
| Malicious Replay Attack | Monotonic sequence validation per token | Reject requests whose counter <= highest observed |

## 4. Alternatives Considered
- **Synchronous Two-Phase Commit (2PC)**: Rejected due to blocking coordinator vulnerability. If the coordinator fails during the prepare phase, participating cohort nodes remain locked indefinitely.
- **Pure Eventual Consistency (Gossip-Only)**: Rejected due to strict financial audit compliance requiring linearizable reads for ledger balance operations.
"""
    meta = {
        "domain": "rfc_specifications",
        "language": "markdown",
        "component": "systems_architecture_rfc",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_c_epoll_event_loop(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    max_events = rng.choice([64, 128, 256, 512])
    port = rng.choice([8080, 8443, 9000, 9200])
    text = f"""/* Module: epoll_tcp_server.c
 * Ultra-high concurrency non-blocking TCP echo server utilizing Linux epoll(7).
 * Configured with Edge-Triggered mode (EPOLLET) and non-blocking sockets.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define MAX_EVENTS {max_events}
#define BUFFER_SIZE 4096

static int set_nonblocking(int fd) {{
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}}

int main(int argc, char *argv[]) {{
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {{
        perror("socket");
        exit(EXIT_FAILURE);
    }}

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons({port});

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {{
        perror("bind");
        close(listen_fd);
        exit(EXIT_FAILURE);
    }}

    if (set_nonblocking(listen_fd) < 0) {{
        perror("set_nonblocking listen_fd");
        close(listen_fd);
        exit(EXIT_FAILURE);
    }}

    if (listen(listen_fd, SOMAXCONN) < 0) {{
        perror("listen");
        close(listen_fd);
        exit(EXIT_FAILURE);
    }}

    int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd < 0) {{
        perror("epoll_create1");
        close(listen_fd);
        exit(EXIT_FAILURE);
    }}

    struct epoll_event ev;
    ev.events = EPOLLIN | EPOLLET; // Edge-triggered
    ev.data.fd = listen_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_fd, &ev) < 0) {{
        perror("epoll_ctl listen_fd");
        close(listen_fd);
        close(epoll_fd);
        exit(EXIT_FAILURE);
    }}

    struct epoll_event events[MAX_EVENTS];
    char buf[BUFFER_SIZE];

    printf("Server listening on port {port} with epoll edge-triggered I/O\\n");

    while (1) {{
        int n_fds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        if (n_fds < 0) {{
            if (errno == EINTR) continue;
            perror("epoll_wait");
            break;
        }}

        for (int i = 0; i < n_fds; ++i) {{
            if (events[i].data.fd == listen_fd) {{
                // Drain all pending incoming connections under EPOLLET
                while (1) {{
                    struct sockaddr_in client_addr;
                    socklen_t client_len = sizeof(client_addr);
                    int client_fd = accept4(listen_fd, (struct sockaddr *)&client_addr, &client_len, SOCK_NONBLOCK | SOCK_CLOEXEC);
                    if (client_fd < 0) {{
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {{
                            break; // All connections drained
                        }}
                        perror("accept4");
                        break;
                    }}

                    struct epoll_event client_ev;
                    client_ev.events = EPOLLIN | EPOLLET | EPOLLRDHUP;
                    client_ev.data.fd = client_fd;
                    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &client_ev);
                }}
            }} else {{
                int fd = events[i].data.fd;
                if (events[i].events & (EPOLLRDHUP | EPOLLHUP | EPOLLERR)) {{
                    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
                    close(fd);
                    continue;
                }}

                if (events[i].events & EPOLLIN) {{
                    // Drain read buffer completely under EPOLLET
                    while (1) {{
                        ssize_t count = read(fd, buf, sizeof(buf));
                        if (count < 0) {{
                            if (errno == EAGAIN || errno == EWOULDBLOCK) {{
                                break; // Read complete for this notification
                            }}
                            perror("read");
                            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
                            close(fd);
                            break;
                        }} else if (count == 0) {{
                            // Remote closed connection
                            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
                            close(fd);
                            break;
                        }}

                        // Echo back data
                        write(fd, buf, count);
                    }}
                }}
            }}
        }}
    }}

    close(listen_fd);
    close(epoll_fd);
    return 0;
}}
"""
    meta = {
        "domain": "systems_networking",
        "language": "c",
        "component": "epoll_event_loop",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_python_consistent_hashing(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    replicas = rng.choice([100, 150, 200, 256])
    text = f'''"""Consistent Hashing Ring Implementation with Virtual Nodes and Ketama-style Partitioning.

Guarantees:
1. Minimal key migration: When a node is added or removed, only K/N keys are remapped on average.
2. Uniform key distribution across asymmetric heterogeneous physical nodes via {replicas} virtual nodes.
3. O(log V) lookup time via binary search over sorted integer hash ring.
"""

from __future__ import annotations

import bisect
import hashlib
from typing import Dict, List, Optional, Set


class ConsistentHashRing:
    """Thread-safe consistent hash ring maintaining {replicas} virtual replicas per node."""

    def __init__(self, virtual_replicas: int = {replicas}) -> None:
        self.virtual_replicas = virtual_replicas
        self.ring: List[int] = []
        self.ring_map: Dict[int, str] = {{}}  # hash_value -> physical_node_id
        self.nodes: Set[str] = set()

    def _hash(self, key: str) -> int:
        """Calculates 32-bit unsigned integer hash using MD5 digest."""
        digest = hashlib.md5(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big")

    def add_node(self, node_id: str) -> None:
        """Registers a physical node and creates virtual replicas around the hash ring."""
        if node_id in self.nodes:
            return

        self.nodes.add(node_id)
        for i in range(self.virtual_replicas):
            vnode_key = f"{{node_id}}#vnode_{{i}}"
            h = self._hash(vnode_key)
            bisect.insort(self.ring, h)
            self.ring_map[h] = node_id

    def remove_node(self, node_id: str) -> None:
        """Removes physical node and all associated virtual replicas."""
        if node_id not in self.nodes:
            return

        self.nodes.remove(node_id)
        for i in range(self.virtual_replicas):
            vnode_key = f"{{node_id}}#vnode_{{i}}"
            h = self._hash(vnode_key)
            idx = bisect.bisect_left(self.ring, h)
            if idx < len(self.ring) and self.ring[idx] == h:
                del self.ring[idx]
            self.ring_map.pop(h, None)

    def get_node(self, key: str) -> Optional[str]:
        """Locates the physical node responsible for a key via clockwise successor search."""
        if not self.ring:
            return None

        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h)

        # Wrap around to beginning of ring if past the last virtual node
        if idx == len(self.ring):
            idx = 0

        target_hash = self.ring[idx]
        return self.ring_map[target_hash]

    def get_preference_list(self, key: str, count: int) -> List[str]:
        """Returns `count` distinct physical nodes responsible for replicas (e.g. for Dynamo N-ary replication)."""
        if not self.ring or count <= 0:
            return []

        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h)
        selected_nodes: List[str] = []
        visited_hashes = 0

        while len(selected_nodes) < min(count, len(self.nodes)) and visited_hashes < len(self.ring):
            target_hash = self.ring[idx % len(self.ring)]
            node = self.ring_map[target_hash]
            if node not in selected_nodes:
                selected_nodes.append(node)
            idx += 1
            visited_hashes += 1

        return selected_nodes
'''
    meta = {
        "domain": "distributed_systems",
        "language": "python",
        "component": "consistent_hashing_ring",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_rust_arena_allocator(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    chunk_size = rng.choice([4096, 8192, 16384, 65536])
    text = f"""//! Module: arena_allocator.rs
//! High-performance monotonic bump allocator (Arena) with strict memory alignment.
//!
//! # Invariants:
//! 1. Individual allocations within an arena cannot be freed independently (zero per-allocation overhead).
//! 2. All allocations are properly aligned according to `std::mem::align_of::<T>()`.
//! 3. All allocated memory is reclaimed in bulk in O(1) time when the arena is dropped or reset.

use std::alloc::{{alloc, dealloc, Layout}};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ptr::NonNull;

pub struct Chunk {{
    ptr: NonNull<u8>,
    layout: Layout,
    offset: usize,
    capacity: usize,
    next: Option<Box<Chunk>>,
}}

impl Chunk {{
    fn new(capacity: usize) -> Self {{
        let layout = Layout::from_size_align(capacity, 16).expect("Valid alignment");
        let ptr = unsafe {{
            let raw = alloc(layout);
            NonNull::new(raw).expect("Memory allocation failure")
        }};

        Self {{
            ptr,
            layout,
            offset: 0,
            capacity,
            next: None,
        }}
    }}
}}

impl Drop for Chunk {{
    fn drop(&mut self) {{
        unsafe {{
            dealloc(self.ptr.as_ptr(), self.layout);
        }}
    }}
}}

pub struct Arena {{
    current_chunk: UnsafeCell<Chunk>,
    default_capacity: usize,
}}

impl Arena {{
    pub fn new() -> Self {{
        Self::with_capacity({chunk_size})
    }}

    pub fn with_capacity(capacity: usize) -> Self {{
        Self {{
            current_chunk: UnsafeCell::new(Chunk::new(capacity)),
            default_capacity: capacity,
        }}
    }}

    pub fn alloc<T>(&self, val: T) -> &mut T {{
        let layout = Layout::new::<T>();
        let ptr = self.alloc_layout(layout);
        unsafe {{
            let typed_ptr = ptr.as_ptr() as *mut T;
            std::ptr::write(typed_ptr, val);
            &mut *typed_ptr
        }}
    }}

    pub fn alloc_layout(&self, layout: Layout) -> NonNull<u8> {{
        let chunk = unsafe {{ &mut *self.current_chunk.get() }};

        // Align current offset to required alignment
        let align = layout.align();
        let aligned_offset = (chunk.offset + (align - 1)) & !(align - 1);

        if aligned_offset + layout.size() <= chunk.capacity {{
            let alloc_ptr = unsafe {{ chunk.ptr.as_ptr().add(aligned_offset) }};
            chunk.offset = aligned_offset + layout.size();
            NonNull::new(alloc_ptr).unwrap()
        }} else {{
            // Current chunk is exhausted; allocate a new chunk
            let next_capacity = std::cmp::max(self.default_capacity, layout.size() * 2);
            let mut new_chunk = Chunk::new(next_capacity);

            let alloc_ptr = new_chunk.ptr.as_ptr();
            new_chunk.offset = layout.size();

            // Link old chunk into list
            let old_chunk = std::mem::replace(chunk, new_chunk);
            chunk.next = Some(Box::new(old_chunk));

            NonNull::new(alloc_ptr).unwrap()
        }}
    }}

    pub fn reset(&mut self) {{
        let chunk = self.current_chunk.get_mut();
        chunk.offset = 0;
        chunk.next = None;
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_monotonic_arena_allocation() {{
        let arena = Arena::with_capacity({chunk_size});
        let val1 = arena.alloc(42u64);
        let val2 = arena.alloc(1337u32);
        assert_eq!(*val1, 42);
        assert_eq!(*val2, 1337);
    }}
}}
"""
    meta = {
        "domain": "systems_memory",
        "language": "rust",
        "component": "bump_arena_allocator",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


def gen_k8s_controller_reconciler(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    resource_name = rng.choice(["DatabaseCluster", "ModelServingDeployment", "DistributedCacheInstance"])
    text = f"""// Package controller implements a Kubernetes Custom Resource Reconciler pattern in Go.
// Uses client-go informers, workqueue, and exponential backoff retry loops.

package controller

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

type ReconcileResult struct {{
	Requeue      bool
	RequeueAfter time.Duration
}}

type {resource_name}Reconciler struct {{
	kubeClient kubernetes.Interface
	queue      workqueue.RateLimitingInterface
	indexer    cache.Indexer
	informer   cache.Controller
}}

func New{resource_name}Reconciler(client kubernetes.Interface, queue workqueue.RateLimitingInterface, indexer cache.Indexer, informer cache.Controller) *{resource_name}Reconciler {{
	return &{resource_name}Reconciler{{
		kubeClient: client,
		queue:      queue,
		indexer:    indexer,
		informer:   informer,
	}}
}}

func (r *{resource_name}Reconciler) Reconcile(ctx context.Context, key string) (ReconcileResult, error) {{
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {{
		klog.Errorf("Invalid resource key: %s, err: %v", key, err)
		return ReconcileResult{{}}, nil
	}}

	obj, exists, err := r.indexer.GetByKey(key)
	if err != nil {{
		return ReconcileResult{{Requeue: true, RequeueAfter: 5 * time.Second}}, err
	}}

	if !exists {{
		klog.Infof("Resource %s has been deleted. Triggering cleanup finalizer", key)
		return ReconcileResult{{}}, nil
	}}

	_ = obj // Inspect resource spec and sync child statefulsets/services
	klog.V(2).Infof("Reconciling %s/%s", namespace, name)

	// Idempotent state convergence:
	// Verify underlying StatefulSet exists, create or update matching spec.
	return ReconcileResult{{Requeue: false}}, nil
}}

func (r *{resource_name}Reconciler) RunWorker(ctx context.Context) {{
	for r.processNextItem(ctx) {{
	}}
}}

func (r *{resource_name}Reconciler) processNextItem(ctx context.Context) bool {{
	key, shutdown := r.queue.Get()
	if shutdown {{
		return false
	}}
	defer r.queue.Done(key)

	res, err := r.Reconcile(ctx, key.(string))
	if err != nil {{
		if r.queue.NumRequeues(key) < 5 {{
			r.queue.AddRateLimited(key)
			return true
		}}
		r.queue.Forget(key)
		klog.Errorf("Dropping %s out of the queue after max retries: %v", key, err)
		return true
	}}

	if res.RequeueAfter > 0 {{
		r.queue.AddAfter(key, res.RequeueAfter)
	}} else if res.Requeue {{
		r.queue.AddRateLimited(key)
	}} else {{
		r.queue.Forget(key)
	}}

	return true
}}
"""
    meta = {
        "domain": "cloud_infrastructure",
        "language": "go",
        "component": "k8s_controller_reconciler",
        "tokens": estimate_tokens(text)
    }
    return text.strip(), meta


# ==============================================================================
# Master Generator Pipeline
# ==============================================================================

GENERATOR_REGISTRY: List[Callable[[random.Random], Tuple[str, Dict[str, Any]]]] = [
    gen_rust_tokio_actor_system,
    gen_go_worker_pool_pipeline,
    gen_cpp_lockfree_ring_buffer,
    gen_python_async_connection_pool,
    gen_typescript_state_machine,
    gen_advanced_sql_window_cte,
    gen_distributed_raft_consensus,
    gen_lsm_tree_storage_engine,
    gen_red_black_tree_proof,
    gen_segment_tree_lazy_propagation,
    gen_systems_rfc_spec,
    gen_c_epoll_event_loop,
    gen_python_consistent_hashing,
    gen_rust_arena_allocator,
    gen_k8s_controller_reconciler,
]


def generate_base_code_corpus(
    output_dir: Path,
    count: int = 5000,
    seed: int = 42,
    max_mb: float = 28.0,
    prefix: str = "base_code_part"
) -> Dict[str, Any]:
    """Generate partitioned base code pretraining documents strictly partitioned under 28 MB."""
    rng = random.Random(seed)
    max_bytes = int(max_mb * 1024 * 1024)
    writer = ShardedCodeWriter(output_dir=output_dir, prefix=prefix, max_bytes=max_bytes)

    domain_counts: Dict[str, int] = {}
    lang_counts: Dict[str, int] = {}
    total_tokens_est = 0

    print(f"Generating {count:,} Code & Systems Architecture base pretraining documents into {output_dir}...")

    try:
        for i in range(count):
            gen_fn = rng.choice(GENERATOR_REGISTRY)
            text, meta = gen_fn(rng)

            writer.write_document(text=text, meta=meta)

            d = meta.get("domain", "code")
            lang = meta.get("language", "unknown")
            toks = meta.get("tokens", estimate_tokens(text))

            domain_counts[d] = domain_counts.get(d, 0) + 1
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            total_tokens_est += toks

            if (i + 1) % 1000 == 0 or (i + 1) == count:
                print(f"  Progress: {i + 1:,} / {count:,} records processed...")
    finally:
        writer.close()

    manifest = {
        "version": "base-code-v1.0",
        "total_records": writer.total_records,
        "total_bytes": writer.total_bytes,
        "total_mb": writer.total_bytes / (1024 * 1024),
        "estimated_tokens": total_tokens_est,
        "partitions_count": len(writer.written_files),
        "partition_files": [f.name for f in writer.written_files],
        "domain_distribution": domain_counts,
        "language_distribution": lang_counts,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\nCompleted! Generated {writer.total_records:,} records across {len(writer.written_files)} partitions.")
    print(f"Total Size: {manifest['total_mb']:.2f} MB (~{total_tokens_est / 1e6:.1f} M tokens)")
    print(f"Manifest written to: {manifest_path}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Code & Systems Architecture Base Pretraining Data")
    parser.add_argument("--output-dir", type=str, default=r"E:\AI_Projects\dataset\code_pretraining", help="Output directory")
    parser.add_argument("--count", type=int, default=15000, help="Number of documents to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-mb", type=float, default=28.0, help="Partition ceiling in MB")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    generate_base_code_corpus(
        output_dir=out_path,
        count=args.count,
        seed=args.seed,
        max_mb=args.max_mb,
    )


if __name__ == "__main__":
    main()
