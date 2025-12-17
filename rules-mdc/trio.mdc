---
description: This guide enforces best practices for using Trio, Python's structured concurrency library, ensuring robust, type-safe, and performant asynchronous applications.
globs: **/*.py
---
# trio Best Practices

Trio is the definitive choice for new async-first Python projects in 2025, prioritizing usability and correctness through structured concurrency. Adhere to these guidelines for maintainable, high-performance code.

## 1. Structured Concurrency with Nurseries

Always manage concurrent tasks within a `trio.open_nursery()`. This guarantees that all spawned tasks complete or are cancelled together, preventing hidden task leaks and simplifying error handling.

❌ **BAD: Fire-and-forget tasks**
```python
import trio

async def background_task():
    await trio.sleep(100) # This task will run indefinitely, unmanaged

async def main():
    # No nursery, no explicit management
    trio.lowlevel.spawn_system_task(background_task)
    print("Main task exiting, background task is orphaned.")

trio.run(main)
```

✅ **GOOD: Tasks within a nursery**
```python
import trio

async def managed_task(task_id: int):
    print(f"Task {task_id} started.")
    await trio.sleep(1)
    print(f"Task {task_id} finished.")

async def main():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(managed_task, 1)
        nursery.start_soon(managed_task, 2)
    print("All tasks in nursery completed.")

trio.run(main)
```

## 2. Explicit Cancellation and Timeouts

Trio's cancellation model is robust. Use `cancel_scope` for explicit control over task lifetimes and `trio.Cancelled` for graceful cleanup.

❌ **BAD: Ignoring cancellation or using raw `try...except`**
```python
import trio

async def long_running_operation():
    try:
        await trio.sleep(10)
        print("Operation completed.")
    except Exception: # Too broad, doesn't specifically handle cancellation
        print("Operation failed somehow.")

async def main():
    with trio.move_on_after(0.1) as cancel_scope:
        await long_running_operation()
    if cancel_scope.cancelled_caught:
        print("Operation was cancelled.")

trio.run(main)
```

✅ **GOOD: Handling `trio.Cancelled` for cleanup**
```python
import trio

async def long_running_operation():
    try:
        print("Starting long operation...")
        await trio.sleep(10)
        print("Operation completed.")
    except trio.Cancelled:
        print("Operation was cancelled, performing cleanup.")
        # Perform necessary cleanup here
        raise # Re-raise to propagate cancellation

async def main():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(long_running_operation)
        await trio.sleep(0.1)
        nursery.cancel_scope.cancel() # Explicitly cancel the nursery

trio.run(main)
```

## 3. I/O Hygiene: Never Block the Event Loop

Blocking calls (e.g., `time.sleep()`, synchronous file I/O, `requests.get()`) will starve the event loop and halt all other concurrent tasks. Use Trio-aware alternatives or offload to a thread.

❌ **BAD: Blocking call in an `async def` function**
```python
import time
import trio

async def fetch_data_blocking():
    print("Fetching data (blocking)...")
    time.sleep(2) # Blocks the entire event loop!
    print("Data fetched.")
    return {"data": "example"}

async def main():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(fetch_data_blocking)
        nursery.start_soon(fetch_data_blocking) # These will run sequentially

trio.run(main)
```

✅ **GOOD: Offloading blocking calls with `trio.to_thread.run_sync`**
```python
import time
import trio

def _blocking_fetch_data(): # This is a regular sync function
    print("Fetching data (blocking in thread)...")
    time.sleep(2)
    print("Data fetched in thread.")
    return {"data": "example"}

async def fetch_data_non_blocking():
    return await trio.to_thread.run_sync(_blocking_fetch_data)

async def main():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(fetch_data_non_blocking)
        nursery.start_soon(fetch_data_non_blocking) # These will run concurrently

trio.run(main)
```
For network I/O, use `trio.socket`, `trio-http-client`, or `httpx` (with `trio` backend).

## 4. Type Safety and Linting

Mandate type hints for all `async def` functions, variables, and return types. Use `mypy --strict` and `flake8-async` (or Ruff with `ASYNC` rules) to catch common async-related errors early.

❌ **BAD: Missing type hints, potential runtime errors**
```python
import trio

async def process_item(item): # 'item' has no type hint
    await trio.sleep(0.1)
    return item * 2

async def main():
    result = await process_item("text") # Will fail at runtime if 'item' is str
    print(result)

trio.run(main)
```

✅ **GOOD: Comprehensive type hints**
```python
import trio

async def process_item(item: int) -> int:
    await trio.sleep(0.1)
    return item * 2

async def main():
    result = await process_item(5) # Type checker will catch incorrect usage
    print(result)

trio.run(main)
```
Configure `ruff` to enable `ASYNC` rules (e.g., `select = ["E", "F", "ASYNC"]` in `pyproject.toml`).

## 5. Interoperability with AnyIO

When building libraries or applications that need to be framework-agnostic (e.g., compatible with FastAPI, which uses AnyIO), write your async code using AnyIO primitives. This keeps your codebase portable across Trio, asyncio, and Curio.

```python
# Using AnyIO for portable async code
import anyio

async def do_something_concurrently():
    async with anyio.create_task_group() as tg:
        tg.start_soon(anyio.sleep, 1)
        tg.start_soon(anyio.sleep, 2)
    print("AnyIO tasks completed.")

async def main():
    await do_something_concurrently()

# Run with Trio backend
anyio.run(main, backend="trio")
```

## 6. Testing Approaches

Leverage `trio.testing` for reliable and deterministic tests of your asynchronous code.

```python
import trio
import trio.testing
import pytest

async def worker(value: int) -> int:
    await trio.sleep(0.01)
    return value * 2

@pytest.mark.trio
async def test_worker_doubles_value():
    # Use trio.testing.wait_all_tasks_blocked() for deterministic execution
    async with trio.open_nursery() as nursery:
        send_channel, receive_channel = trio.open_memory_channel(0)
        nursery.start_soon(lambda: nursery.start_soon(worker, 5)) # Example task
        await trio.testing.wait_all_tasks_blocked() # Ensures tasks yield
        # Assertions about task state or results
        assert await worker(3) == 6
```