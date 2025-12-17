---
description: Definitive guidelines for writing robust, performant, and maintainable asynchronous Python code using the asyncio standard library.
globs: **/*.py
---
# asyncio Best Practices

`asyncio` is Python's standard library for writing concurrent, I/O-bound code using `async`/`await`. This guide provides opinionated, actionable rules to ensure your `asyncio` applications are structured, performant, and easy to maintain.

## 1. Structured Concurrency: Entry Points and Task Management

Always use `asyncio.run()` as the single top-level entry point for your asynchronous application. Manage concurrent operations explicitly with `asyncio.create_task()`, ensuring all tasks are awaited or handled.

### 1.1. Top-Level Entry Point

Use `asyncio.run()` once to start your main coroutine. Enable debug mode during development for critical warnings about un-awaited coroutines and resource leaks.

❌ BAD: Manually managing event loops or calling `loop.run_until_complete()`.
```python
import asyncio

async def main():
    print("Hello")

# Don't do this in application code; it's low-level and error-prone.
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
```

✅ GOOD: Use `asyncio.run()` with `debug=True` for development.
```python
import asyncio

async def main():
    print("Hello from main!")
    await asyncio.sleep(0.1)
    print("Goodbye from main!")

if __name__ == "__main__":
    asyncio.run(main(), debug=True) # Always enable debug in dev!
```

### 1.2. Launching Concurrent Tasks

Use `asyncio.create_task()` to schedule coroutines to run concurrently. *Always* store a reference to the `Task` object to prevent it from being garbage collected prematurely, which can lead to silent failures and unhandled exceptions.

❌ BAD: Calling a coroutine without `await` or `create_task()`.
```python
import asyncio

async def fetch_data(url: str):
    print(f"Fetching {url}...")
    await asyncio.sleep(1) # Simulate network I/O
    print(f"Finished fetching {url}")
    return f"Data from {url}"

async def main():
    # This coroutine will never run!
    fetch_data("http://example.com/api/data")
    print("Main finished without waiting for data.")

if __name__ == "__main__":
    asyncio.run(main(), debug=True)
# Output: RuntimeWarning: coroutine 'fetch_data' was never awaited
# Main finished without waiting for data.
```

✅ GOOD: Explicitly create and await tasks.
```python
import asyncio
from typing import Awaitable

async def fetch_data(url: str) -> str:
    print(f"Fetching {url}...")
    await asyncio.sleep(1)
    print(f"Finished fetching {url}")
    return f"Data from {url}"

async def main() -> None:
    task: Awaitable[str] = asyncio.create_task(fetch_data("http://example.com/api/data"))
    print("Main continues while data is fetched...")
    result: str = await task # Await the task to get its result
    print(f"Received: {result}")

if __name__ == "__main__":
    asyncio.run(main(), debug=True)
```

### 1.3. Grouping Tasks

For running multiple tasks concurrently and waiting for all of them, use `asyncio.gather()`. For more fine-grained control over completion conditions (e.g., first to complete), use `asyncio.wait()`.

❌ BAD: Awaiting tasks sequentially when they can run in parallel.
```python
import asyncio
import time

async def process_item(item_id: int):
    print(f"Processing item {item_id}...")
    await asyncio.sleep(1)
    print(f"Item {item_id} processed.")
    return f"Result {item_id}"

async def main():
    start = time.monotonic()
    # This will take ~3 seconds total
    await process_item(1)
    await process_item(2)
    await process_item(3)
    print(f"Total time: {time.monotonic() - start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main(), debug=True)
```

✅ GOOD: Use `asyncio.gather()` for concurrent execution.
```python
import asyncio
import time
from typing import List

async def process_item(item_id: int) -> str:
    print(f"Processing item {item_id}...")
    await asyncio.sleep(1)
    print(f"Item {item_id} processed.")
    return f"Result {item_id}"

async def main() -> None:
    start = time.monotonic()
    # This will take ~1 second total
    results: List[str] = await asyncio.gather(
        process_item(1),
        process_item(2),
        process_item(3)
    )
    print(f"All items processed. Results: {results}")
    print(f"Total time: {time.monotonic() - start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main(), debug=True)
```

## 2. Avoiding Blocking Operations

The `asyncio` event loop must *never* be blocked. Any CPU-bound work or synchronous I/O operations must be offloaded to an executor.

### 2.1. CPU-Bound Work

Use `loop.run_in_executor()` with a `ThreadPoolExecutor` (for CPU-bound work in a separate thread) or `ProcessPoolExecutor` (for true parallelism, avoiding GIL) to prevent blocking the event loop.

❌ BAD: Performing heavy computation directly in a coroutine.
```python
import asyncio
import time

async def cpu_intensive_task():
    print("Starting CPU-intensive task...")
    # Simulate heavy computation - this blocks the event loop!
    _ = sum(i * i for i in range(10**7))
    print("Finished CPU-intensive task.")
    return "CPU result"

async def main():
    start = time.monotonic()
    task = asyncio.create_task(cpu_intensive_task())
    await asyncio.sleep(0.1) # This sleep will be delayed by the CPU task
    print("Main continued (but was blocked).")
    result = await task
    end = time.monotonic()
    print(f"Total time: {end - start:.2f}s, Result: {result}")

if __name__ == "__main__":
    asyncio.run(main(), debug=True)
# Output shows sleep was blocked, total time > 1s
```

✅ GOOD: Offload CPU-bound work to `run_in_executor`.
```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

def blocking_cpu_task() -> str:
    print("Starting blocking CPU task in executor...")
    _ = sum(i * i for i in range(10**7))
    print("Finished blocking CPU task in executor.")
    return "CPU result"

async def main() -> None:
    start = time.monotonic()
    # Use default ThreadPoolExecutor (None) or a custom one
    result_future = asyncio.get_running_loop().run_in_executor(
        None, blocking_cpu_task
    )
    await asyncio.sleep(0.1) # This sleep will run concurrently
    print("Main continued