---
description: Definitive guidelines for writing robust, maintainable, and backend-agnostic asynchronous Python code using AnyIO. Prioritize structured concurrency and native AnyIO primitives.
globs: **/*.py
---
# AnyIO Best Practices

AnyIO is the definitive choice for modern asynchronous Python development. It provides a unified, Trio-inspired API on top of `asyncio` or `Trio`, fixing critical design shortcomings of `asyncio` and enforcing structured concurrency. Adhere to these rules to build reliable, high-performance async applications.

## 1. Entry Point and Backend Selection

Always use `anyio.run()` as your application's entry point. Explicitly select your preferred backend and leverage backend-specific optimizations.

❌ **BAD**: Relying on default backend or `asyncio.run()`.
```python
import asyncio
from anyio import run, sleep

async def main():
    print("Running on default backend (asyncio) without uvloop.")
    await sleep(0.1)

asyncio.run(main) # Bypasses AnyIO's entry point
run(main) # Implicit backend, no optimizations
```

✅ **GOOD**: Explicit backend selection and performance enhancements.
```python
from anyio import run, sleep
import sniffio

async def main():
    backend = sniffio.current_async_library()
    print(f"Hello from {backend}!")
    await sleep(0.1)

# For production, prefer