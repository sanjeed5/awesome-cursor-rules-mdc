---
description: Definitive guidelines for building robust, maintainable, and visually appealing Python CLI applications using the rich library.
globs: **/*.py
---
# rich Best Practices

`rich` transforms plain terminal output into engaging, readable interfaces. These rules ensure your `rich` applications are performant, testable, and follow modern Python best practices.

## 1. Code Organization: Centralize `Console` Management

Instantiate `Console` once per application run. Avoid creating new `Console` objects in every function or module, as this leads to inconsistent styling, performance overhead, and potential issues with interactive elements. Pass the `Console` instance explicitly or use a global singleton pattern for CLI applications.

❌ BAD: Repeated `Console` instantiation
```python
# module_a.py
from rich.console import Console
def do_something():
    console = Console() # New console every time
    console.print("Doing something...")

# module_b.py
from rich.console import Console
def do_another_thing():
    console = Console() # Another new console
    console.print("Doing another thing...")
```

✅ GOOD: Single `Console` instance
```python
# main.py
from rich.console import Console
from .module_a import do_something
from .module_b import do_another_thing

console = Console() # Instantiate once at the top-level

def main():
    console.print("[bold green]Starting application...[/bold green]")
    do_something(console)
    do_another_thing(console)
    console.print("[bold green]Application finished.[/bold green]")

if __name__ == "__main__":
    main()

# module_a.py
from rich.console import Console
def do_something(console: Console): # Pass console explicitly
    console.print("Doing something...")

# module_b.py
from rich.console import Console
def do_another_thing(console: Console): # Pass console explicitly
    console.print("Doing another thing...")
```

## 2. Output: Always Use `console.print()`

For any output intended to be styled or managed by `rich`, use `console.print()`. Mixing `print()` with `console.print()` can lead to unpredictable output order and styling issues, especially with interactive elements like progress bars or live displays.

❌ BAD: Mixing `print()` and `console.print()`
```python
from rich.console import Console
console = Console()
print("This line might appear out of order or unstyled.")
console.print("[red]This is styled.[/red]")
```

✅ GOOD: Consistent `console.print()` usage
```python
from rich.console import Console
console = Console()
console.print("This line is managed by rich.")
console.print("[red]This is styled.[/red]")
```

## 3. Interactive Elements: Use `with` Statements

For `rich.progress.Progress` and `rich.live.Live`, always use a `with` statement. This ensures proper resource management, graceful shutdown, and correct rendering behavior, even if errors occur. Failing to use `with` can leave artifacts in the terminal or prevent proper cleanup.

❌ BAD: Manual start/stop (prone to errors)
```python
from rich.progress import Progress
progress = Progress()
progress.start()
# ... do work ...
progress.stop() # Easy to forget or miss on error
```

✅ GOOD: `with` statement for automatic management
```python
from rich.progress import Progress
from rich.console import Console
import time

console = Console()
with Progress(console=console) as progress:
    task = progress.add_task("[green]Processing...", total=100)
    for _ in range(100):
        time.sleep(0.01)
        progress.update(task, advance=1)
```

## 4. Logging: Integrate `RichHandler`

Leverage `rich.logging.RichHandler` for beautiful, readable, and structured logs directly in your terminal. Configure it once at application startup to enhance all standard Python logging output.

```python
import logging
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_level=True, show_path=False)]
)
log = logging.getLogger("rich")

def process_data():
    log.info("Starting data processing...")
    # ... some work ...
    log.warning("Potential issue detected during processing.")
    # ... more work ...
    log.error("Failed to save results!")

if __name__ == "__main__":
    process_data()
```

## 5. Type Hints: Embrace `rich` Types

`rich` provides excellent type annotations. Use them in your function signatures and variable declarations to improve code readability, enable static analysis with Mypy, and catch type-related errors early.

❌ BAD: Untyped `rich` usage
```python
def display_table(data):
    from rich.table import Table
    table = Table("Name", "Value")
    for row in data:
        table.add_row(row[0], str(row[1]))
    # ...
```

✅ GOOD: Type-hinted `rich` usage
```python
from typing import List, Tuple, Any
from rich.console import Console
from rich.table import Table

def display_table(console: Console, data: List[Tuple[str, Any]]):
    table = Table("Name", "Value")
    for row_name, row_value in data:
        table.add_row(row_name, str(row_value))
    console.print(table)

if __name__ == "__main__":
    console = Console()
    sample_data = [("Item A", 100), ("Item B", "Hello")]
    display_table(console, sample_data)
```

## 6. Testing: Capture `rich` Output

When testing `rich` applications with `pytest`, capture `stdout` to assert the content and styling. For more granular control, direct `Console` output to a `StringIO` object, allowing you to inspect the raw `rich` markup.

```python
# my_cli_app.py
from rich.console import Console
def greet(console: Console, name: str):
    console.print(f"Hello, [bold blue]{name}[/bold blue]!")

# test_my_cli_app.py
from io import StringIO
from rich.console import Console
from my_cli_app import greet

def test_greet_output_stringio():
    # Direct console output to a StringIO object for precise capture
    output_buffer = StringIO()
    console = Console(file=output_buffer, force_terminal=True) # force_terminal for consistent output
    greet(console, "World")
    captured = output_buffer.getvalue()
    assert "Hello, World!" in captured
    assert "[bold blue]World[/bold blue]" in captured # Assert rich markup

def test_greet_output_capsys(capsys): # pytest fixture to capture stdout/stderr
    # Use default console which writes to stdout, then capture with capsys
    console = Console(force_terminal=True)
    greet(console, "Test")
    captured_capsys = capsys.readouterr()
    assert "Hello, Test!" in captured_capsys.stdout
```

## 7. Virtual Environments & Packaging

Always use a virtual environment (`venv` or `pipenv`) for `rich` projects. Define `rich` as a dependency in `pyproject.toml` and use `pip install -e .` for editable installs during development. This ensures consistent environments and proper packaging for your CLI tool, making it easily distributable.

```toml
# pyproject.toml
[project]
name = "my-rich-cli"
version = "0.1.0"
dependencies = [
    "rich>=13.0.0",
    # other dependencies
]

[project.scripts]
my-cli = "my_rich_cli.main:main" # Define entry point for your CLI

# Example: my_rich_cli/main.py
from rich.console import Console
def main():
    console = Console()
    console.print("[green]Welcome to My Rich CLI![/green]")

# To set up and run your CLI:
# 1. python -m venv .venv
# 2. source .venv/bin/activate  (or .venv\Scripts\activate on Windows)
# 3. pip install -e .
# 4. my-cli
```