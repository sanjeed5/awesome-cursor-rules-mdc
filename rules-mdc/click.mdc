---
description: This guide outlines definitive best practices for building robust, maintainable, and user-friendly command-line interfaces (CLIs) using the `click` Python library, emphasizing modern patterns and common pitfalls.
globs: **/*.py
---
# click Best Practices

`click` is the undisputed standard for building Python CLIs. Follow these rules to ensure your tools are consistent, maintainable, and a joy to use.

## 1. Code Organization & Structure

For any non-trivial CLI, structure your project to promote modularity and testability. Always use a `src` layout.

### ✅ GOOD: Modular `src` Layout

Separate your main CLI entry point from individual commands. Use `click.Group` to organize subcommands into logical modules.

```python
# src/mycli/__init__.py
import click
from .commands.greet import greet_command
from .commands.config import config_group

@click.group()
@click.version_option(version="1.0.0")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """My Awesome CLI Tool."""
    ctx.ensure_object(dict) # Initialize context object
    # Any global setup or configuration can go here

cli.add_command(greet_command)
cli.add_command(config_group)

# src/mycli/commands/greet.py
import click

@click.command(name="greet")
@click.argument("name", type=str)
@click.option("--formal", is_flag=True, help="Use a formal greeting.")
def greet_command(name: str, formal: bool) -> None:
    """Greets the given NAME."""
    if formal:
        click.secho(f"Greetings, {name}!", fg="blue")
    else:
        click.echo(f"Hello, {name}!")

# src/mycli/commands/config.py
import click

@click.group(name="config")
def config_group() -> None:
    """Manage configuration settings."""
    pass

@config_group.command(name="set")
@click.argument("key", type=str)
@click.argument("value", type=str)
def config_set(key: str, value: str) -> None:
    """Set a configuration KEY to VALUE."""
    click.echo(f"Setting '{key}' to '{value}'")

@config_group.command(name="get")
@click.argument("key", type=str)
def config_get(key: str) -> None:
    """Get the value of a configuration KEY."""
    click.echo(f"Getting value for '{key}'")
```

### ❌ BAD: Monolithic `cli.py`

Avoid dumping all commands and logic into a single file. This quickly becomes unmanageable.

```python
# cli.py (BAD)
import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument("name")
def greet(name): # Logic for greet command
    click.echo(f"Hello {name}!")

@cli.command()
@click.argument("key")
@click.argument("value")
def config_set(key, value): # Logic for config set command
    click.echo(f"Setting {key} to {value}")
```

## 2. Argument Parsing: Arguments vs. Options

Follow `click`'s conventions:
*   **Arguments**: Positional, required values.
*   **Options**: Flag-based, optional settings.

### ✅ GOOD: Clear Distinction

```python
import click

@click.command()
@click.argument("input_file", type=click.Path(exists=True)) # Positional, required file
@click.option("--output", "-o", type=click.Path(), help="Output file path.") # Optional flag
@click.option("--force", is_flag=True, help="Overwrite existing output.") # Flag without value
def process(input_file: str, output: str | None, force: bool) -> None:
    """Processes INPUT_FILE with optional OUTPUT."""
    click.echo(f"Processing {input_file}...")
    if output:
        click.echo(f"Outputting to {output} (Force: {force})")
```

### ❌ BAD: Misusing Arguments for Optional Settings

Using positional arguments for optional values makes `help` output confusing and usage less intuitive.

```python
import click

@click.command()
@click.argument("input_file")
@click.argument("output_file", required=False) # BAD: Optional positional argument
def process_bad(input_file: str, output_file: str | None) -> None:
    """Processes INPUT_FILE. Can optionally specify OUTPUT_FILE."""
    # This makes the CLI harder to use and read.
    click.echo(f"Input: {input_file}, Output: {output_file}")
```

## 3. Output: `click.echo` and `click.secho`

**Always** use `click.echo` or `click.secho` instead of `print()`. This ensures consistent Unicode handling and proper styling across all platforms, especially Windows.

### ✅ GOOD: Platform-Agnostic, Styled Output

```python
import click

@click.command()
def log_messages() -> None:
    """Demonstrates various output types."""
    click.echo("This is a standard message.")
    click.secho("This is an important warning!", fg="yellow", err=True) # To stderr, colored
    click.secho("Success!", fg="green", bold=True)
    click.echo(f"Unicode support: こんにちは世界") # Works everywhere
```

### ❌ BAD: Inconsistent Output

`print()` lacks cross-platform Unicode support and styling capabilities.

```python
import click

@click.command()
def log_messages_bad() -> None:
    """Demonstrates bad output types."""
    print("This message might break on Windows terminals with unicode.") # BAD
    print("\033[91mThis is a red error. (Manual ANSI codes are fragile)\033[0m") # BAD
```

## 4. Type Hints

Mandatory. Use Python's type hints with `click`'s built-in types (e.g., `click.Path`, `click.File`) for robust validation and improved IDE support.

### ✅ GOOD: Fully Type-Hinted

```python
import click
from typing import TextIO

@click.command()
@click.argument("count", type=int)
@click.option("--name", type=str, default="World")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), help="Path to config file.")
@click.option("--output", type=click.File("w"), help="Output stream.")
def greet_typed(count: int, name: str, config: str | None, output: TextIO | None) -> None:
    """Greets NAME COUNT times."""
    for _ in range(count):
        (output or click.get_text_stream("stdout")).write(f"Hello, {name}!\n")
    if config:
        click.echo(f"Using config from: {config}")
```

### ❌ BAD: Untyped Parameters

Lack of type hints makes code harder to understand, validate, and refactor.

```python
import click

@click.command()
@click.argument("count") # BAD: No type hint
@click.option("--name", default="World") # BAD: No type hint
def greet_untyped(count, name):
    """Greets NAME COUNT times."""
    # Type errors will only appear at runtime
    for _ in range(int(count)):
        click.echo(f"Hello, {name}!")
```

## 5. Comprehensive Help Text & Examples

Good help text is crucial. Always provide a concise summary, detailed explanations for options, and practical usage examples.

### ✅ GOOD: Detailed Help with Examples

```python
import click

@click.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("destination", type=click.Path())
@click.option("--recursive", "-r", is_flag=True, help="Copy directories recursively.")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files without prompting.")
def copy(source: str, destination: str, recursive: bool, force: bool) -> None:
    """
    Copies files or directories from SOURCE to DESTINATION.

    Examples:
        \b
        # Copy a single file
        mycli copy file.txt backup/file.txt

        # Copy a directory recursively
        mycli copy -r my_dir backup/my_dir

        # Force overwrite a file
        mycli copy -f important.log /var/log/important.log
    """
    click.echo(f"Copying '{source}' to '{destination}' (Recursive: {recursive}, Force: {force})")
```

### ❌ BAD: Minimalist Help Text

Vague help text leaves users guessing.

```python
import click

@click.command()
@click.argument("source")
@click.argument("destination")
@click.option("--recursive", "-r", is_flag=True) # BAD: No help text
def copy_bad(source: str, destination: str, recursive: bool) -> None:
    """Copies things.""" # BAD: Vague summary
    click.echo(f"Copying {source} to {destination}")
```

## 6. Packaging with `pyproject.toml`

Distribute your CLI tools reliably using `pyproject.toml` for metadata and entry points. This is the modern Python packaging standard.

```toml
# pyproject.toml
[project]
name = "mycli"
version = "1.0.0"
description = "My Awesome CLI Tool"
requires-python = ">=3.10"
dependencies = [
    "click>=8.3.1",
]

[project.scripts]
mycli = "mycli:cli" # Maps 'mycli' command to 'src/mycli/__init__.py:cli'
```

Install with `pipx install .` for isolated execution.

## 7. Testing with `CliRunner`

Automated testing is non-negotiable. Use `click.testing.CliRunner` to simulate CLI invocations and assert outputs.

```python
# tests/test_cli.py
from click.testing import CliRunner
from mycli import cli # Assuming 'mycli' is installed or in PYTHONPATH

def test_greet_command() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["greet", "Alice"])
    assert result.exit_code == 0
    assert "Hello, Alice!" in result.output

def test_greet_formal_option() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["greet", "Bob", "--formal"])
    assert result.exit_code == 0
    assert "Greetings, Bob!" in result.output
    assert "blue" in result.output # Check for color codes if applicable
```