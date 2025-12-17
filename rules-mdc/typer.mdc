---
description: Definitive guidelines for building robust, maintainable, and user-friendly command-line interfaces with Typer, emphasizing modern Python type hints and modular design.
globs: **/*.py
---
# typer Best Practices

Typer is the gold standard for building Python CLIs. It leverages type hints to generate powerful, intuitive interfaces. These rules ensure your Typer applications are consistently well-structured, type-safe, and easy to maintain.

## 1. Code Organization and Structure

Always treat your `Typer` instance as a first-class object. For any CLI beyond a single-file script, modularize your commands.

### ✅ GOOD: Modular CLI with dedicated `cli.py`

Create a main `cli.py` that imports and registers commands from other modules. This enables dependency injection, easier testing, and better scalability.

```python
# my_cli_app/commands/user.py
import typer
from typing_extensions import Annotated

def create_user(
    name: Annotated[str, typer.Argument(help="Name of the new user")],
    email: Annotated[str, typer.Option(help="Email for the new user")],
    admin: Annotated[bool, typer.Option("--admin", "-a", help="Grant admin privileges")] = False,
):
    """Creates a new user."""
    print(f"Creating user: {name} <{email}> (Admin: {admin})")

def delete_user(user_id: Annotated[int, typer.Argument(help="ID of the user to delete")]):
    """Deletes a user by ID."""
    print(f"Deleting user with ID: {user_id}")

# This is a Typer sub-application
user_app = typer.Typer(name="user", help="Manage users")
user_app.command("create")(create_user)
user_app.command("delete")(delete_user)

# my_cli_app/commands/project.py
import typer
from typing_extensions import Annotated

def list_projects(
    status: Annotated[str, typer.Option(help="Filter by project status")] = "active"
):
    """Lists projects."""
    print(f"Listing projects with status: {status}")

project_app = typer.Typer(name="project", help="Manage projects")
project_app.command("list")(list_projects)

# my_cli_app/cli.py
import typer
from my_cli_app.commands.user import user_app
from my_cli_app.commands.project import project_app
from typing_extensions import Annotated
from rich.console import Console

# Inject a Rich console for consistent output
console = Console()

def version_callback(print_version: bool):
    if print_version:
        console.print("[bold green]My CLI App[/] version [yellow]1.0.0[/]")
        raise typer.Exit()

app = typer.Typer(
    name="my-cli-app",
    help="A powerful CLI application.",
    no_args_is_help=True
)

@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show the application version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
):
    """
    Manage your application resources.
    """
    # This callback can be used to inject common dependencies or setup
    app.extra["console"] = console # Example of injecting console

app.add_typer(user_app)
app.add_typer(project_app)

if __name__ == "__main__":
    app()

```

### ❌ BAD: Monolithic `cli.py` or `typer.run()` for complex apps

Avoid a single, giant file for all commands or using `typer.run()` when you have multiple commands. This quickly becomes unmanageable.

```python
# main.py (BAD for multi-command apps)
import typer

app = typer.Typer()

@app.command()
def create_user(name: str, email: str, admin: bool = False):
    print(f"Creating user: {name} <{email}> (Admin: {admin})")

@app.command()
def delete_user(user_id: int):
    print(f"Deleting user with ID: {user_id}")

@app.command()
def list_projects(status: str = "active"):
    print(f"Listing projects with status: {status}")

if __name__ == "__main__":
    app()
```

## 2. Type Hints and `Annotated`

Always use explicit type hints. For Typer-specific metadata (help text, defaults, etc.), use `typing_extensions.Annotated`. This keeps your code clean and leverages Typer's full power.

### ✅ GOOD: `Annotated` for all Typer parameters

Clearly define types and Typer options/arguments. Use `...` for required options.

```python
from typing_extensions import Annotated
import typer

def process_data(
    input_file: Annotated[str, typer.Argument(help="Path to the input file")],
    output_dir: Annotated[str, typer.Option(help="Directory for output files", rich_help_panel="File Paths")],
    chunk_size: Annotated[int, typer.Option("--chunk-size", "-c", help="Processing chunk size")] = 1024,
    force: Annotated[bool, typer.Option("--force", "-f", help="Force overwrite existing files")] = False,
    config_path: Annotated[str, typer.Option(help="Path to configuration file")] = ..., # Required option
):
    """Processes data from an input file."""
    print(f"Processing '{input_file}' to '{output_dir}' with chunk size {chunk_size}.")
    if force:
        print("Forcing overwrite.")
    print(f"Using config from: {config_path}")
```

### ❌ BAD: Implicit types or `typer.Option()` without `Annotated`

This sacrifices clarity, editor support, and future-proofing.

```python
import typer

# BAD: No explicit type hint, Typer metadata not in Annotated
def process_data(input_file, output_dir=typer.Option(help="Output dir"), chunk_size: int = 1024):
    print("Processing...")

# BAD: Type hint but Typer metadata not in Annotated
def old_style_option(name: str = typer.Option("World")):
    print(f"Hello {name}")
```

## 3. Packaging and Entry Points

For installable CLIs, use `pyproject.toml` to define your project and `console_scripts` entry points. This makes your tool easily installable via `pip install .` and executable from the shell.

### ✅ GOOD: `pyproject.toml` with `console_scripts`

```toml
# pyproject.toml
[project]
name = "my-cli-app"
version = "1.0.0"
dependencies = [
    "typer[rich]>=0.12.3", # Always pin versions
    "pydantic>=2.0",
    "typing-extensions>=4.0",
]

[project.scripts]
my-cli = "my_cli_app.cli:app" # Points to the Typer app instance

# Or for a simple single-command app:
# my-simple-cli = "my_cli_app.cli:main" # Points to typer.run(main)
```

## 4. Rich Output and Pydantic Validation

Enhance user experience with `Rich` for styled output and robust validation with `Pydantic` models for complex configurations.

### ✅ GOOD: `Rich` for styled output, `Pydantic` for config

```python
# my_cli_app/config.py
from pydantic import BaseModel, Field
from typing import Optional

class AppSettings(BaseModel):
    log_level: str = Field("INFO", description="Logging level")
    api_key: Optional[str] = Field(None, description="API key for external service")
    timeout: int = Field(30, description="Request timeout in seconds")

# my_cli_app/cli.py (excerpt)
import typer
from rich.console import Console
from rich.panel import Panel
from my_cli_app.config import AppSettings
from typing_extensions import Annotated

console = Console()

@app.command()
def run_task(
    task_id: Annotated[str, typer.Argument(help="ID of the task to run")],
    settings_path: Annotated[Optional[str], typer.Option(help="Path to settings file")] = None,
):
    """Executes a specific task."""
    console.print(Panel(f"[bold blue]Running Task: {task_id}[/bold blue]", expand=False))

    settings = AppSettings()
    if settings_path:
        # Load settings from file (e.g., JSON, YAML)
        # For simplicity, let's assume it's loaded and validated
        console.print(f"Loaded settings from [green]{settings_path}[/green]")
        # Example: settings = AppSettings.model_validate_json(Path(settings_path).read_text())

    console.print(f"Log Level: [yellow]{settings.log_level}[/yellow]")
    console.print(f"Timeout: [yellow]{settings.timeout}[/yellow] seconds")
    if settings.api_key:
        console.print("[green]API Key configured.[/green]")
    else:
        console.print("[red]Warning: No API Key configured.[/red]")

    # Access console from injected extra
    console = app.extra.get("console", Console())
    console.log("Task execution complete.")
```

## 5. Argument Parsing and Validation

Understand the distinction between arguments and options. Leverage Typer's features for robust validation.

### ✅ GOOD: Clear arguments/options, Enums for choices, callbacks for complex logic

```python
from typing_extensions import Annotated
import typer
from enum import Enum

class Environment(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

def validate_port(value: int):
    if not 1024 <= value <= 65535:
        raise typer.BadParameter("Port must be between 1024 and 65535.")
    return value

@app.command()
def deploy(
    env: Annotated[Environment, typer.Argument(help="Deployment environment")],
    branch: Annotated[str, typer.Option(help="Git branch to deploy")] = "main",
    port: Annotated[int, typer.Option(help="Port for the service", callback=validate_port)] = 8000,
):
    """Deploys the application to a specified environment."""
    print(f"Deploying branch '{branch}' to {env.value} on port {port}.")
```

### ❌ BAD: Mutable default arguments, manual validation, unclear distinctions

Mutable defaults can lead to unexpected behavior. Manual validation bypasses Typer's error handling.

```python
# BAD: Mutable default argument
def add_item(items: list = []): # This list is shared across calls!
    items.append("new_item")
    print(items)

# BAD: Manual validation instead of callback
def start_server(port: int):
    if not 1024 <= port <= 65535:
        print("Error: Port must be between 1024 and 65535.")
        return # Typer doesn't know this is an error
    print(f"Starting server on port {port}")
```