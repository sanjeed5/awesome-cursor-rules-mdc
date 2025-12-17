---
description: This guide defines best practices for configuring Python packages using setuptools, emphasizing modern, declarative `pyproject.toml` workflows for robust and maintainable projects.
globs: **/*.py
---
# setuptools Best Practices

`setuptools` is the de-facto standard for Python packaging. This guide outlines the definitive, modern approach to configuring your Python projects, prioritizing declarative `pyproject.toml` files, automated versioning, and clear dependency management. Adhere to these rules for consistent, maintainable, and standards-compliant package distributions.

## 1. Code Organization and Structure

Always use the `src/` layout for your package source code. This clearly separates your package's importable code from project-level files (like `pyproject.toml`, `README.md`, `tests/`).

❌ BAD: Flat layout
```
my_package/
├── __init__.py
├── module.py
├── pyproject.toml
└── tests/
```

✅ GOOD: `src/` layout
```
my_package/
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── module.py
├── pyproject.toml
├── README.md
└── tests/
```

## 2. Declarative Configuration in `pyproject.toml`

Configure all project metadata and build settings declaratively in `pyproject.toml`. This aligns with PEP 517/518, enables isolated builds, and makes your package metadata machine-readable without executing code. Avoid `setup.py` for metadata.

**Always include `[build-system]` and `[project]` tables.**

```toml title="pyproject.toml"
[build-system]
requires = ["setuptools>=61", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "my-awesome-package"
dynamic = ["version"] # Managed by setuptools-scm
description = "A short description of my awesome package."
readme = "README.md"
requires-python = ">=3.8"
license = { file = "LICENSE" }
authors = [
    { name = "Your Name", email = "your.email@example.com" },
]
keywords = ["python", "utility", "example"]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[project.urls]
Homepage = "https://github.com/yourusername/my-awesome-package"
Documentation = "https://my-awesome-package.readthedocs.io/"
Repository = "https://github.com/yourusername/my-awesome-package"

[tool.setuptools]
package-dir = {"" = "src"} # Essential for src/ layout

[tool.setuptools_scm]
write_to = "src/my_package/_version.py" # Optional: writes version to a file
```

## 3. Version Management with `setuptools-scm`

Derive your package version directly from VCS (Git) tags using `setuptools-scm`. This eliminates manual version bumps and ensures your package version always reflects your repository state.

1.  **Declare `setuptools-scm`** in `build-system.requires`.
2.  **Set `dynamic = ["version"]`** in `[project]`.
3.  **Configure `[tool.setuptools_scm]`** (optional, but recommended for `write_to`).

❌ BAD: Manual version in `pyproject.toml` or `__init__.py`
```toml
# pyproject.toml
[project]
version = "0.1.0" # ❌ BAD: Requires manual updates
```

✅ GOOD: Dynamic versioning with `setuptools-scm`
```toml title="pyproject.toml"
[build-system]
requires = ["setuptools>=61", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "my-package"
dynamic = ["version"] # ✅ GOOD: Version derived from Git tags

[tool.setuptools_scm]
# Optional: write the version to a file for runtime access
write_to = "src/my_package/_version.py"
```

## 4. Dependency Declaration

Manage all dependencies declaratively in `pyproject.toml`.

*   **`dependencies`**: For runtime dependencies.
*   **`optional-dependencies`**: For optional dependencies (e.g., `dev`, `test`, `docs`).

❌ BAD: `dependency_links` or `requirements.txt` for runtime dependencies
```toml
# pyproject.toml
[project]
# ❌ BAD: dependency_links is deprecated.
# ❌ BAD: requirements.txt is for application environments, not libraries.
```

✅ GOOD: `dependencies` and `optional-dependencies`
```toml title="pyproject.toml"
[project]
dependencies = [ # ✅ GOOD: Runtime dependencies
    "requests~=2.30",
    "rich>=13.0.0",
]

[project.optional-dependencies] # ✅ GOOD: Optional dependencies
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "black",
    "isort",
    "mypy",
    "setuptools-scm[toml]>=8", # Include setuptools-scm for dev installs
]
docs = [
    "sphinx",
    "furo",
]
```

## 5. Entry Points for Scripts and Plugins

Expose console scripts, GUI entry points, or plugin APIs using the `[project.scripts]` and `[project.entry-points]` tables. This is the standard way to create discoverable extensions and command-line tools.

```toml title="pyproject.toml"
[project.scripts] # ✅ GOOD: Console scripts
my-cli = "my_package.cli:main"

[project.gui-scripts] # ✅ GOOD: GUI scripts
my-gui = "my_package.gui:start_app"

[project.entry-points."my_package.plugins"] # ✅ GOOD: Custom plugin entry points
plugin_a = "my_package.plugins:PluginA"
plugin_b = "my_package.plugins:PluginB"
```

## 6. Resource Files and Namespace Packages

Bundle non-code assets (e.g., templates, data files) cleanly using `package_data`. For larger projects, use native namespace package support (PEP 420).

```toml title="pyproject.toml"
[tool.setuptools.package-data]
"my_package" = ["data/*.json", "templates/*.html"] # ✅ GOOD: Include specific data files
```

For namespace packages, simply omit `__init__.py` from the parent directory and ensure `setuptools` is configured to find them (e.g., via `package-dir` for `src/` layout).

```
src/
└── my_namespace/
    ├── my_package_a/
    │   └── __init__.py
    └── my_package_b/
        └── __init__.py
```

## 7. Development Workflow

Always use a virtual environment. For local development, install your package in "editable" mode to reflect source changes instantly.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # On Unix/macOS
# .venv\Scripts\activate # On Windows

# Install your package in editable mode with dev dependencies
pip install -e ".[dev]" # ✅ GOOD: Editable install for live code changes
```
**Pitfall**: Entry points (scripts, plugins) are only discoverable *after* the package is installed (even in editable mode). If you add a new entry point, you might need to re-run `pip install -e .` for it to register.

## 8. Building and Publishing

Generate standard distribution packages (wheels and source distributions) using `build`, and upload them securely with `twine`.

```bash
# Ensure build and twine are installed in your dev environment
pip install build twine

# Clean any previous builds
rm -rf dist/ build/ *.egg-info/

# Build distributions
python -m build # ✅ GOOD: Creates .whl and .tar.gz in dist/

# Upload to TestPyPI (always test here first!)
twine upload --repository testpypi dist/* # ✅ GOOD: Secure upload

# Upload to PyPI
twine upload dist/*
```

## 9. Type Hints

While not `setuptools`-specific, consistently apply type hints throughout your Python codebase. This improves code readability, maintainability, and enables static analysis tools to catch errors early. `setuptools` itself does not dictate type hint usage, but a well-packaged project is a well-typed project.

```python
# ✅ GOOD: Use type hints for clarity and static analysis
def calculate_sum(a: int, b: int) -> int:
    return a + b
```

## 10. Custom Build Logic (When Absolutely Necessary)

If you *must* perform custom build steps (e.g., compiling C extensions, generating code), `setup.py` is still the place for imperative logic. However, keep it minimal and ensure it only handles build-time operations, not metadata.

```python title="setup.py"
# ❌ BAD: Defining metadata here. Use pyproject.toml instead.
# from setuptools import setup
# setup(name='my_package', version='0.1.0', ...)

# ✅ GOOD: Minimal setup.py for custom build logic (e.g., C extensions)
from setuptools import setup, Extension

my_extension = Extension(
    'my_package.my_c_module',
    sources=['src/my_package/my_c_module.c'],
)

setup(
    ext_modules=[my_extension],
)
```
**Note**: Even for extensions, much can be configured in `pyproject.toml` directly with `setuptools`'s `[tool.setuptools.extensions]` table. Only resort to `setup.py` if truly complex, imperative logic is required.