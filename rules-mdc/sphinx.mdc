---
description: This rule file provides opinionated, practical guidelines for setting up, writing, and maintaining Python project documentation using Sphinx, emphasizing modern best practices, clear structure, and automated validation.
globs: **/*.py
---
# Sphinx Best Practices

Sphinx is the definitive tool for Python project documentation. This guide outlines the essential workflow and best practices for creating intelligent, beautiful, and maintainable docs. Always prioritize consistency within your project over strict adherence to external guides, but default to these modern standards.

## 1. Project Setup and Structure

Establish a clean, predictable documentation structure from the start.

### 1.1. Use a Virtual Environment

Always isolate your documentation build dependencies. This prevents conflicts and ensures reproducible builds.

```bash
# ❌ BAD: Installing Sphinx globally
pip install sphinx

# ✅ GOOD: Create and activate a virtual environment
python -m venv .venv/docs
source .venv/docs/bin/activate  # On Windows: .venv\docs\Scripts\activate
pip install sphinx sphinx-autobuild myst-parser sphinx-rtd-theme sphinx.ext.napoleon
```

### 1.2. Standard Directory Layout

Place all documentation source files in a dedicated `docs/` directory at your project root.

```
my_project/
├── .venv/
├── my_package/
│   └── __init__.py
│   └── module.py
├── docs/
│   ├── conf.py
│   ├── index.md        # Or index.rst
│   ├── tutorial/       # For manual narrative content
│   │   └── getting_started.md
│   ├── api/            # For auto-generated API docs (managed by apidoc)
│   │   └── my_package.rst
│   ├── _static/        # Custom static files (images, CSS)
│   ├── _templates/     # Custom Jinja2 templates
│   └── Makefile        # Or make.bat for Windows
├── README.md
├── CHANGELOG.md
├── LICENSE
└── pyproject.toml
```

### 1.3. Initialize with `sphinx-quickstart`

Use the quickstart tool once to scaffold your `docs/` directory. Accept the defaults, then customize.

```bash
# From your project root:
cd docs
sphinx-quickstart
```

### 1.4. Prefer MyST Markdown for New Content

For narrative documentation, MyST Markdown (`.md`) is more approachable than reStructuredText (`.rst`) while retaining Sphinx's powerful features. Use it for all new manual content.

**`docs/conf.py`:**

```python
# ✅ GOOD: Enable MyST Parser
extensions = [
    'myst_parser',
    # ... other extensions
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
```

## 2. Docstring Conventions and API Generation

Generate API documentation directly from your Python code's docstrings.

### 2.1. Adopt Google-Style Docstrings with Type Hints

Write docstrings in Google style for clarity and consistency. Always include type hints in your function/method signatures. The `sphinx.ext.napoleon` extension will parse these automatically.

**`docs/conf.py`:**

```python
# ✅ GOOD: Enable Napoleon for Google/NumPy style docstrings
extensions = [
    # ... other extensions
    'sphinx.ext.napoleon',
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False # Stick to one style
napoleon_use_param = True
napoleon_use_rtype = True
```

**`my_package/module.py`:**

```python
# ❌ BAD: Inconsistent, missing types, hard to parse
def calculate_sum(a, b):
    """Adds two numbers."""
    return a + b

# ✅ GOOD: Google-style, type-hinted docstring
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of `a` and `b`.
    """
    return a + b
```

### 2.2. Automate API Documentation with `autodoc` and `autosummary`

Use `sphinx.ext.autodoc` to pull docstrings and `sphinx.ext.autosummary` to generate clean API tables.

**`docs/conf.py`:**

```python
# ✅ GOOD: Enable core API extensions
extensions = [
    # ... other extensions
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]
autodoc_member_order = 'bysource' # Keep members in source order
autodoc_typehints = 'signature'   # Show type hints in signatures
autosummary_generate = True       # Automatically generate stub files
```

**`docs/api/index.rst` (or `.md` with MyST directives):**

```rst
.. automodule:: my_package.module
   :members:

.. autosummary::
   :toctree: _autosummary
   :template: module.rst

   my_package.module.calculate_sum
   my_package.module.MyClass
```

### 2.3. Manage `sphinx-apidoc` Carefully

Run `sphinx-apidoc` once to scaffold API `.rst` files. **Never let it overwrite your manual documentation.**

```bash
# From docs/ directory:
# ❌ BAD: Running apidoc without exclusion, risks overwriting manual content
sphinx-apidoc -o api/ ../my_package

# ✅ GOOD: Run apidoc once, then manually integrate and exclude.
# First run (from docs/):
sphinx-apidoc -o api/ ../my_package -f -e -M

# Then, add 'api/' to your master toctree in index.md/rst.
# Manually edit generated files (e.g., api/modules.rst) to include specific modules.
# Add 'api/my_package.rst' to `exclude_patterns` in conf.py if you want to prevent
# apidoc from regenerating *that specific file* on subsequent runs,
# but generally, it's better to let apidoc manage its own output directory
# and only exclude specific files if they are manually edited.
# For manual tutorials, ensure they are in a separate directory (e.g., `tutorial/`)
# and *not* in the apidoc output directory.
```

## 3. Content Management and Cross-Referencing

Structure your documentation logically and enable seamless navigation.

### 3.1. Master `toctree` for Structure

Organize your documentation using the `toctree` directive in your main `index.md` (or `index.rst`).

**`docs/index.md`:**

```markdown
# My Project Documentation

Welcome to the documentation for My Project!

```{toctree}
:maxdepth: 2
:caption: Contents:

tutorial/getting_started
api/index
```

### 3.2. Link External Projects with `intersphinx`

Easily link to documentation of other Python projects (e.g., Python, NumPy, Django).

**`docs/conf.py`:**

```python
# ✅ GOOD: Enable intersphinx and define mappings
extensions = [
    # ... other extensions
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'requests': ('https://requests.readthedocs.io/en/latest/', None),
}
```

**`docs/tutorial/getting_started.md`:**

```markdown
# Getting Started

This project uses [Python's built-in `list` type](:py:class:`python:list`).
For network requests, we rely on the [Requests library](:py:mod:`requests:requests`).
```

## 4. Building and Testing Documentation

Integrate documentation builds into your development and CI workflow.

### 4.1. Use `sphinx-autobuild` for Live Previews

During development, `sphinx-autobuild` provides live-reloading previews, accelerating your writing process.

```bash
# From docs/ directory:
# ✅ GOOD: Live preview with automatic rebuild on file changes
sphinx-autobuild . _build/html --port 8000
```

### 4.2. Integrate Docs Build into CI/CD

Ensure your documentation builds successfully and is up-to-date by including a build step in your CI pipeline. Treat documentation errors as build failures.

**`.github/workflows/ci.yml` (Example for GitHub Actions):**

```yaml
# ✅ GOOD: Build docs in CI
name: CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m venv .venv/docs
        source .venv/docs/bin/activate
        pip install -e .[docs] # Install project and docs dependencies
    - name: Build Sphinx documentation
      run: |
        source .venv/docs/bin/activate
        cd docs
        make html # Or sphinx-build -W -b html . _build/html
```

### 4.3. Enforce Docstring Style and Quality

Use linters like `pydocstyle` to enforce PEP 257 docstring conventions and `flake8` for general PEP 8 compliance.

```bash
# ✅ GOOD: Linting for code and docstrings
pip install flake8 pydocstyle

# In your CI or pre-commit hook:
flake8 my_package/
pydocstyle my_package/
```

### 4.4. Test Docstring Examples with `doctest`

Embed simple, runnable examples in your docstrings and use `sphinx.ext.doctest` to verify them.

**`docs/conf.py`:**

```python
# ✅ GOOD: Enable doctest
extensions = [
    # ... other extensions
    'sphinx.ext.doctest',
]
```

**`my_package/module.py`:**

```python
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    >>> multiply(2, 3)
    6
    >>> multiply(0, 5)
    0
    """
    return a * b
```

## 5. Packaging and Distribution

Ensure your documentation is easily accessible and distributable.

### 5.1. Host on Read the Docs

Read the Docs provides free, automated hosting for Sphinx documentation, integrating directly with your version control system. Configure it to build on every commit.

### 5.2. Include Docs in Source Distribution

If your documentation is essential for offline use or specific deployment scenarios, ensure it's included in your project's source distribution.

**`pyproject.toml` (or `MANIFEST.in` for older projects):**

```toml
# ✅ GOOD: Include docs in source distribution
[tool.setuptools.package-data]
"my_package" = ["py.typed"] # Example for type hints
"my_project" = ["docs/*"] # Include entire docs directory
```