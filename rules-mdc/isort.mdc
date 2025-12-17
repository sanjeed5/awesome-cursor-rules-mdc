---
description: This guide provides definitive rules for configuring and using isort to maintain consistent, readable, and merge-conflict-free Python import statements.
globs: **/*.py
---
# isort Best Practices

`isort` is the definitive tool for sorting Python imports. It ensures consistency, reduces merge conflicts, and improves readability by automatically organizing imports according to PEP 8 and project-specific rules.

## 1. Core Configuration: `pyproject.toml` is Non-Negotiable

Always configure `isort` in `pyproject.toml`. This centralizes tooling configuration and ensures consistent behavior across all development environments.

**❌ BAD: Scattered or Missing Configuration**
Relying on command-line flags or `.isort.cfg` files leads to inconsistency and makes project setup brittle.

**✅ GOOD: Centralized `pyproject.toml`**
Use the `black` profile and explicitly define your project's line length.

```toml
# pyproject.toml
[tool.isort]
profile = "black"
line_length = 88
# Add any top-level internal packages to known_first_party
# This ensures isort correctly groups your internal modules.
known_first_party = ["my_project_core", "my_project_api"]
# Add third-party packages that isort might misclassify as first-party
known_third_party = ["flask", "django", "requests"]
```

## 2. Enforce with `pre-commit` Hooks

Automate `isort` execution on every commit. This is the only way to guarantee consistent import ordering across the team.

**❌ BAD: Manual `isort` Runs**
Forgetting to run `isort` manually leads to unformatted code in the repository and wasted time in code reviews.

**✅ GOOD: `pre-commit` Integration**
Add `isort` to your `.pre-commit-config.yaml`.

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/PyCQA/isort
  rev: 5.13.2 # Use the latest stable version
  hooks:
    - id: isort
      args: ["--profile", "black", "--line-length", "88"]
```

## 3. Code Organization and Import Grouping

`isort` excels at grouping imports. Leverage `known_first_party` to correctly categorize your internal modules.

**❌ BAD: Ambiguous Internal Imports**
Without `known_first_party`, `isort` might treat your internal modules (e.g., `from my_project.utils import ...`) as third-party or local, leading to incorrect grouping.

**✅ GOOD: Explicit `known_first_party` for Modular Projects**
If your project has a `src` directory or multiple top-level packages, define them.

```python
# my_project_core/logic.py
import os # Standard library
import requests # Third-party
from my_project_api.models import User # First-party (correctly grouped)

# ... rest of your code
```

## 4. Avoid Wildcard Imports

Wildcard imports (`from module import *`) obscure dependencies and make code harder to read and refactor. `isort` will sort them, but they are a fundamental anti-pattern.

**❌ BAD: Wildcard Imports**
```python
from my_module import * # What exactly is imported?
```

**✅ GOOD: Explicit Imports**
```python
from my_module import specific_function, AnotherClass
```

## 5. Relative vs. Absolute Imports

Use absolute imports for clarity and maintainability. Relative imports are acceptable *within* a subpackage but should be used sparingly.

**❌ BAD: Over-reliance on Relative Imports**
```python
# In my_package/sub_package/module_a.py
from ..another_sub_package.module_b import func # Hard to trace origin
```

**✅ GOOD: Absolute Imports for Top-Level Modules**
```python
# In my_package/sub_package/module_a.py
from my_project.another_sub_package.module_b import func # Clear path
```

## 6. Performance and CI/CD Integration

`isort` is fast. Integrate it into your CI/CD pipeline as a check-only step to prevent unformatted code from being merged.

**❌ BAD: No CI/CD Check**
Allowing unformatted code to pass CI means `pre-commit` is the only gate, which can be bypassed.

**✅ GOOD: CI/CD `isort --check-only`**
Run `isort --check-only .` in your CI pipeline. This fails the build if imports are not sorted, forcing developers to fix them locally.

## 7. Ruff and `isort` Coexistence

While Ruff can replace `isort`, many teams still use `isort` for its dedicated import handling. If you're using both, ensure Ruff's import sorting is disabled to avoid conflicts.

**✅ GOOD: Disable Ruff's Import Sorting if Using `isort`**
```toml
# pyproject.toml (for Ruff)
[tool.ruff]
# Disable isort-related rules (e.g., I001 for unsorted imports)
ignore = ["I001"]
```