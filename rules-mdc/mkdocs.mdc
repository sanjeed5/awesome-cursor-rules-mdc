---
description: Definitive guidelines for writing, structuring, and maintaining high-quality MkDocs documentation using modern best practices and Material for MkDocs.
globs: **/*.{md,mdx}
---
# mkdocs Best Practices

This guide establishes the definitive standards for writing and maintaining documentation using MkDocs, especially with the Material for MkDocs theme. Adhere to these rules to ensure consistency, readability, and maintainability across all our projects.

## 1. Code Organization & Structure

Always organize your documentation for logical navigation and maintainability.

### 1.1 Root Structure

Place `mkdocs.yml` at the project root, and all Markdown source files within a top-level `docs/` directory. The homepage must be `index.md`.

❌ BAD:
```
my-project/
├── docs/
│   ├── mkdocs.yml  # mkdocs.yml should be at project root
│   └── homepage.md # Homepage should be index.md
```

✅ GOOD:
```
my-project/
├── mkdocs.yml
└── docs/
    ├── index.md
    └── user-guide/
        └── getting-started.md
```

### 1.2 Navigation Ordering

Control navigation explicitly using the `mkdocs-awesome-pages-plugin` and `.pages` files. Never rely on default alphabetical ordering.

❌ BAD:
```
# mkdocs.yml
nav:
  - Home: index.md
  - API: api.md
  - Setup: setup.md # Order is alphabetical, not logical
```

✅ GOOD:
```
# docs/.pages
nav:
  - Home: index.md
  - Setup: setup.md
  - API: api.md
  - ... # Include remaining pages automatically
```

## 2. Markdown Style & Content

Enforce a strict Markdown style for readability and consistency, ideally with `markdownlint`.

### 2.1 Headings Hierarchy

Use a single H1 (`#`) per page for the main title. Subsequent headings must follow a strict, sequential hierarchy (H2, H3, etc.). Never skip heading levels.

❌ BAD:
```markdown
# My Page Title
### A Sub-section # Skipped H2
```

✅ GOOD:
```markdown
# My Page Title
## Introduction
### Key Concepts
```

### 2.2 Line Length

Limit text lines to 80 characters for improved readability and diffs. Exceptions are allowed for long URLs, tables, and code blocks.

❌ BAD:
```markdown
This is a very long line of text that exceeds eighty characters and makes it difficult to read and review, especially in side-by-side diffs.
```

✅ GOOD:
```markdown
This is a standard line of text that adheres to the eighty-character limit.
It improves readability and simplifies code reviews.
```

### 2.3 Fenced Code Blocks

Always use fenced code blocks with language identifiers for syntax highlighting. Never use indented code blocks.

❌ BAD:
```markdown
    This is an indented code block.
    It lacks syntax highlighting and is harder to read.
```

✅ GOOD:
```python
def hello_world():
    print("Hello, MkDocs!")
```

### 2.4 Internal Links

Use relative paths for all internal links to Markdown files. MkDocs automatically converts these to correct HTML paths.

❌ BAD:
```markdown
Please see the [Getting Started](/user-guide/getting-started/) guide. # Absolute path
Please see the [Getting Started](https://example.com/user-guide/getting-started.md) guide. # Full URL
```

✅ GOOD:
```markdown
Please see the [Getting Started](user-guide/getting-started.md) guide. # Relative path
```

### 2.5 External Links & Buttons

Mark external links to open in a new tab using `{target=_blank}`. Use Material for MkDocs button syntax for prominent calls to action.

❌ BAD:
```markdown
[Our GitHub](https://github.com/our-org) # Opens in same tab
<a href="https://github.com/our-org" class="button">GitHub</a> # Raw HTML
```

✅ GOOD:
```markdown
[Our GitHub](https://github.com/our-org){target=_blank}
[Visit Our Blog](https://practical.li/blog){.md-button .md-button-primary target=_blank}
```

### 2.6 Admonitions

Use Material for MkDocs admonitions for notes, warnings, and tips. Never use blockquotes for this purpose.

❌ BAD:
```markdown
> **Warning:** This action is irreversible.
```

✅ GOOD:
```markdown
!!! warning "Irreversible Action"
    This action is irreversible. Proceed with caution.
```

## 3. Performance & Maintainability

Prioritize automated checks and efficient content delivery.

### 3.1 Automated Linting & Validation

Integrate `markdownlint` and `mkdocs-code-validator` into your CI pipeline. This catches style violations and validates code snippets automatically.

❌ BAD: Relying solely on manual review for Markdown style and code correctness.

✅ GOOD:
```yaml
# .github/workflows/ci.yml
name: Docs CI
on: [push, pull_request]
jobs:
  lint-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: avto-dev/markdown-lint@v1 # markdownlint
      - uses: actions/setup-python@v3
        with:
          python-version: '3.x'
      - run: pip install mkdocs mkdocs-material mkdocs-code-validator
      - run: mkdocs build --strict # mkdocs-code-validator runs during build
```

## 4. Common Pitfalls & Gotchas

Avoid these common mistakes to prevent broken documentation.

### 4.1 Broken Navigation

An invalid `.pages` file will prevent your site from building. Always test changes locally with `mkdocs serve`.

❌ BAD:
```yaml
# docs/.pages (invalid syntax)
nav:
  - Introduction: intro.md
  - Section 1
    - Page 1: page1.md # Incorrect indentation
```

✅ GOOD:
```yaml
# docs/.pages
nav:
  - Introduction: intro.md
  - Section 1:
    - Page 1: page1.md
```

### 4.2 Raw HTML Usage

Avoid embedding raw HTML unless absolutely necessary (e.g., specific embeds like YouTube videos). Prefer Markdown and Material for MkDocs extensions.

❌ BAD:
```markdown
<p style="color: red;">Important text</p>
```

✅ GOOD:
```markdown
!!! danger "Important"
    Important text
```