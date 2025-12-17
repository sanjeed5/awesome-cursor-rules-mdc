---
description: This guide defines the definitive Git best practices for our team, ensuring clean history, efficient collaboration, and high code quality through structured workflows, commit standards, and automated hooks.
globs: **/*
---
# Git Best Practices

Git is the backbone of our development workflow. Adhering to these practices ensures a clean, readable history, minimizes conflicts, and enables rapid, reliable delivery. These are non-negotiable standards.

## 1. Branching Strategy: Feature-Branch Workflow

We use a **feature-branch workflow** with short-lived branches. All development happens on dedicated branches, never directly on `main` or `develop`.

*   **`main` branch**: Always production-ready. Only merges from release branches or squashed feature branches are allowed. Protected.
*   **`develop` branch**: Integrates completed features. Protected.
*   **Feature branches**: Created from `develop`, short-lived, for a single feature or bug fix. Merged into `develop` via Pull Requests (PRs).

### Guideline: Branch Naming

Name branches clearly and consistently, linking directly to a ticket or feature.

❌ **BAD**:
```bash
git checkout -b my-feature
git checkout -b fix
```

✅ **GOOD**:
```bash
# For a new feature (e.g., JIRA ticket FEAT-123)
git checkout -b feat/FEAT-123-add-user-profile

# For a bug fix (e.g., JIRA ticket BUG-456)
git checkout -b fix/BUG-456-auth-redirect-loop

# For a refactor
git checkout -b refactor/improve-logging-middleware
```

## 2. Commit Messages: Conventional Commits

Every commit message must follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This enables automated changelog generation and semantic versioning.

**Format**: `<type>(<scope>): <description>`

*   **`type`**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
*   **`scope` (optional)**: The part of the codebase affected (e.g., `auth`, `api`, `ui`, `database`).
*   **`description`**: Concise, imperative, present tense. Max 50 characters.
*   **Body (optional)**: More detailed explanation, breaking changes, references to issues.

### Guideline: Write Clear, Atomic Commits

Each commit should represent a single logical change.

❌ **BAD**:
```bash
git commit -m "Fixed bug and added new feature"
```

✅ **GOOD**:
```bash
# First commit for the fix
git commit -m "fix(auth): Correctly handle expired tokens" -m "This fixes an issue where users were stuck in a redirect loop after their session expired. Now, they are redirected to the login page."

# Second commit for the feature
git commit -m "feat(profile): Implement user profile view" -m "Adds a new page for users to view and edit their profile information."
```

## 3. History Management: Rebase, Don't Merge (on feature branches)

Maintain a linear, clean history on feature branches before merging into `develop`. Use `git rebase -i` to squash, reorder, or edit commits.

### Guideline: Clean Up Local History Before PR

Before pushing your feature branch for a Pull Request, rebase it onto the latest `develop` and squash related commits into logical units.

❌ **BAD**:
```bash
# On feature/my-feature branch
git pull origin develop # Creates a merge commit
git push origin feature/my-feature # Pushes messy history
```

✅ **GOOD**:
```bash
# On feature/my-feature branch
git fetch origin
git rebase -i origin/develop # Interactively clean up commits
# ... complete rebase, squash WIP commits ...
git push --force-with-lease origin feature/my-feature # Force push after rebase
```
**Note**: Only force push your *own* feature branches that haven't been merged or shared widely. Never force push to `main` or `develop`.

### Guideline: Merge Feature Branches with `--no-ff`

When merging a feature branch into `develop` (via PR), always use `--no-ff` to preserve the branch history.

❌ **BAD**:
```bash
# After PR approval, merging directly on local machine
git checkout develop
git merge feature/my-feature # Might fast-forward, losing branch context
```

✅ **GOOD**:
```bash
# After PR approval, on develop branch
git checkout develop
git pull origin develop # Ensure develop is up-to-date
git merge --no-ff feature/my-feature -m "Merge feat(FEAT-123): Add user profile"
git push origin develop
```
**Note**: Our PR process on GitHub/GitLab should enforce this automatically.

## 4. Code Quality & Security: Git Hooks with `pre-commit`

Automate code quality checks and security scans *before* code hits the repository. We use the `pre-commit` framework.

### Guideline: Install and Use `pre-commit`

Every developer must install and keep `pre-commit` hooks updated. This prevents common issues like linting errors, formatting inconsistencies, and accidental secret commits.

1.  **Install `pre-commit`**:
    ```bash
    pip install pre-commit # If using Python
    # Or via brew, npm, etc.
    ```
2.  **Install hooks in your repo**:
    ```bash
    pre-commit install
    ```
3.  **Update hooks regularly**:
    ```bash
    pre-commit autoupdate
    ```

### Example `.pre-commit-config.yaml` (project-specific)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: detect-private-key
      - id: no-commit-to-branch
        args: [--branch, main, --branch, develop] # Prevent direct commits to protected branches

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: 'v0.1.9'
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline'] # Manage known false positives
```
This configuration will run `black` (Python formatter), `ruff` (Python linter), and `detect-secrets` on staged files, among other checks, before a commit is allowed.

## 5. Repository Hygiene: `.gitignore` and Large Files

Keep your repository clean and focused on source code.

### Guideline: Use a Comprehensive `.gitignore`

Exclude generated files, dependencies, build artifacts, and sensitive information.

❌ **BAD**:
```
# .gitignore
*.log
```

✅ **GOOD**:
```
# .gitignore
# Operating System
.DS_Store
Thumbs.db

# Build artifacts
/dist/
/build/

# Dependencies
/node_modules/
/venv/
__pycache__/

# IDE files
.idea/
.vscode/

# Environment variables & secrets
.env
*.env
config.local.js
```

### Guideline: Manage Large Files with Git LFS

Never commit large binary files (images, videos, large datasets, compiled executables) directly to Git. Use Git Large File Storage (LFS).

1.  **Install Git LFS**: `git lfs install`
2.  **Track file types**:
    ```bash
    git lfs track "*.psd"
    git lfs track "assets/*.mp4"
    ```
3.  **Add to `.gitattributes`**: This command updates `.gitattributes` which must be committed.
4.  **Add and commit files normally**:
    ```bash
    git add .gitattributes
    git add my_large_file.psd
    git commit -m "chore(assets): Add large PSD file via LFS"
    ```

## 6. Resolving Conflicts: Proactive and Careful

Merge conflicts are inevitable. Resolve them carefully and proactively.

### Guideline: Pull Frequently

Pull changes from `develop` (or your base branch) frequently to minimize the scope of potential conflicts.

```bash
git checkout feature/my-feature
git pull origin develop --rebase # Rebase your branch onto develop to avoid merge commits
```

### Guideline: Resolve Conflicts Manually

Use your IDE's merge tool or `git mergetool` to resolve conflicts. Understand each change.

❌ **BAD**:
```bash
git merge develop --no-edit -X theirs # Blindly taking "their" changes
```

✅ **GOOD**:
```bash
git merge develop # Git will prompt for conflicts
# Open conflicted files in your IDE, resolve manually
# Or use `git mergetool`
git add <resolved_files>
git commit -m "Merge develop into feat/my-feature and resolve conflicts"
```