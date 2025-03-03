# Cursor Rules CLI

> **Disclaimer:** This project is not officially associated with or endorsed by Cursor. It is a community-driven initiative to enhance the Cursor experience.

A simple tool that helps you find and install the right Cursor rules for your project. It scans your project to identify libraries and frameworks you're using and suggests matching rules.

## Features

- 🔍 Auto-detects libraries in your project
- 📝 Supports direct library specification
- 📥 Downloads and installs rules into Cursor
- 🎨 Provides a colorful, user-friendly interface
- 🔀 Works with custom rule repositories
- 🔒 100% privacy-focused (all scanning happens locally)

## Installation

```bash
pip install cursor-rules
```

## Basic Usage

```bash
# Scan current project and install matching rules
cursor-rules

# Specify libraries directly (skips project scanning)
cursor-rules --libraries "react,tailwind,typescript"

# Scan a specific project directory
cursor-rules -d /path/to/my/project
```

## Common Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview without installing anything |
| `--force` | Replace existing rules |
| `-v, --verbose` | Show detailed output |
| `--quick-scan` | Faster scan (checks package files only) |
| `--max-results N` | Show top N results (default: 20) |

## Custom Repositories (not tested yet)

```bash
# Use rules from your own repository
cursor-rules --custom-repo your-username/your-repo

# Save repository setting for future use
cursor-rules --custom-repo your-username/your-repo --save-config
```

## Configuration

```bash
# View current settings
cursor-rules --show-config

# Save settings globally
cursor-rules --save-config

# Save settings for current project only
cursor-rules --save-project-config
```

## Full Options Reference

Run `cursor-rules --help` to see all available options.

## License

MIT

## Todo:
- [ ] Test the custom repo feature