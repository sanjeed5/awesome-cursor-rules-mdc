# Cursor Rules CLI

> **Disclaimer:** This project is not officially associated with or endorsed by Cursor. It is a community-driven initiative to enhance the Cursor experience.

<a href="https://www.producthunt.com/posts/cursor-rules-cli?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-cursor&#0045;rules&#0045;cli" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=936513&theme=light&t=1741030422709" alt="Cursor&#0032;Rules&#0032;CLI - Auto&#0045;install&#0032;relevant&#0032;Cursor&#0032;rules&#0032;with&#0032;one&#0032;simple&#0032;command | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>

A simple tool that helps you find and install the right Cursor rules for your project. It scans your project to identify libraries and frameworks you're using and suggests matching rules.

![Cursor Rules CLI Demo](cursor-rules-cli.gif)

## Features

- 🔍 Auto-detects libraries in your project
- 📝 Supports direct library specification
- 📥 Downloads and installs rules into Cursor
- 🎨 Provides a colorful, user-friendly interface
- 🔀 Works with custom rule repositories
- 🔒 100% privacy-focused (all scanning happens locally)
- 🔄 GitHub API integration for reliable downloads

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

## Custom Repositories

```bash
# Use rules from your own GitHub repository
cursor-rules --source https://github.com/your-username/your-repo

# Save repository setting for future use
cursor-rules --source https://github.com/your-username/your-repo --save-config
```

## Repository URL Format

The tool now uses the GitHub API to reliably download rules. You can specify the repository URL in several formats:

```bash
# Standard GitHub repository URL (recommended)
cursor-rules --source https://github.com/username/repo

# With a specific branch
cursor-rules --source https://github.com/username/repo/tree/branch-name

# Legacy raw content URL will also work
cursor-rules --source https://raw.githubusercontent.com/username/repo/branch
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