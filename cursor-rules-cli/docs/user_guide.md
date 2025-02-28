# Cursor Rules CLI - User Guide

## Introduction

Cursor Rules CLI is a command-line tool that helps you discover and install relevant Cursor rules (MDC files) for your projects. The tool scans your project directory to identify libraries and frameworks, then suggests and installs appropriate rules to enhance your development experience with Cursor.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installing from PyPI

```bash
pip install cursor-rules
```

### Installing from Source

```bash
git clone https://github.com/sanjeed5/awesome-cursor-rules-mdc.git
cd awesome-cursor-rules-mdc/cursor-rules-cli
pip install -e .
```

## Basic Usage

To scan your current project directory and install relevant rules:

```bash
cursor-rules
```

This will:
1. Scan your project directory for libraries and frameworks
2. Match detected libraries with available rules
3. Present a list of relevant rules for selection
4. Download and install selected rules to your `.cursor/rules` directory

## Command-Line Options

```
usage: cursor-rules [-h] [-d DIRECTORY] [--dry-run] [--force] [--source SOURCE]
                    [--rules-json RULES_JSON] [-v]

Scan your project and install relevant Cursor rules (.mdc files).

options:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        Project directory to scan (default: current directory)
  --dry-run             Show what would be done without making changes
  --force               Force overwrite existing rules
  --source SOURCE       Base URL for downloading rules
  --rules-json RULES_JSON
                        Path to custom rules.json file
  -v, --verbose         Enable verbose output
```

### Examples

#### Scan a specific directory

```bash
cursor-rules -d /path/to/your/project
```

#### Dry run (no changes)

```bash
cursor-rules --dry-run
```

#### Force overwrite existing rules

```bash
cursor-rules --force
```

#### Use verbose output

```bash
cursor-rules -v
```

#### Use a custom rules.json file

```bash
cursor-rules --rules-json /path/to/custom/rules.json
```

## Interactive Selection

After scanning your project, the tool will present a list of relevant rules:

```
Available Cursor rules for your project:
1. react.mdc - React best practices
2. typescript.mdc - TypeScript tips and tricks
3. eslint.mdc - ESLint configuration help

Select rules to install:
  * Enter comma-separated numbers (e.g., 1,3,5)
  * Type 'all' to select all rules
  * Type 'none' to cancel
>
```

You can:
- Enter specific numbers (e.g., `1,3`) to select individual rules
- Type `all` to select all rules
- Type `none` or press Enter to cancel

## Rule Installation

Selected rules will be downloaded and installed to your `.cursor/rules` directory. If a rule already exists, it will not be overwritten unless you use the `--force` option.

Existing rules will be backed up before being overwritten.

## Troubleshooting

### No libraries detected

If the tool doesn't detect any libraries in your project, try:
- Using the `-v` option for verbose output to see what the scanner is checking
- Ensuring your project has dependency files (package.json, requirements.txt, etc.)
- Running the tool from the root directory of your project

### Rule installation fails

If rule installation fails, check:
- That you have write permissions to the `.cursor/rules` directory
- That you're not trying to overwrite existing rules without the `--force` option
- Your internet connection if downloads are failing

## Getting Help

If you encounter any issues or have questions, please:
- Check the [GitHub repository](https://github.com/sanjeed5/awesome-cursor-rules-mdc) for updates
- Open an issue on GitHub if you find a bug
- Refer to the developer documentation for more technical details 