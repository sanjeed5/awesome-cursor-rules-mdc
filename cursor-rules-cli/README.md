# Cursor Rules CLI

A command-line tool to scan your projects and suggest relevant Cursor rules (.mdc files) based on the libraries and frameworks you're using.

## Features

- 🔍 Automatically scans your project to detect libraries and frameworks
- 🔄 Matches detected libraries with available Cursor rules
- 📥 Downloads and installs relevant rules to your `.cursor/rules` directory
- 🎨 Interactive and colorful CLI interface
- 🔀 Supports custom repositories for advanced users

## Installation

```bash
# Clone the repository
git clone https://github.com/sanjeed5/awesome-cursor-rules-mdc
cd awesome-cursor-rules-mdc/cursor-rules-cli

# Install using pip
pip install -e .
```

## Usage

```bash
# Scan the current directory and install relevant rules
cursor-rules

# Scan a specific directory
cursor-rules scan -d /path/to/project

# Show what would be done without making changes
cursor-rules scan --dry-run

# Force overwrite existing rules
cursor-rules scan --force

# Enable verbose output
cursor-rules scan -v
```

## Advanced Usage

### Using a Custom Repository (untested)

You can use your own forked repository of Cursor rules. This is a CLI-wide setting that applies to all projects:

```bash
# Use a custom repository for the current scan
cursor-rules scan --custom-repo username/repo

# Set custom repository globally without running scan
cursor-rules scan --custom-repo username/repo --set-repo

# Save this setting as your default configuration
cursor-rules scan --custom-repo username/repo --save-config
```

Note: Custom repository settings are always global and not project-specific. The repository must contain a valid `rules.json` file at the root level.

### Configuration Management

```bash
# Show current configuration
cursor-rules scan --show-config

# Save current settings as global configuration
cursor-rules scan --save-config

# Save current settings as project-specific configuration (except repository settings)
cursor-rules scan --save-project-config
```

## Configuration Files

The CLI uses the following configuration files:

- Global configuration: `~/.cursor/rules-cli-config.json` (includes custom repository settings)
- Project-specific configuration: `.cursor-rules-cli.json` in the project directory (excludes repository settings)

Project-specific configuration takes precedence over global configuration for project-specific settings.

## Available Options

```
  -h, --help            Show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        Project directory to scan (default: current directory)
  --dry-run             Show what would be done without making changes
  --force               Force overwrite existing rules
  --source SOURCE       Base URL for downloading rules
  --custom-repo CUSTOM_REPO
                        GitHub username/repo for a forked repository (e.g., 'username/repo')
  --set-repo            Set custom repository without running scan
  --rules-json RULES_JSON
                        Path to custom rules.json file
  --save-config         Save current settings as default configuration
  --save-project-config
                        Save current settings as project-specific configuration (except repository settings)
  --show-config         Show current configuration
  -v, --verbose         Enable verbose output
```

## License

MIT 

## Todo:
- [ ] Test the custom repo