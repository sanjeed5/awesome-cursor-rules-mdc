# Cursor Rules CLI

A simple tool that helps you find and install the right Cursor rules for your project. It looks at what libraries and frameworks you're using and suggests matching rules.

## What does it do?

- 🔍 Looks at your project and figures out what libraries you're using
- 📝 Lets you directly specify libraries if you know what you want
- 📥 Downloads and installs the right rules into your Cursor
- 🎨 Shows everything in a nice, colorful interface
- 🔀 Works with custom rule repositories too

## Getting Started

### Quick Install

```bash
pip install cursor-rules
```

### Running it

The easiest way to use it:
```bash
# Just run this in your project folder
cursor-rules
```

This will:
1. Look at your project
2. Show you what rules match your project
3. Let you choose which ones to install
4. Install them for you

### I know what rules I want

If you already know what libraries you want rules for:
```bash
# Just list them with commas
cursor-rules --libraries "react,tailwind,typescript"
```

This will:
1. Skip scanning your project directory
2. Only match rules for the libraries you specified
3. Let you choose which ones to install
4. Install them for you

### Common Use Cases

```bash
# Looking at a specific project folder
cursor-rules -d /path/to/my/project

# Just want to see what it would do (without installing anything)
cursor-rules --dry-run

# Want to replace existing rules
cursor-rules --force

# Want to see more details about what's happening
cursor-rules -v
```

## Making it Work Your Way

### Want to see fewer/more results?
```bash
# Show only 5 results
cursor-rules --max-results 5

# Only show really good matches
cursor-rules --min-score 0.8
```

### Skip Project Scanning
```bash
# Directly specify libraries (skips project scanning)
cursor-rules --libraries "react,vue,django"

# Useful when you know exactly what you need
cursor-rules --libraries "pytorch,pandas,matplotlib" --force
```

### Quick Scan vs Deep Scan
```bash
# Do a quick scan (faster but might miss some things)
cursor-rules --quick-scan
```

### Using Your Own Rules Repository

Have your own collection of rules? You can use those instead:
```bash
# Use rules from your repository
cursor-rules --custom-repo your-username/your-repo

# Make it remember your repository for next time
cursor-rules --custom-repo your-username/your-repo --save-config
```

### Saving Your Preferences

```bash
# See what settings you're using
cursor-rules --show-config

# Save your current settings for all projects
cursor-rules --save-config

# Save settings just for this project
cursor-rules --save-project-config
```

## Need More Help?

Run this to see all available options:
```bash
cursor-rules --help
```

This will show you all the options:

```
Options:
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
                        Save current settings as project-specific configuration
  --show-config         Show current configuration
  --quick-scan          Perform a quick scan (only check package files, not imports)
  --max-results MAX_RESULTS
                        Maximum number of rules to display (default: 20)
  --min-score MIN_SCORE
                        Minimum relevance score for rules (0-1, default: 0.5)
  --libraries LIBRARIES
                        Comma-separated list of libraries to match directly (e.g., 'react,vue,django'). 
                        When specified, project scanning is skipped.
  -v, --verbose         Enable verbose output
```

## License

MIT

## Todo:
- [ ] Test the custom repo feature