# MDC Generator for Library Best Practices

This repository contains Cursor rule files (.mdc) that provide AI-powered best practices and guidelines for various libraries and frameworks. The rules in `rules-mdc/` are comprehensive and focused, offering guidance for a wide range of programming libraries and frameworks.

## Repository Structure

```
.
├── rules-mdc/          # Comprehensive rule files in a flat structure
├── cursor-rules-cli/   # Source code and configuration for rule generation
│   ├── generate_mdc_files.py
│   ├── libraries.json
│   ├── config.yaml
│   └── ...
└── logs/               # Log files from rule generation runs
```

## Features

- **Tag-Based Library Organization**: Libraries are organized with tags instead of nested categories for more flexible categorization
- **Flat Directory Structure**: All MDC files are stored in a single directory for easier access
- **Configurable**: All settings can be adjusted in `config.yaml`
- **Parallel Processing**: Process multiple libraries simultaneously
- **Progress Tracking**: Resume from where you left off if the process is interrupted
- **Smart Glob Pattern Generation**: Automatically determines appropriate glob patterns for different libraries
- **Rate Limiting**: Configurable API rate limits to avoid throttling
- **Robust Error Handling**: Comprehensive error handling and logging

## Prerequisites

1. Python 3.8+
2. Required packages (install via `uv sync`)
3. Exa API key (set as environment variable `EXA_API_KEY`)
4. LiteLLM API key (set as environment variable based on your provider)

## Configuration

The script uses a configuration file (`config.yaml`) with the following structure:

```yaml
paths:
  mdc_instructions: "mdc-instructions.txt"
  libraries_json: "libraries.json"
  output_dir: "rules-mdc"
  exa_results_dir: "exa_results"

api:
  llm_model: "gemini/gemini-2.0-flash"
  rate_limit_calls: 2000
  rate_limit_period: 60
  max_retries: 3
  retry_min_wait: 4
  retry_max_wait: 10

processing:
  max_workers: 4
  chunk_size: 50000

tags:
  primary_tags: ["python", "javascript", "typescript", "rust", "go", "java", "php", "ruby"]
  category_tags: ["frontend", "backend", "database", "ai", "ml", "testing", "development"]
```

## Usage

### Basic Usage

```bash
uv run generate_mdc_files.py
```

### Test Mode (Process Only One Library)

```bash
uv run generate_mdc_files.py --test
```

### Process Specific Tags or Libraries

```bash
uv run generate_mdc_files.py --tag python
uv run generate_mdc_files.py --library react
```

### Adjust Parallel Processing

```bash
uv run generate_mdc_files.py --workers 8
```

### Adjust Rate Limits

```bash
uv run generate_mdc_files.py --rate-limit 50
```

### Verbose Logging

```bash
uv run generate_mdc_files.py --verbose
```

## File Structure

The generated MDC files are organized in a flat structure:

```
rules-mdc/
├── react.mdc
├── vue.mdc
├── django.mdc
├── flask.mdc
├── pytest.mdc
└── ...
```

## Available Libraries

All currently supported libraries can be found in `libraries.json`. This file organizes libraries with tags for flexible categorization:

```json
{
  "libraries": [
    {
      "name": "react",
      "tags": ["frontend", "framework", "javascript"]
    },
    {
      "name": "django",
      "tags": ["backend", "framework", "python", "orm", "full-stack"]
    },
    ...
  ]
}
```

The libraries cover a wide range of categories including:
- Frontend frameworks and libraries
- Backend frameworks
- UI libraries and components
- State management solutions
- Database tools and ORMs
- Development tools (testing, linting, etc.)
- Cross-platform frameworks
- AI/ML libraries
- Web technologies
- And many more

## Progress Tracking

The script maintains a progress file (`mdc_generation_progress.json`) to track which libraries have been processed. This allows you to resume the process if it's interrupted.

## Troubleshooting

- If you encounter API rate limit issues, adjust the `rate_limit_calls` in the configuration.
- If the script is using too much memory, reduce the `chunk_size` in the configuration.
- If the script is too slow, increase the `max_workers` in the configuration (but be mindful of API rate limits).