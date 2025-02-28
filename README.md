# MDC Rules Generator

This project generates Cursor MDC (Markdown Cursor) rule files from a structured JSON file containing library information. It uses Exa for semantic search and LLM (Gemini) for content generation.

## Features

- Generates comprehensive MDC rule files for libraries
- Uses Exa for semantic web search to gather best practices
- Leverages LLM to create detailed, structured content
- Supports parallel processing for efficiency
- Tracks progress to allow resuming interrupted runs
- Smart retry system that focuses on failed libraries by default

## Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) for dependency management
- API keys for:
  - Exa (for semantic search)
  - LLM provider (Gemini, OpenAI, or Anthropic)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sanjeed5/awesome-cursor-rules-mdc.git
   cd awesome-cursor-rules-mdc
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with your API keys:
   ```
   EXA_API_KEY=your_exa_api_key
   GOOGLE_API_KEY=your_google_api_key  # For Gemini
   # Or use one of these depending on your LLM choice:
   # OPENAI_API_KEY=your_openai_api_key
   # ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Configuration

The script uses a `config.yaml` file for configuration. You can modify this file to adjust:

- API rate limits
- Output directories
- LLM model selection
- Processing parameters

## Usage

Run the script with:

```bash
uv run src/generate_mdc_files.py
```

By default, the script will only process libraries that failed in previous runs. This behavior helps ensure reliability and efficiency.

### Command-line Options

- `--test`: Run in test mode (process only one library)
- `--tag TAG`: Process only libraries with a specific tag
- `--library LIBRARY`: Process only a specific library
- `--output OUTPUT_DIR`: Specify output directory for MDC files
- `--verbose`: Enable verbose logging
- `--workers N`: Set number of parallel workers
- `--rate-limit N`: Set API rate limit calls per minute
- `--regenerate-all`: Process all libraries, including previously completed ones

### Examples

Process failed libraries (default behavior):
```bash
uv run src/generate_mdc_files.py
```

Regenerate all libraries:
```bash
uv run src/generate_mdc_files.py --regenerate-all
```

Process only Python libraries:
```bash
uv run src/generate_mdc_files.py --tag python
```

Process a specific library:
```bash
uv run src/generate_mdc_files.py --library react
```

## Project Structure

- `src/`: Source code directory
  - `generate_mdc_files.py`: Main script
  - `config.yaml`: Configuration file
  - `mdc-instructions.txt`: Instructions for MDC generation
  - `logs/`: Log files directory
  - `exa_results/`: Directory for Exa search results
- `rules.json`: Input file with library information
- `rules-mdc/`: Output directory for generated MDC files
- `pyproject.toml`: Project dependencies and metadata

## License

[MIT License](LICENSE)