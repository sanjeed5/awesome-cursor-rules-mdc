# Cursor Rules MDC Converter

This script converts Cursor rules from various formats to the `.mdc` format, using LLM to intelligently process and structure the rules.

## Setup

1. Install dependencies using `uv`:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
uv sync
```

2. Set up your environment variables:
Create a `.env` file with your LLM API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

The script processes files from the `awesome-cursorrules/rules` directory and:
1. Creates `.mdc` versions in the same folder as the original files
2. Creates a new folder structure in `awesome-cursor-mdc-rules` with the converted files

### Test Mode

By default, the script runs in test mode, which only processes 2 folders. To process all folders:

1. Open `convert_to_mdc.py`
2. Find the line `test_mode = True`
3. Change it to `test_mode = False`

### Running the Script

```bash
uv run convert_to_mdc.py
```

## Features

- Retries on API failures with exponential backoff
- Processes files in both original location and separate directory
- Maintains folder structure
- Detailed logging
- Test mode for initial validation

## Error Handling

The script includes comprehensive error handling:
- Logs errors for individual file processing
- Continues processing even if some files fail
- Retries LLM API calls on failure
