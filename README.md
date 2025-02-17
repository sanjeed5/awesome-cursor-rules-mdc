# Awesome Cursor Rules .mdc [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

All thanks to the awesome content on https://github.com/PatrickJS/awesome-cursorrules which was processed to create this. Waiting for this to be integrated into the awesome https://cursor.directory/ ([PR raised](https://github.com/PatrickJS/awesome-cursorrules/pull/60))


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
1. Creates `.mdc` versions in the same folder as the original files.
2. Creates a new folder structure in `awesome-cursor-mdc-rules` with the converted files.
3. Generates a conversion report in `conversion_report.json` after processing.

### Test Mode

By default, the script runs in test mode, which only processes 2 folders. To process all folders:

1. Open `convert_to_mdc.py`
2. Find the line `test_mode = True`
3. Change it to `test_mode = False`

### Running the Script

```bash
uv run convert_to_mdc.py
```

### Analyzing Conversion Status

To analyze the conversion status and generate a report, run the following command:

```bash
uv run test_conversion.py
```

This will compare the source and output folders, providing a summary of the conversion process and saving the report as `conversion_report.json`.

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
