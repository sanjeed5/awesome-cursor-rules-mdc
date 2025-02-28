# Cursor Rules CLI Tool Plan

## Overview
A command-line tool that scans a user's project directory to identify libraries and frameworks in use, then suggests relevant Cursor rules (.mdc files) for download from the awesome-cursor-rules-mdc repository.

## Core Features
1. **Project Scanning**: Detect libraries/frameworks used in the project
2. **Rule Matching**: Match detected libraries with available MDC rules
3. **Interactive Selection**: Allow users to select which rules to download
4. **Installation**: Download and install selected rules to `.cursor/rules` directory
5. **Security & Trust**: Implement measures to ensure user safety and build trust

## Technical Implementation

### Project Structure
```
cursor-rules-cli/
├── src/
│   ├── __init__.py
│   ├── main.py            # Entry point
│   ├── scanner.py         # Project scanning logic
│   ├── matcher.py         # Library matching logic
│   ├── downloader.py      # Rule downloading functionality
│   ├── installer.py       # Rule installation to .cursor/rules
│   └── utils.py           # Utility functions
├── data/
│   └── libraries.json     # Local copy of libraries mapping
├── tests/                 # Unit tests
├── setup.py               # Package configuration
├── README.md              # Documentation
└── LICENSE                # License file
```

### Workflow
1. **Scanning Phase**:
   - Scan project for package manager files (package.json, requirements.txt, etc.)
   - Check import statements in source files
   - Identify key framework-specific files/patterns
   - Create a list of detected libraries

2. **Matching Phase**:
   - Fetch latest rules.json from GitHub or use local copy if unavailable
   - Cross-reference detected libraries with available MDC rules
   - Generate a list of relevant rules

3. **Selection Phase**:
   - Present an interactive CLI interface showing available rules
   - Allow users to select all, none, or specific rules
   - Provide brief descriptions of what each rule does

4. **Installation Phase**:
   - Create `.cursor/rules` directory if it doesn't exist
   - Download selected rule files from the repository
   - Display summary of downloaded rules

### Security Considerations
1. **Transparency**:
   - Show preview of rule content before installation
   - Clearly indicate source URLs for downloads
   - Open source the tool itself for inspection

2. **Verification**:
   - Verify file integrity with checksums
   - Only download from trusted sources
   - Implement rate limiting to avoid API abuse

3. **User Control**:
   - Always require explicit confirmation before downloading/installing
   - Provide dry-run option to see what would happen without making changes
   - Allow users to specify custom rule sources

### Distribution
1. Package as a Python package installable via pip/uv
2. Publish to PyPI
3. Provide standalone executables for major platforms

## Next Steps
1. Develop core scanning functionality
2. Implement library matching logic
3. Create interactive CLI interface
4. Build downloading and installation components
5. Add security features
6. Write tests and documentation
7. Package and publish
