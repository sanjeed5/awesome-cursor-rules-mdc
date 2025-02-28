# Publishing to PyPI

This document provides instructions for publishing the `cursor-rules` package to PyPI.

## Prerequisites

1. Create a PyPI account at https://pypi.org/account/register/
2. Generate an API token at https://pypi.org/manage/account/token/
3. Install required tools:
   ```bash
   pip install build twine
   ```

## Publishing Steps

1. Update the version in `src/__init__.py`

2. Build the package:
   ```bash
   python -m build
   ```

3. Test the package locally:
   ```bash
   pip install --force-reinstall dist/cursor_rules-*.whl
   cursor-rules --help
   ```

4. Upload to TestPyPI (optional):
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

5. Install from TestPyPI (optional):
   ```bash
   pip install --index-url https://test.pypi.org/simple/ cursor-rules
   ```

6. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

## Using API Tokens

When using twine, you can either:

1. Create a `.pypirc` file in your home directory:
   ```
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = your_pypi_token

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = your_testpypi_token
   ```

2. Or provide credentials via environment variables:
   ```bash
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=your_pypi_token
   ```

3. Or enter them when prompted by twine.

## Updating the Package

1. Make your changes
2. Update the version in `src/__init__.py`
3. Rebuild and upload following the steps above 