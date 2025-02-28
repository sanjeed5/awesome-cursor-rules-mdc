#!/usr/bin/env python
"""
Setup script for cursor-rules.
"""

from setuptools import setup, find_packages
import os
import shutil
from pathlib import Path

# Read version from __init__.py
with open(os.path.join("src", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.1.0"

# Read long description from README.md
long_description = "A CLI tool to scan projects and install relevant Cursor rules (.mdc files)."
readme_path = Path("README.md")
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Copy rules.json to src directory if it doesn't exist
rules_json_src = Path("rules.json")
rules_json_dest = Path("src/rules.json")
if rules_json_src.exists() and not rules_json_dest.exists():
    shutil.copy2(rules_json_src, rules_json_dest)
    print(f"Copied rules.json to {rules_json_dest}")

setup(
    name="cursor-rules",
    version=version,
    description="A CLI tool to scan projects and install relevant Cursor rules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="sanjeed5",
    author_email="hi@sanjeed.in",
    url="https://github.com/sanjeed5/awesome-cursor-rules-mdc",
    package_dir={"cursor_rules_cli": "src"},
    packages=["cursor_rules_cli"],
    include_package_data=True,
    package_data={
        "cursor_rules_cli": ["*.json"],
    },
    entry_points={
        "console_scripts": [
            "cursor-rules=cursor_rules_cli.main:main",
        ],
    },
    python_requires=">=3.8",
    keywords="cursor, rules, mdc, cli",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        "requests>=2.25.0",
        "colorama>=0.4.4",
        "tqdm>=4.62.0",
        "urllib3>=2.0.0",
        "validators>=0.20.0",
    ]
) 