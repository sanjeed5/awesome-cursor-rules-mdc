import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from exa_py import Exa

from exa_search import get_library_best_practices_exa
from logger import logger
from mdc import generate_mdc_rules_from_exa
from models import LibraryInfo, MDCRule


def initialize_exa_client() -> Optional[Exa]:
    """Initialize the Exa client with API key from environment."""
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        logger.warning("EXA_API_KEY not found in environment variables")
        return None

    try:
        return Exa(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Exa client: {str(e)}")
        return None


def create_mdc_file(rule: MDCRule, output_path: Path) -> None:
    """Create a single .mdc file from a rule."""
    # Clean up the content by removing yaml markers and extra formatting
    content = rule.content.strip()
    content = content.replace("```yaml", "")
    content = content.replace("```", "")
    content = content.strip()

    # Remove any nested frontmatter
    if content.startswith("---"):
        try:
            second_marker = content.find("---", 3)
            if second_marker != -1:
                content = content[second_marker + 3 :].strip()
        except Exception as e:
            logger.warning(f"Error removing nested frontmatter: {str(e)}")

    mdc_content = f"""---
description: {rule.description}
globs: {rule.glob_pattern}
---
{content}"""

    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mdc_content)
    logger.info(f"Created {output_path}")


def process_single_library(
    library_info: Dict[str, Any], output_dir: str, exa_client: Optional[Exa] = None
) -> Tuple[str, bool]:
    """Process a single library and generate MDC file. Returns (library_key, success)."""
    library_name = library_info["name"]
    library_tags = library_info["tags"]

    # Create a unique key for this library - using just the name for flat structure
    library_key = library_name

    logger.info(f"Processing library: {library_name} (Tags: {', '.join(library_tags)})")

    try:
        # Create output directory structure
        output_path = Path(output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # Create library info object
        library_obj = LibraryInfo(name=library_name, tags=library_tags)

        # Get best practices using Exa
        logger.info(f"Getting best practices for {library_name} using Exa")
        exa_results = get_library_best_practices_exa(
            library_name, exa_client, library_tags
        )

        # Store citations in library info
        library_obj.citations = exa_results.get("citations", [])

        # Generate MDC rules directly from Exa results
        logger.info(
            f"Generating MDC rules for {library_name} directly from Exa results"
        )
        rules = generate_mdc_rules_from_exa(library_obj, exa_results)

        # Save each rule to a file
        for rule in rules:
            # Create safe filename
            safe_name = re.sub("[^a-z0-9-]+", "-", library_name.lower()).strip("-")
            mdc_filename = f"{safe_name}.mdc"

            # Determine output path - directly in the output directory
            output_file = output_path / mdc_filename

            # Create MDC file
            create_mdc_file(rule, output_file)

        logger.info(f"Successfully processed {library_name}")
        return (library_key, True)

    except Exception as e:
        logger.error(f"Error processing library {library_name}: {str(e)}")
        return (library_key, False)
