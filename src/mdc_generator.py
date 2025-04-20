import concurrent.futures
import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from exa_py import Exa

from config import CONFIG
from exa_search import get_library_best_practices_exa
from logger import logger
from models import LibraryInfo, MDCRule
from progress_tracker import ProgressTracker


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


def process_rules_json(
    json_path: str,
    output_dir: str,
    test_mode: bool = False,
    specific_library: Optional[str] = None,
    specific_tag: Optional[str] = None,
    retry_failed_only: bool = CONFIG.processing.retry_failed_only,
):
    """Process rules.json and generate MDC files."""
    # Check if rules.json exists
    json_path = Path(json_path)
    if not json_path.exists():
        logger.error(f"Rules JSON file not found at {json_path}")
        return

    # Load rules.json
    with open(json_path, "r") as f:
        libraries_data = json.load(f)

    # Check if the new flat structure is used
    if "libraries" not in libraries_data:
        logger.error(
            "The rules.json file does not use the expected flat structure with tags."
        )
        return

    # Create progress tracker
    progress_tracker = ProgressTracker()

    # Initialize Exa client
    exa_client = initialize_exa_client()
    if exa_client is None:
        logger.warning(
            "Exa client initialization failed. Will generate content using LLM only."
        )

    # Create output directory
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Get libraries list
    libraries = libraries_data["libraries"]

    # Extract all library names for progress tracking
    all_library_names = [lib["name"] for lib in libraries]

    # Check for new libraries and update progress tracker
    progress_tracker.update_progress_with_new_libraries(all_library_names)

    # Filter libraries based on target tag or library if specified
    if specific_library:
        filtered_libraries = [
            lib for lib in libraries if lib["name"] == specific_library
        ]
    elif specific_tag:
        filtered_libraries = [lib for lib in libraries if specific_tag in lib["tags"]]
    else:
        filtered_libraries = libraries

    # If test mode is enabled, just process one library
    if test_mode:
        if filtered_libraries:
            test_library = filtered_libraries[0]
            library_key, success = process_single_library(
                test_library, output_dir, exa_client
            )
            if success:
                progress_tracker.mark_library_completed(library_key)
            else:
                progress_tracker.mark_library_failed(library_key)
        else:
            logger.error("No libraries found to process in test mode")
        return

    # Prepare list of libraries to process
    libraries_to_process = []

    # If retry_failed_only is True, only process libraries that have failed before
    # and any new libraries that have been added
    if retry_failed_only:
        failed_libraries = progress_tracker.get_failed_libraries()
        new_libraries = progress_tracker.identify_new_libraries(all_library_names)
        target_libraries = set(failed_libraries).union(set(new_libraries))

        if failed_libraries:
            logger.info(f"Retrying {len(failed_libraries)} failed libraries")
        if new_libraries:
            logger.info(f"Processing {len(new_libraries)} new libraries")

        for library in filtered_libraries:
            library_key = library["name"]
            if library_key in target_libraries:
                libraries_to_process.append(library)
    else:
        # Normal processing - skip completed libraries
        for library in filtered_libraries:
            library_key = library["name"]

            # Skip if already processed
            if progress_tracker.is_library_processed(library_key):
                logger.info(f"Skipping already processed library: {library['name']}")
                continue

            # Add to processing list
            libraries_to_process.append(library)

    if not libraries_to_process:
        logger.info(
            "No libraries to process. All libraries may have been completed already."
        )
        return

    # Process libraries in parallel
    max_workers = CONFIG.processing.max_workers
    logger.info(
        f"Processing {len(libraries_to_process)} libraries with {max_workers} workers"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with fixed arguments
        process_func = partial(
            process_single_library, output_dir=output_dir, exa_client=exa_client
        )

        # Submit all tasks
        future_to_library = {
            executor.submit(process_func, library): library
            for library in libraries_to_process
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_library):
            library = future_to_library[future]
            try:
                library_key, success = future.result()
                if success:
                    progress_tracker.mark_library_completed(library_key)
                else:
                    progress_tracker.mark_library_failed(library_key)
            except Exception as e:
                logger.error(f"Exception processing {library['name']}: {str(e)}")
                progress_tracker.mark_library_failed(library["name"])


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
