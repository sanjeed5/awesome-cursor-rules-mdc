import concurrent.futures
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from exa_py import Exa

from config import CONFIG
from exa_search import get_library_best_practices_exa
from logger import logger
from models import LibraryInfo
from progress_tracker import ProgressTracker

libraries_to_process = []
progress_tracker = ProgressTracker()


def initialize_exa_client() -> Optional[Exa]:
    """Initialize the Exa client with API key from environment."""
    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        logger.warning("EXA_API_KEY not found in environment variables")
        return None

    try:
        return Exa(api_key=exa_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Exa client: {str(e)}")
        return None


def process_rules_json(json_path: str):
    """Process rules.json."""
    json_path = Path(json_path)
    if not json_path.exists():
        logger.error(f"Rules JSON file not found at {json_path}")
        return

    with open(json_path, "r") as f:
        libraries_data = json.load(f)

    # Validate rules.json file structure
    if "libraries" not in libraries_data:
        logger.error(
            "The rules.json file does not use the expected flat structure with tags."
        )
        return
    else:
        return libraries_data


def prepare_for_processing(output_dir: str):
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


def prepare_libraries_for_processing(
    libraries_data: Dict[str, Any],
    specific_library: Optional[str] = None,
    specific_tag: Optional[str] = None,
):
    # Get libraries list
    libraries = libraries_data["libraries"]

    # Extract all library names for progress tracking
    all_library_names = [lib["name"] for lib in libraries]

    # Check for new libraries and update progress tracker
    progress_tracker.update_progress_with_new_libraries(all_library_names)

    # Filter libraries based on target tag or library if specified
    filtered_libraries = []
    if specific_library:
        filtered_libraries = [
            lib for lib in libraries if lib["name"] == specific_library
        ]
    elif specific_tag:
        filtered_libraries = [lib for lib in libraries if specific_tag in lib["tags"]]
    else:
        filtered_libraries = libraries

    return filtered_libraries, all_library_names


def process_in_test_mode(
    filtered_libraries: List[str],
    output_dir: str,
    exa_client: Optional[Exa] = None,
):
    """If test mode is enabled, just process one library."""
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


def retry_failed_libraries(
    filtered_libraries: List[str],
    all_library_names: List[str],
):
    """Only process libraries that have failed and any new libraries that have been added."""
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


def create_processing_list(
    filtered_libraries: List[str],
    all_library_names: List[str],
    retry_failed_only: bool = False,
):
    libraries_to_process = []

    if retry_failed_only:
        retry_failed_libraries(filtered_libraries, all_library_names)
    else:
        for library in filtered_libraries:
            library_key = library["name"]

            if progress_tracker.is_library_processed(library_key):
                logger.info(f"Skipping already processed library: {library['name']}")
                continue
            else:
                libraries_to_process.append(library)

    if not libraries_to_process:
        logger.info("No libraries remaining to process.")
        return

    return libraries_to_process


def search_exa_for_best_practices(
    library_info: Dict[str, Any],
    exa_client: Optional[Exa] = None,
) -> Tuple[str, bool]:
    """Search Exa for best practices for given library."""
    library_name = library_info["name"]
    library_tags = library_info["tags"]

    # Create a unique key for this library - using just the name for flat structure
    library_key = library_name

    logger.info(f"Processing library: {library_name} (Tags: {', '.join(library_tags)})")

    try:
        library_obj = LibraryInfo(name=library_name, tags=library_tags)

        # Get best practices using Exa
        logger.info(f"Searching Exa for {library_name} best practices")
        exa_results = get_library_best_practices_exa(
            library_name, exa_client, library_tags
        )

        library_obj.citations = exa_results.get("citations", [])

        logger.info(f"Successfully searched Exa for {library_name} best practices")
        return (library_key, True)

    except Exception as e:
        logger.error(f"Error searching Exa for {library_name} best practices: {str(e)}")
        return (library_key, False)


def process_all_libraries(
    libraries_to_process: List[Dict[str, Any]],
    output_dir: str,
    exa_client: Optional[Exa] = None,
):
    max_workers = CONFIG.processing.max_workers
    logger.info(
        f"Parallel processing {len(libraries_to_process)} libraries with {max_workers} workers"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with fixed arguments
        process_func = partial(
            search_exa_for_best_practices(
                output_dir=output_dir,
                exa_client=exa_client,
            ),
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
