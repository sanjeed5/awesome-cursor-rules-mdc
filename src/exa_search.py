import json
import re
from typing import Any, Dict, List, Optional

from exa_py import Exa
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential

from config import CONFIG
from constants import SCRIPT_DIR
from logger import logger


@retry(
    stop=stop_after_attempt(CONFIG.exa_api.max_retries),
    wait=wait_exponential(
        multiplier=1,
        min=CONFIG.exa_api.retry_min_wait,
        max=CONFIG.exa_api.retry_max_wait,
    ),
)
@sleep_and_retry
@limits(
    calls=CONFIG.exa_api.rate_limit_calls,
    period=CONFIG.exa_api.rate_limit_period,
)
def search_for_best_practices(
    library_name: str,
    exa_client: Optional[Exa] = None,
    tags: List[str] = None,
) -> Dict[str, Any]:
    """Use Exa to search for best practices for a given library."""

    if exa_client is None:
        logger.warning("Exa client not initialized, falling back to LLM generation")
        return {"answer": "", "citations": []}

    try:
        if tags and len(tags) > 0:
            # Select up to 3 most relevant tags to keep the query focused
            relevant_tags = tags[:3] if len(tags) > 3 else tags
            tags_str = " ".join(relevant_tags)
            query = f"{library_name} best practices coding standards for {tags_str} development"
        else:
            query = f"{library_name} best practices coding standards"

        logger.info(f"Searching Exa for: {query}")

        # Call Exa API
        result = exa_client.answer(query, text=True)
        logger.info(f"Exa search results: {result}")
        return result

    except Exception as e:
        logger.error(f"Error searching Exa for {library_name}: {str(e)}")
        return {"answer": "", "citations": []}


def save_exa_results_to_file(result: object, library_name: str):
    """Save Exa search results to a file."""
    result_dict = {
        "answer": result.answer if hasattr(result, "answer") else "",
        "citations": result.citations if hasattr(result, "citations") else [],
    }

    exa_results_dir = SCRIPT_DIR / CONFIG.paths.exa_results_dir
    exa_results_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub("[^a-z0-9-]+", "-", library_name.lower()).strip("-")
    results_file = exa_results_dir / f"{safe_name}_exa_result.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, default=str)

    logger.info(f"Saved Exa search results for {library_name} to {results_file}")

    if result_dict["answer"]:
        logger.info(f"Successfully retrieved Exa results for {library_name}")
        return result_dict
    else:
        logger.warning(f"No results found via Exa for {library_name}")
        return {"answer": "", "citations": []}


def get_library_best_practices_exa(
    library_name: str, exa_client: Optional[Exa] = None, tags: List[str] = None
) -> Dict[str, Any]:
    """Get best practices for a given library using Exa."""

    result = search_for_best_practices(library_name, exa_client, tags)
    save_exa_results_to_file(result, library_name)
    return result
