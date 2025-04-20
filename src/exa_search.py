from logger import logger
from config import CONFIG
from exa_py import Exa
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
from typing import Dict, Any, Optional, List
import json
import re
from constants import SCRIPT_DIR

@retry(stop=stop_after_attempt(CONFIG.exa_api.max_retries), 
       wait=wait_exponential(multiplier=1, min=CONFIG.exa_api.retry_min_wait, max=CONFIG.exa_api.retry_max_wait))
@sleep_and_retry
@limits(calls=CONFIG.exa_api.rate_limit_calls, period=CONFIG.exa_api.rate_limit_period)
def get_library_best_practices_exa(library_name: str, exa_client: Optional[Exa] = None, tags: List[str] = None) -> Dict[str, Any]:
    """Use Exa to search for best practices for a library."""
    if exa_client is None:
        logger.warning("Exa client not initialized, falling back to LLM generation")
        return {"answer": "", "citations": []}
    
    try:
        # Construct a query for best practices that includes tags for better context
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
        
        # Convert the AnswerResponse object to a dictionary
        result_dict = {
            "answer": result.answer if hasattr(result, 'answer') else "",
            "citations": result.citations if hasattr(result, 'citations') else []
        }
        
        # Save the Exa result to a file
        exa_results_dir = SCRIPT_DIR / CONFIG.paths.exa_results_dir
        exa_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a safe filename
        safe_name = re.sub('[^a-z0-9-]+', '-', library_name.lower()).strip('-')
        result_file = exa_results_dir / f"{safe_name}_exa_result.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"Saved Exa result for {library_name} to {result_file}")
        
        if result_dict["answer"]:
            logger.info(f"Successfully retrieved Exa results for {library_name}")
            return result_dict
        
        # If no results found, return empty
        logger.warning(f"No results found via Exa for {library_name}")
        return {"answer": "", "citations": []}
        
    except Exception as e:
        logger.error(f"Error fetching from Exa for {library_name}: {str(e)}")
        return {"answer": "", "citations": []}
