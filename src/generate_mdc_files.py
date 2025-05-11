import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import datetime
from dotenv import load_dotenv
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
from pydantic import BaseModel, Field
import re
import argparse
from exa_py import Exa
import concurrent.futures
from functools import partial
import yaml
import sys
import shutil

# Enable JSON schema validation in LiteLLM
import litellm
litellm.enable_json_schema_validation = True

# Get the script directory for absolute path resolution
SCRIPT_DIR = Path(__file__).parent.absolute()

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = SCRIPT_DIR / "logs"
logs_dir.mkdir(exist_ok=True)

# Create a timestamp for the log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = logs_dir / f"mdc_generation_{timestamp}.log"

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Starting MDC generation. Logs will be saved to {log_file}")

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_CONFIG = {
    "paths": {
        "mdc_instructions": "mdc-instructions.txt",
        "rules_json": "../rules.json",
        "output_dir": "../rules-mdc",
        "exa_results_dir": "exa_results"  # Directory to store Exa results
    },
    "api": {
        "llm_model": "gemini/gemini-2.0-flash",
        "rate_limit_calls": 2000,
        "rate_limit_period": 60,
        "max_retries": 3,
        "retry_min_wait": 4,
        "retry_max_wait": 10
    },
    "exa_api": {
        "rate_limit_calls": 5,
        "rate_limit_period": 1,
        "max_retries": 3,
        "retry_min_wait": 1,
        "retry_max_wait": 5
    },
    "processing": {
        "max_workers": 4,
        "chunk_size": 50000,
        "retry_failed_only": True  # Default to retry only failed libraries
    }
}

# Load or create configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml or create default if not exists."""
    config_path = SCRIPT_DIR / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info("Loaded configuration from config.yaml")
                # Merge with defaults to ensure all required keys exist
                merged_config = DEFAULT_CONFIG.copy()
                for section, values in config.items():
                    if section in merged_config and isinstance(values, dict):
                        merged_config[section].update(values)
                    else:
                        merged_config[section] = values
                return merged_config
        except Exception as e:
            logger.warning(f"Error loading config.yaml: {str(e)}. Using default configuration.")
            return DEFAULT_CONFIG
    else:
        # Create default config file
        try:
            with open(config_path, "w") as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
            logger.info("Created default configuration file config.yaml")
        except Exception as e:
            logger.warning(f"Could not create config.yaml: {str(e)}")
        return DEFAULT_CONFIG

# Global configuration
CONFIG = load_config()

class MDCRule(BaseModel):
    """Model for MDC rule data."""
    name: str = Field(..., description="A meaningful name for the rule that reflects its purpose")
    glob_pattern: str = Field(..., description="Glob pattern to specify which files/folders the rule applies to")
    description: str = Field(..., description="A clear description of what the rule does and when it should be applied")
    content: str = Field(..., description="The actual rule content")

class LibraryInfo(BaseModel):
    """Model for library information."""
    name: str
    tags: List[str]
    best_practices: str = ""
    citations: List[Dict[str, Any]] = []

class ProgressTracker:
    """Track progress of MDC generation to allow resuming."""
    def __init__(self, tracking_file: str = "mdc_generation_progress.json"):
        self.tracking_file = SCRIPT_DIR / tracking_file
        self.progress: Dict[str, str] = self._load_progress()
        logger.debug(f"Loaded progress tracker with {len(self.progress)} entries")
    
    def _load_progress(self) -> Dict[str, str]:
        """Load existing progress from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    progress = json.load(f)
                    logger.debug(f"Loaded progress from {self.tracking_file}")
                    return progress
            except json.JSONDecodeError:
                logger.warning("Progress file corrupted, starting fresh")
                return {}
        logger.debug(f"Progress file {self.tracking_file} not found, starting fresh")
        return {}
    
    def save_progress(self):
        """Save current progress to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def is_library_processed(self, library_key: str) -> bool:
        """Check if a library has been successfully processed."""
        return self.progress.get(library_key) == "completed"
    
    def is_library_failed(self, library_key: str) -> bool:
        """Check if a library has failed processing."""
        return self.progress.get(library_key) == "failed"
    
    def mark_library_completed(self, library_key: str):
        """Mark a library as successfully processed."""
        self.progress[library_key] = "completed"
        self.save_progress()
    
    def mark_library_failed(self, library_key: str):
        """Mark a library as failed."""
        self.progress[library_key] = "failed"
        self.save_progress()
    
    def get_failed_libraries(self) -> List[str]:
        """Get a list of failed libraries."""
        return [key for key, value in self.progress.items() if value == "failed"]
    
    def identify_new_libraries(self, all_libraries: List[str]) -> List[str]:
        """Identify libraries that are not yet in the progress tracker."""
        new_libraries = [lib for lib in all_libraries if lib not in self.progress]
        logger.debug(f"Found {len(new_libraries)} new libraries out of {len(all_libraries)} total libraries")
        if new_libraries:
            logger.debug(f"New libraries: {', '.join(new_libraries)}")
        return new_libraries
    
    def update_progress_with_new_libraries(self, all_libraries: List[str]) -> List[str]:
        """
        Update progress tracker with new libraries and return the list of newly added libraries.
        New libraries are marked as not processed (not in the progress tracker).
        """
        new_libraries = self.identify_new_libraries(all_libraries)
        if new_libraries:
            logger.info(f"Found {len(new_libraries)} new libraries to process: {', '.join(new_libraries)}")
        return new_libraries

class MDCRuleResponse(BaseModel):
    """Pydantic model for LLM response structure."""
    name: str = Field(..., description="Short descriptive name for the rule")
    glob_pattern: str = Field(..., description="Valid glob pattern for target files")
    description: str = Field(..., description="1-2 sentence description of what the rule does")
    content: str = Field(..., description="Formatted rule content using markdown")

def validate_environment_variables() -> bool:
    """Validate that all required environment variables are set."""
    required_vars = []
    
    # Check for LiteLLM API key (depends on the model being used)
    model = CONFIG["api"]["llm_model"]
    if model.startswith("gemini"):
        required_vars.append("GEMINI_API_KEY")
    elif model.startswith("gpt") or model.startswith("openai"):
        required_vars.append("OPENAI_API_KEY")
    elif model.startswith("claude") or model.startswith("anthropic"):
        required_vars.append("ANTHROPIC_API_KEY")
    
    # Check for Exa API key
    required_vars.append("EXA_API_KEY")
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

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

@retry(stop=stop_after_attempt(CONFIG["exa_api"]["max_retries"]), 
       wait=wait_exponential(multiplier=1, min=CONFIG["exa_api"]["retry_min_wait"], max=CONFIG["exa_api"]["retry_max_wait"]))
@sleep_and_retry
@limits(calls=CONFIG["exa_api"]["rate_limit_calls"], period=CONFIG["exa_api"]["rate_limit_period"])
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
        exa_results_dir = SCRIPT_DIR / CONFIG["paths"]["exa_results_dir"]
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

@sleep_and_retry
@limits(calls=CONFIG["api"]["rate_limit_calls"], period=CONFIG["api"]["rate_limit_period"])
@retry(stop=stop_after_attempt(CONFIG["api"]["max_retries"]), 
       wait=wait_exponential(multiplier=1, min=CONFIG["api"]["retry_min_wait"], max=CONFIG["api"]["retry_max_wait"]))
def generate_mdc_rules_from_exa(library_info: LibraryInfo, exa_results: Dict[str, Any]) -> List[MDCRule]:
    """Generate MDC rules for a library using LLM directly from Exa results."""
    try:
        # Load MDC instructions
        mdc_instructions_path = SCRIPT_DIR / CONFIG["paths"]["mdc_instructions"]
        if not mdc_instructions_path.exists():
            logger.warning(f"MDC instructions file not found at {mdc_instructions_path}")
            mdc_instructions = "Create rules with clear descriptions and appropriate glob patterns."
        else:
            try:
                mdc_instructions = mdc_instructions_path.read_text()
            except Exception as e:
                logger.warning(f"Could not read MDC instructions file: {str(e)}")
                mdc_instructions = "Create rules with clear descriptions and appropriate glob patterns."
        
        # Extract Exa answer and citations
        exa_answer = exa_results.get("answer", "")
        citations = exa_results.get("citations", [])
        
        # Extract text from citations
        citation_texts = []
        for citation in citations:
            if isinstance(citation, dict) and citation.get("text"):
                citation_texts.append(citation.get("text", ""))
            elif hasattr(citation, 'text') and citation.text:
                citation_texts.append(citation.text)
        
        # Combine all citation texts
        all_citation_text = "\n\n".join(citation_texts)
        
        # Format tags for prompt
        tags_str = ", ".join(library_info.tags)
        
        # Enhanced prompt template
        enhanced_prompt_template = """Create a comprehensive Cursor rule file (.mdc) for the {library_name} library following these guidelines:

{mdc_instructions}

Library Information:
- Name: {library_name}
- Tags: {tags}

{exa_content_section}

Your task is to create an EXTREMELY DETAILED and COMPREHENSIVE guide that covers:

1. Code Organization and Structure
2. Common Patterns and Anti-patterns
3. Performance Considerations
4. Security Best Practices
5. Testing Approaches
6. Common Pitfalls and Gotchas
7. Tooling and Environment

You MUST format your response as a valid JSON object with these exact keys:
- name: A short descriptive name for the rule (e.g., "{library_name} Best Practices")
- glob_pattern: The most appropriate glob pattern for this library based on the file types it typically works with
- description: A clear 1-2 sentence description of what the rule covers
- content: The formatted rule content with comprehensive best practices in markdown format

DO NOT include any text outside of the JSON object. The response must be a single, valid JSON object."""
        
        # Determine if we need to generate content from scratch
        if len(exa_answer.strip()) < 100 and len(all_citation_text.strip()) < 200:
            exa_content_section = f"""I need you to research and generate comprehensive best practices for {library_info.name} from your knowledge.

Please be extremely thorough and detailed, covering all aspects of {library_info.name} development.
Your guidance should be useful for both beginners and experienced developers."""
        else:
            chunk_size = CONFIG["processing"]["chunk_size"]
            exa_content_section = f"""Based on the following information about {library_info.name} best practices:

Exa search results:
{exa_answer}

Additional information from citations:
{all_citation_text[:chunk_size]}

Please synthesize, enhance, and expand upon this information to create the most comprehensive guide possible.
Add any important best practices that might be missing from the search results."""
        
        prompt = enhanced_prompt_template.format(
            library_name=library_info.name,
            tags=tags_str,
            exa_content_section=exa_content_section,
            mdc_instructions=mdc_instructions
        )
        
        logger.info(f"Sending enhanced prompt to LLM for {library_info.name}")
        
        # Use LiteLLM's completion with JSON mode
        response = completion(
            model=CONFIG["api"]["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=4000,
            temperature=0.7
        )
        
        # Get the response content and parse it
        try:
            if hasattr(response.choices[0].message, 'content'):
                # Standard response format
                content = response.choices[0].message.content
            elif hasattr(response.choices[0].message, 'text'):
                # Gemini format
                content = response.choices[0].message.text
            else:
                raise ValueError("Unexpected response format from LLM")

            # Clean the content to ensure it's valid JSON
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Parse the JSON response
            json_response = json.loads(content)
            
            # Validate against our schema
            rule_data = MDCRuleResponse(**json_response)
            
            # Create MDCRule instance using model_dump instead of dict
            rule = MDCRule(**rule_data.model_dump())
            logger.info(f"Successfully generated enhanced rule for {library_info.name}")
            
            return [rule]
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for {library_info.name}: {str(e)}\nResponse content: {content}")
            raise
        except Exception as e:
            logger.error(f"Error processing response for {library_info.name}: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Error generating MDC rule for {library_info.name}: {str(e)}")
        raise

def create_mdc_file(rule: MDCRule, output_path: Path) -> None:
    """Create a single .mdc file from a rule."""
    # Clean up the content by removing yaml markers and extra formatting
    content = rule.content.strip()
    content = content.replace('```yaml', '')
    content = content.replace('```', '')
    content = content.strip()
    
    # Remove any nested frontmatter
    if content.startswith('---'):
        try:
            second_marker = content.find('---', 3)
            if second_marker != -1:
                content = content[second_marker + 3:].strip()
        except Exception as e:
            logger.warning(f"Error removing nested frontmatter: {str(e)}")
    
    mdc_content = f"""---
description: {rule.description}
globs: {rule.glob_pattern}
---
{content}"""
    
    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(mdc_content)
    logger.info(f"Created {output_path}")

def process_single_library(library_info: Dict[str, Any], output_dir: str, exa_client: Optional[Exa] = None) -> Tuple[str, bool]:
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
        library_obj = LibraryInfo(
            name=library_name,
            tags=library_tags
        )
        
        # Get best practices using Exa
        logger.info(f"Getting best practices for {library_name} using Exa")
        exa_results = get_library_best_practices_exa(library_name, exa_client, library_tags)
        
        # Store citations in library info
        library_obj.citations = exa_results.get("citations", [])
        
        # Generate MDC rules directly from Exa results
        logger.info(f"Generating MDC rules for {library_name} directly from Exa results")
        rules = generate_mdc_rules_from_exa(library_obj, exa_results)
        
        # Save each rule to a file
        for rule in rules:
            # Create safe filename
            safe_name = re.sub('[^a-z0-9-]+', '-', library_name.lower()).strip('-')
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

def process_rules_json(json_path: str, output_dir: str, test_mode: bool = False,
                      specific_library: Optional[str] = None, specific_tag: Optional[str] = None,
                      retry_failed_only: bool = False):
    """Process rules.json and generate MDC files."""
    # Check if rules.json exists
    # Use absolute paths for clarity and robustness
    rules_json_path_abs = (SCRIPT_DIR / json_path).resolve() # Ensure path is resolved from SCRIPT_DIR

    if not rules_json_path_abs.exists():
        logger.error(f"The rules.json file was not found at {rules_json_path_abs}")
        return

    logger.info(f"Processing rules.json from: {rules_json_path_abs}")
    try:
        with open(rules_json_path_abs, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {rules_json_path_abs}")
        return
    except Exception as e:
        logger.error(f"Error reading {rules_json_path_abs}: {str(e)}")
        return

    # Validate the structure of 'data' before attempting to sort
    if "libraries" not in data:
        logger.error(f"The rules.json file ({rules_json_path_abs}) does not contain the required 'libraries' key.")
        return
    if not isinstance(data.get("libraries"), list):
        logger.error(f"The 'libraries' key in {rules_json_path_abs} is not a list.")
        return

    # Sort the list in place
    logger.info(f"Sorting libraries in {rules_json_path_abs} alphabetically by name.")
    data["libraries"].sort(key=lambda lib: lib.get("name", "").lower()) # Handles missing 'name' gracefully

    # Write the modified dictionary back to the file
    try:
        with open(rules_json_path_abs, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully sorted and saved {rules_json_path_abs}")
    except Exception as e:
        logger.error(f"Error writing sorted data back to {rules_json_path_abs}: {str(e)}")
        return # Critical to stop if write fails

    # 'data' in memory and the file on disk are now sorted.
    # Subsequent code will use data.get("libraries", []) or data["libraries"].
    
    # Check if the list is empty and log a warning if so.
    # This check is on the list itself, obtained from the (potentially) modified 'data'.
    if not data.get("libraries", []): # Safely get the list, defaults to empty if key somehow vanished (shouldn't happen)
        logger.warning(f"The 'libraries' list in {rules_json_path_abs} is empty. No libraries to process.")
        # Script will continue with an empty list of libraries.

    libraries_data = data.get("libraries", [])

    # Create progress tracker
    progress_tracker = ProgressTracker()
    
    # Initialize Exa client
    exa_client = initialize_exa_client()
    if exa_client is None:
        logger.warning("Exa client initialization failed. Will generate content using LLM only.")
    
    # Create output directory
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Get libraries list
    libraries = libraries_data
    
    # Extract all library names for progress tracking
    all_library_names = [lib["name"] for lib in libraries]
    
    # Check for new libraries and update progress tracker
    progress_tracker.update_progress_with_new_libraries(all_library_names)
    
    # Filter libraries based on target tag or library if specified
    if specific_library:
        # When specific library is provided, force process it regardless of progress status
        filtered_libraries = [lib for lib in libraries if lib["name"] == specific_library]
        if not filtered_libraries:
            logger.error(f"Library '{specific_library}' not found in rules.json")
            return
        libraries_to_process = filtered_libraries
    elif specific_tag:
        filtered_libraries = [lib for lib in libraries if specific_tag in lib["tags"]]
        # For tag filtering, apply normal progress tracking rules
        libraries_to_process = []
        for library in filtered_libraries:
            library_key = library["name"]
            if not progress_tracker.is_library_processed(library_key) or retry_failed_only and progress_tracker.is_library_failed(library_key):
                libraries_to_process.append(library)
    else:
        # Normal processing - apply progress tracking rules
        libraries_to_process = []
        for library in libraries:
            library_key = library["name"]
            if not progress_tracker.is_library_processed(library_key) or retry_failed_only and progress_tracker.is_library_failed(library_key):
                libraries_to_process.append(library)
    
    # If test mode is enabled, just process one library
    if test_mode:
        if libraries_to_process:
            test_library = libraries_to_process[0]
            library_key, success = process_single_library(test_library, output_dir, exa_client)
            if success:
                progress_tracker.mark_library_completed(library_key)
            else:
                progress_tracker.mark_library_failed(library_key)
        else:
            logger.error("No libraries found to process in test mode")
        return
    
    if not libraries_to_process:
        if specific_library:
            logger.error(f"Library '{specific_library}' was not found or could not be processed")
        else:
            logger.info("No libraries to process. All libraries may have been completed already.")
        return
    
    # Process libraries in parallel
    max_workers = CONFIG["processing"]["max_workers"]
    logger.info(f"Processing {len(libraries_to_process)} libraries with {max_workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with fixed arguments
        process_func = partial(process_single_library, output_dir=output_dir, exa_client=exa_client)
        
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
                progress_tracker.mark_library_failed(library['name'])

def main():
    """Main function to run the MDC generation process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate MDC files from rules.json")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only one library)")
    parser.add_argument("--tag", type=str, help="Process only libraries with a specific tag")
    parser.add_argument("--library", type=str, help="Process only a specific library")
    parser.add_argument("--output", type=str, default=None, help="Output directory for MDC files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--workers", type=int, help=f"Number of parallel workers (default: {CONFIG['processing']['max_workers']})")
    parser.add_argument("--rate-limit", type=int, help=f"API rate limit calls per minute (default: {CONFIG['api']['rate_limit_calls']})")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save log files")
    parser.add_argument("--exa-results-dir", type=str, default=None, help="Directory to save Exa results")
    parser.add_argument("--regenerate-all", action="store_true", help="Regenerate all libraries, including previously completed ones")
    parser.add_argument("--debug", action="store_true", help="Enable extra debug output")
    args = parser.parse_args()
    
    # Set up verbose logging if requested
    if args.verbose or args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Override config with command line arguments
    if args.workers:
        CONFIG["processing"]["max_workers"] = args.workers
    
    if args.rate_limit:
        CONFIG["api"]["rate_limit_calls"] = args.rate_limit
    
    if args.exa_results_dir:
        CONFIG["paths"]["exa_results_dir"] = args.exa_results_dir
    
    # Set paths with absolute paths
    rules_json_path = SCRIPT_DIR / Path(CONFIG["paths"]["rules_json"])
    output_dir = args.output if args.output else SCRIPT_DIR / Path(CONFIG["paths"]["output_dir"])
    
    # Check if rules.json exists
    if not rules_json_path.exists():
        logger.error(f"rules.json not found at {rules_json_path}")
        return
    
    # Debug output
    if args.debug:
        logger.debug(f"Rules JSON path: {rules_json_path}")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"Config: {CONFIG}")
    
    # Validate environment variables
    if not validate_environment_variables():
        logger.error("Missing required environment variables. Exiting.")
        sys.exit(1)
    
    try:
        # Process rules.json and generate MDC files
        process_rules_json(
            rules_json_path, 
            output_dir, 
            test_mode=args.test,
            specific_library=args.library,
            specific_tag=args.tag,
            retry_failed_only=not args.regenerate_all  # Invert regenerate-all flag
        )
        logger.info("MDC generation completed successfully!")
        
        # Copy rules.json to cursor-rules-cli folder
        # This is the source of truth - the CLI package will use this during setup
        cursor_rules_cli_dir = Path(SCRIPT_DIR).parent / "cursor-rules-cli"
        cursor_rules_cli_dir.mkdir(exist_ok=True)
        
        target_rules_json = cursor_rules_cli_dir / "rules.json"
        
        try:
            shutil.copy2(rules_json_path, target_rules_json)
            logger.info(f"Successfully copied rules.json (source of truth) to {target_rules_json}")
            
            # Add a comment file to explain the source of truth
            with open(cursor_rules_cli_dir / "RULES_JSON_README.md", "w") as f:
                f.write("""# About rules.json

This rules.json file is automatically copied from the project root.
It is the source of truth for library detection rules.

DO NOT EDIT THIS FILE DIRECTLY.
Instead, edit the main rules.json in the project root and run the generate_mdc_files.py script.
""")
            logger.info("Created RULES_JSON_README.md to document the source of truth")
            
        except Exception as e:
            logger.error(f"Failed to copy rules.json to cursor-rules-cli folder: {str(e)}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 