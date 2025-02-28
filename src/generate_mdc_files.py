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

# Enable JSON schema validation in LiteLLM
import litellm
litellm.enable_json_schema_validation = True

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = Path("logs")
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
        "rules_json": "rules.json",
        "output_dir": "rules-mdc",
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
    "processing": {
        "max_workers": 4,
        "chunk_size": 50000
    }
}

# Load or create configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml or create default if not exists."""
    config_path = Path("config.yaml")
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
        self.tracking_file = Path(tracking_file)
        self.progress: Dict[str, str] = self._load_progress()
    
    def _load_progress(self) -> Dict[str, str]:
        """Load existing progress from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Progress file corrupted, starting fresh")
                return {}
        return {}
    
    def save_progress(self):
        """Save current progress to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def is_library_processed(self, library_key: str) -> bool:
        """Check if a library has been successfully processed."""
        return self.progress.get(library_key) == "completed"
    
    def mark_library_completed(self, library_key: str):
        """Mark a library as successfully processed."""
        self.progress[library_key] = "completed"
        self.save_progress()
    
    def mark_library_failed(self, library_key: str):
        """Mark a library as failed."""
        self.progress[library_key] = "failed"
        self.save_progress()

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

@retry(stop=stop_after_attempt(CONFIG["api"]["max_retries"]), 
       wait=wait_exponential(multiplier=1, min=CONFIG["api"]["retry_min_wait"], max=CONFIG["api"]["retry_max_wait"]))
def get_library_best_practices_exa(library_name: str, exa_client: Optional[Exa] = None) -> Dict[str, Any]:
    """Use Exa to search for best practices for a library."""
    if exa_client is None:
        logger.warning("Exa client not initialized, falling back to LLM generation")
        return {"answer": "", "citations": []}
    
    try:
        # Construct a query for best practices
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
        exa_results_dir = Path(CONFIG["paths"]["exa_results_dir"])
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

def determine_glob_pattern(library_name: str, tags: List[str]) -> str:
    """Determine the appropriate glob pattern based on library name and tags."""
    # Default glob pattern
    default_glob = "**/*"
    
    # Check for language-specific patterns
    if "python" in tags:
        return "**/*.py"
    elif "javascript" in tags or "typescript" in tags:
        if "react" in tags:
            return "**/*.{jsx,tsx}"
        elif "vue" in tags:
            return "**/*.vue"
        elif "svelte" in tags:
            return "**/*.svelte"
        elif "angular" in tags:
            return "**/*.{ts,html}"
        return "**/*.{js,ts,jsx,tsx}"
    elif "rust" in tags:
        return "**/*.rs"
    elif "go" in tags:
        return "**/*.go"
    elif "java" in tags:
        return "**/*.java"
    elif "php" in tags:
        return "**/*.php"
    elif "ruby" in tags:
        return "**/*.rb"
    
    # Check for specific libraries with known file extensions
    if library_name == "django":
        return "**/*.{py,html}"
    elif library_name == "flask":
        return "**/*.py"
    elif library_name == "fastapi":
        return "**/*.py"
    elif library_name == "pytest":
        return "**/test_*.py"
    elif library_name == "tailwind":
        return "**/*.{css,html,jsx,tsx}"
    
    # Return default if no specific pattern found
    return default_glob

@sleep_and_retry
@limits(calls=CONFIG["api"]["rate_limit_calls"], period=CONFIG["api"]["rate_limit_period"])
@retry(stop=stop_after_attempt(CONFIG["api"]["max_retries"]), 
       wait=wait_exponential(multiplier=1, min=CONFIG["api"]["retry_min_wait"], max=CONFIG["api"]["retry_max_wait"]))
def generate_mdc_rules_from_exa(library_info: LibraryInfo, exa_results: Dict[str, Any]) -> List[MDCRule]:
    """Generate MDC rules for a library using LLM directly from Exa results."""
    try:
        # Load MDC instructions
        mdc_instructions_path = Path(CONFIG["paths"]["mdc_instructions"])
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
        
        # Combine all citation texts - Gemini can handle much larger inputs
        all_citation_text = "\n\n".join(citation_texts)
        
        # Suggest a glob pattern based on library tags
        suggested_glob = determine_glob_pattern(library_info.name, library_info.tags)
        
        # Format tags for prompt
        tags_str = ", ".join(library_info.tags)
        
        # Enhanced prompt template for both cases
        enhanced_prompt_template = """Create a comprehensive Cursor rule file (.mdc) for the {library_name} library following these guidelines:

{mdc_instructions}

Library Information:
- Name: {library_name}
- Tags: {tags}
- Suggested glob pattern: {suggested_glob}

{exa_content_section}

Your task is to create an EXTREMELY DETAILED and COMPREHENSIVE guide that covers:

1. Code Organization and Structure:
   - Directory structure best practices
   - File naming conventions
   - Module organization
   - Component architecture
   - Code splitting strategies

2. Common Patterns and Anti-patterns:
   - Design patterns specific to {library_name}
   - Recommended approaches for common tasks
   - Anti-patterns and code smells to avoid
   - State management best practices
   - Error handling patterns

3. Performance Considerations:
   - Optimization techniques
   - Memory management
   - Rendering optimization
   - Bundle size optimization
   - Lazy loading strategies

4. Security Best Practices:
   - Common vulnerabilities and how to prevent them
   - Input validation
   - Authentication and authorization patterns
   - Data protection strategies
   - Secure API communication

5. Testing Approaches:
   - Unit testing strategies
   - Integration testing
   - End-to-end testing
   - Test organization
   - Mocking and stubbing

6. Common Pitfalls and Gotchas:
   - Frequent mistakes developers make
   - Edge cases to be aware of
   - Version-specific issues
   - Compatibility concerns
   - Debugging strategies

7. Tooling and Environment:
   - Recommended development tools
   - Build configuration
   - Linting and formatting
   - Deployment best practices
   - CI/CD integration

Format your response as a valid JSON object with exactly these keys:
  - name: a short descriptive name for the rule (e.g., "{library_name} Best Practices")
  - glob_pattern: the most appropriate glob pattern for this library based on the file types it typically works with (you can use the suggested pattern or improve it)
  - description: a clear 1-2 sentence description of what the rule covers
  - content: the formatted rule content with comprehensive best practices in markdown format
"""
        
        # Determine if we need to generate content from scratch
        if len(exa_answer.strip()) < 100 and len(all_citation_text.strip()) < 200:
            # Not enough content from Exa, generate from scratch
            exa_content_section = f"""I need you to research and generate comprehensive best practices for {library_info.name} from your knowledge.

Please be extremely thorough and detailed, covering all aspects of {library_info.name} development.
Your guidance should be useful for both beginners and experienced developers.
"""
            
            prompt = enhanced_prompt_template.format(
                library_name=library_info.name,
                tags=tags_str,
                suggested_glob=suggested_glob,
                exa_content_section=exa_content_section,
                mdc_instructions=mdc_instructions
            )
        else:
            # Use existing Exa content
            chunk_size = CONFIG["processing"]["chunk_size"]
            exa_content_section = f"""Based on the following information about {library_info.name} best practices:

Exa search results:
{exa_answer}

Additional information from citations:
{all_citation_text[:chunk_size]}

Please synthesize, enhance, and expand upon this information to create the most comprehensive guide possible.
Add any important best practices that might be missing from the search results.
"""
            
            prompt = enhanced_prompt_template.format(
                library_name=library_info.name,
                tags=tags_str,
                suggested_glob=suggested_glob,
                exa_content_section=exa_content_section,
                mdc_instructions=mdc_instructions
            )
        
        logger.info(f"Sending enhanced prompt to LLM for {library_info.name}")
        response = completion(
            model=CONFIG["api"]["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short descriptive name for the rule"
                        },
                        "glob_pattern": {
                            "type": "string",
                            "description": "Valid glob pattern for target files"
                        },
                        "description": {
                            "type": "string",
                            "description": "1-2 sentence description of what the rule does"
                        },
                        "content": {
                            "type": "string",
                            "description": "Formatted rule content using markdown"
                        }
                    },
                    "required": ["name", "glob_pattern", "description", "content"]
                }
            }
        )
        
        # Parse the JSON response
        json_response = json.loads(response.choices[0].message.content)
        logger.info(f"Successfully generated enhanced rule for {library_info.name}")
        
        # Create MDCRule instance from the JSON
        rule = MDCRule(**json_response)
        return [rule]
        
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
        exa_results = get_library_best_practices_exa(library_name, exa_client)
        
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
                      specific_library: Optional[str] = None, specific_tag: Optional[str] = None):
    """Process rules.json and generate MDC files."""
    # Check if rules.json exists
    if not os.path.exists(json_path):
        logger.error(f"Rules JSON file not found at {json_path}")
        return
    
    # Load rules.json
    with open(json_path, 'r') as f:
        libraries_data = json.load(f)
    
    # Check if the new flat structure is used
    if "libraries" not in libraries_data:
        logger.error("The rules.json file does not use the expected flat structure with tags.")
        return
    
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
    libraries = libraries_data["libraries"]
    
    # Filter libraries based on target tag or library if specified
    if specific_library:
        filtered_libraries = [lib for lib in libraries if lib["name"] == specific_library]
    elif specific_tag:
        filtered_libraries = [lib for lib in libraries if specific_tag in lib["tags"]]
    else:
        filtered_libraries = libraries
    
    # If test mode is enabled, just process one library
    if test_mode:
        if filtered_libraries:
            test_library = filtered_libraries[0]
            library_key, success = process_single_library(test_library, output_dir, exa_client)
            if success:
                progress_tracker.mark_library_completed(library_key)
            else:
                progress_tracker.mark_library_failed(library_key)
        else:
            logger.error("No libraries found to process in test mode")
        return
    
    # Prepare list of libraries to process
    libraries_to_process = []
    
    for library in filtered_libraries:
        # Create a unique key for this library
        library_key = library["name"]
        
        # Skip if already processed
        if progress_tracker.is_library_processed(library_key):
            logger.info(f"Skipping already processed library: {library['name']}")
            continue
        
        # Add to processing list
        libraries_to_process.append(library)
    
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
    parser.add_argument("--output", type=str, default=CONFIG["paths"]["output_dir"], help="Output directory for MDC files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--workers", type=int, help=f"Number of parallel workers (default: {CONFIG['processing']['max_workers']})")
    parser.add_argument("--rate-limit", type=int, help=f"API rate limit calls per minute (default: {CONFIG['api']['rate_limit_calls']})")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save log files")
    parser.add_argument("--exa-results-dir", type=str, default=CONFIG["paths"]["exa_results_dir"], help="Directory to save Exa results")
    args = parser.parse_args()
    
    # Set up verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Override config with command line arguments
    if args.workers:
        CONFIG["processing"]["max_workers"] = args.workers
    
    if args.rate_limit:
        CONFIG["api"]["rate_limit_calls"] = args.rate_limit
    
    if args.exa_results_dir:
        CONFIG["paths"]["exa_results_dir"] = args.exa_results_dir
    
    # Set paths
    rules_json_path = CONFIG["paths"]["rules_json"]
    output_dir = args.output or CONFIG["paths"]["output_dir"]
    
    # Check if rules.json exists
    if not os.path.exists(rules_json_path):
        logger.error(f"rules.json not found at {rules_json_path}")
        return
    
    # Check if EXA_API_KEY is set
    if not os.getenv("EXA_API_KEY"):
        logger.warning("EXA_API_KEY environment variable not set. Will generate content using LLM only.")
    
    try:
        # Process rules.json and generate MDC files
        process_rules_json(
            rules_json_path, 
            output_dir, 
            test_mode=args.test,
            specific_library=args.library,
            specific_tag=args.tag
        )
        logger.info("MDC generation completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 