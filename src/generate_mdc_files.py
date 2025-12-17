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
        "llm_model": "gemini/gemini-2.5-flash",
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

def get_library_best_practices_exa(library_name: str, exa_client: Optional[Exa] = None, tags: List[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
    """Use Exa to search for best practices for a library.
    
    Checks for cached results first to avoid redundant API calls.
    Set force_refresh=True to bypass cache and fetch fresh results.
    """
    # Create a safe filename for cache lookup
    safe_name = re.sub('[^a-z0-9-]+', '-', library_name.lower()).strip('-')
    exa_results_dir = SCRIPT_DIR / CONFIG["paths"]["exa_results_dir"]
    cache_file = exa_results_dir / f"{safe_name}_exa_result.json"
    
    # Check cache first (unless force_refresh is True)
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_result = json.load(f)
            # Validate cache has expected structure
            if "answer" in cached_result:
                logger.info(f"Using cached Exa results for {library_name}")
                return cached_result
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Cache file corrupted for {library_name}, will fetch fresh: {e}")
    
    # No cache or force refresh - call Exa API
    return _fetch_from_exa(library_name, exa_client, tags, cache_file)


@retry(stop=stop_after_attempt(CONFIG["exa_api"]["max_retries"]), 
       wait=wait_exponential(multiplier=1, min=CONFIG["exa_api"]["retry_min_wait"], max=CONFIG["exa_api"]["retry_max_wait"]))
@sleep_and_retry
@limits(calls=CONFIG["exa_api"]["rate_limit_calls"], period=CONFIG["exa_api"]["rate_limit_period"])
def _fetch_from_exa(library_name: str, exa_client: Optional[Exa], tags: List[str], cache_file: Path) -> Dict[str, Any]:
    """Internal function to fetch from Exa API with rate limiting and retries."""
    if exa_client is None:
        logger.warning("Exa client not initialized, falling back to LLM generation")
        return {"answer": "", "citations": []}
    
    try:
        # Construct a query for best practices that includes tags for better context
        current_year = datetime.datetime.now().year
        if tags and len(tags) > 0:
            # Select up to 3 most relevant tags to keep the query focused
            relevant_tags = tags[:3] if len(tags) > 3 else tags
            tags_str = " ".join(relevant_tags)
            query = f"{library_name} best practices coding standards {current_year} modern {tags_str} development"
        else:
            query = f"{library_name} best practices coding standards {current_year} modern"
            
        logger.info(f"Searching Exa for: {query}")
        
        # Call Exa API
        result = exa_client.answer(query, text=True)
        
        # Convert the AnswerResponse object to a dictionary
        result_dict = {
            "answer": result.answer if hasattr(result, 'answer') else "",
            "citations": result.citations if hasattr(result, 'citations') else []
        }
        
        # Save the Exa result to cache file
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"Saved Exa result for {library_name} to {cache_file}")
        
        if result_dict["answer"]:
            logger.info(f"Successfully retrieved Exa results for {library_name}")
            return result_dict
        
        # If no results found, return empty
        logger.warning(f"No results found via Exa for {library_name}")
        return {"answer": "", "citations": []}
        
    except Exception as e:
        logger.error(f"Error fetching from Exa for {library_name}: {str(e)}")
        return {"answer": "", "citations": []}

def _get_relevant_sections(tags: List[str]) -> List[str]:
    """Return relevant sections based on library tags."""
    # Base sections that apply to most libraries
    base_sections = [
        "Code Organization and Structure",
        "Common Patterns and Anti-patterns",
        "Performance Considerations",
        "Common Pitfalls and Gotchas",
    ]
    
    # Tag-specific sections
    tag_sections = {
        "backend": ["Security Best Practices", "Error Handling", "API Design"],
        "frontend": ["Component Architecture", "State Management", "Accessibility"],
        "database": ["Security Best Practices", "Query Optimization", "Data Modeling"],
        "orm": ["Security Best Practices", "Query Optimization", "Migration Patterns"],
        "authentication": ["Security Best Practices", "Session Management", "Token Handling"],
        "security": ["Security Best Practices", "Vulnerability Prevention", "Input Validation"],
        "testing": ["Test Organization", "Mocking Strategies", "Coverage Patterns"],
        "css": ["Responsive Design", "Browser Compatibility", "Naming Conventions"],
        "ui": ["Component Architecture", "Accessibility", "Theming"],
        "state-management": ["State Organization", "Side Effects", "Debugging"],
        "api": ["Error Handling", "Request/Response Patterns", "Rate Limiting"],
        "cli": ["Argument Parsing", "Error Messages", "Exit Codes"],
        "devops": ["Configuration Management", "Environment Variables", "Logging"],
        "python": ["Type Hints", "Virtual Environments", "Packaging"],
        "typescript": ["Type Safety", "Module Organization", "Strict Mode"],
        "react": ["Hooks Best Practices", "Component Lifecycle", "Re-render Optimization"],
        "vue": ["Composition API", "Reactivity", "Component Communication"],
        "node": ["Async Patterns", "Error Handling", "Module System"],
    }
    
    # Collect additional sections based on tags
    additional = []
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in tag_sections:
            for section in tag_sections[tag_lower]:
                if section not in base_sections and section not in additional:
                    additional.append(section)
    
    # Add testing section if not already covered
    if "Testing Approaches" not in additional and "testing" not in [t.lower() for t in tags]:
        additional.append("Testing Approaches")
    
    return base_sections + additional[:4]  # Limit to avoid overwhelming the prompt


def _get_glob_hint(library_name: str, tags: List[str]) -> str:
    """Suggest a glob pattern based on library name and tags."""
    # Direct library mappings for common libraries
    library_globs = {
        "react": "**/*.{jsx,tsx}",
        "vue": "**/*.{vue,js,ts}",
        "svelte": "**/*.{svelte,js,ts}",
        "angular": "**/*.{ts,html}",
        "next.js": "**/*.{js,jsx,ts,tsx}",
        "nuxt": "**/*.{vue,js,ts}",
        "prisma": "**/*.prisma",
        "drizzle": "**/*.{ts,js}",
        "tailwind": "**/*.{css,html,jsx,tsx,vue}",
        "sass": "**/*.{scss,sass}",
        "less": "**/*.less",
        "express": "**/*.{js,ts}",
        "fastapi": "**/*.py",
        "django": "**/*.py",
        "flask": "**/*.py",
        "pytest": "**/*.py",
        "jest": "**/*.{js,ts,jsx,tsx}",
        "vitest": "**/*.{js,ts,jsx,tsx}",
        "playwright": "**/*.{js,ts}",
        "cypress": "**/*.{js,ts,cy.js,cy.ts}",
        "docker": "**/Dockerfile,**/docker-compose*.{yml,yaml}",
        "kubernetes": "**/*.{yaml,yml}",
        "terraform": "**/*.tf",
        "graphql": "**/*.{graphql,gql}",
        "sql": "**/*.sql",
    }
    
    # Check for direct match
    name_lower = library_name.lower()
    if name_lower in library_globs:
        return library_globs[name_lower]
    
    # Tag-based fallbacks
    tag_globs = {
        "python": "**/*.py",
        "typescript": "**/*.{ts,tsx}",
        "javascript": "**/*.{js,jsx}",
        "rust": "**/*.rs",
        "go": "**/*.go",
        "java": "**/*.java",
        "ruby": "**/*.rb",
        "php": "**/*.php",
        "css": "**/*.{css,scss,sass,less}",
        "html": "**/*.{html,htm}",
        "yaml": "**/*.{yaml,yml}",
        "json": "**/*.json",
        "markdown": "**/*.{md,mdx}",
    }
    
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in tag_globs:
            return tag_globs[tag_lower]
    
    # Default fallback
    return "**/*"


@sleep_and_retry
@limits(calls=CONFIG["api"]["rate_limit_calls"], period=CONFIG["api"]["rate_limit_period"])
@retry(stop=stop_after_attempt(CONFIG["api"]["max_retries"]), 
       wait=wait_exponential(multiplier=1, min=CONFIG["api"]["retry_min_wait"], max=CONFIG["api"]["retry_max_wait"]))
def generate_mdc_rules_from_exa(library_info: LibraryInfo, exa_results: Dict[str, Any]) -> List[MDCRule]:
    """Generate MDC rules for a library using LLM directly from Exa results."""
    try:
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
        
        # Determine relevant sections based on tags
        sections = _get_relevant_sections(library_info.tags)
        sections_text = "\n".join(f"- {section}" for section in sections)
        
        # Suggest glob pattern based on tags
        glob_hint = _get_glob_hint(library_info.name, library_info.tags)
        
        # Simplified prompt that asks for direct markdown output
        prompt_template = """Create a comprehensive Cursor IDE rule file for {library_name}.

## About Cursor Rules
Cursor rules provide persistent instructions to the AI coding assistant. Good rules are:
- **Focused and actionable** - provide clear, specific guidance (not generic advice)
- **Under 500 lines** - keep content concise and targeted  
- **Practical** - include concrete, copy-pasteable code examples
- **Clear** - write like internal documentation that your team would reference
- **Opinionated** - recommend specific approaches rather than listing all options
- **Avoid vague guidance** - be specific about *when* and *why* to use patterns

## Library Information
- **Name:** {library_name}
- **Tags:** {tags}
- **Target files:** {glob_hint}

{exa_content_section}

## Content Strategy
Focus on these areas relevant to {library_name}:
{sections}

## Writing Style Requirements:
1. **Be opinionated and decisive** - Say "Use X for Y" not "You could use X or Y or Z"
2. **Emphasize common mistakes** - "❌ BAD" vs "✅ GOOD" examples are highly valuable
3. **Real-world scenarios** - Focus on practical situations developers encounter daily
4. **Code-heavy** - Show, don't just tell. Every guideline should have a code example
5. **Context matters** - Explain *when* to apply patterns, not just *what* they are
6. **Modern best practices** - Focus on current, recommended approaches (e.g., hooks over classes in React)

## Output Format
Output a complete .mdc file with YAML frontmatter followed by markdown content.

Use this EXACT format:

---
description: [A clear 1-2 sentence description focusing on WHAT this helps developers do]
globs: {glob_hint}
---

# {library_name} Best Practices

[Your comprehensive guide content here in markdown format]

## Critical Guidelines:
- Start with frontmatter (the --- YAML section)
- Use proper markdown formatting (headers, lists, code blocks)
- Include concrete, copy-pasteable code examples for EVERY recommendation
- Be specific and opinionated - recommend one good way over listing all options
- Emphasize common pitfalls with ❌ BAD vs ✅ GOOD comparisons
- Focus on real-world use cases developers face daily
- Keep under 500 lines total
- Write like clear internal documentation, not generic tutorials
- Be specific to {library_name} - avoid generic software advice

---
**IMPORTANT: Use modern best practices as of {current_date}. Focus on current, recommended approaches and avoid deprecated patterns.**"""
        
        # Determine content section based on available information
        if len(exa_answer.strip()) < 100 and len(all_citation_text.strip()) < 200:
            exa_content_section = f"""## Your Task
Generate comprehensive best practices for {library_info.name} from your expert knowledge.
Be thorough and detailed, covering patterns, anti-patterns, performance, security, and testing.
Make it useful for both beginners and experienced developers."""
        else:
            chunk_size = CONFIG["processing"]["chunk_size"]
            exa_content_section = f"""## Reference Information
Use this information about {library_info.name} best practices as a foundation:

### Research Summary:
{exa_answer}

### Additional Context:
{all_citation_text[:chunk_size]}

## Your Task
Synthesize, enhance, and expand upon this information to create a comprehensive guide.
Add important best practices that may be missing. Focus on practical, actionable advice."""
        
        # Get current date for context (at the end for prompt caching)
        current_date = datetime.datetime.now().strftime("%B %Y")
        
        prompt = prompt_template.format(
            library_name=library_info.name,
            tags=tags_str,
            exa_content_section=exa_content_section,
            sections=sections_text,
            glob_hint=glob_hint,
            current_date=current_date
        )
        
        logger.info(f"Sending prompt to LLM for {library_info.name}")
        
        # System prompt for consistent behavior
        system_prompt = """You are a senior software engineer writing internal documentation for your team.
Your team uses Cursor IDE and relies on these rules as their definitive guide.

Your writing style:
- **Opinionated** - Recommend THE best approach, not all possible options
- **Practical** - Every statement should be immediately actionable
- **Example-driven** - Show code for every guideline (❌ BAD vs ✅ GOOD format)
- **Real-world focused** - Address actual problems developers face daily
- **Concise but complete** - Dense with useful information, zero fluff
- **Modern** - Always use current best practices, avoid deprecated patterns

Output complete .mdc files with YAML frontmatter and markdown content.
Write like you're teaching a teammate who will reference this daily."""
        
        # Use LiteLLM's completion - no JSON mode needed
        response = completion(
            model=CONFIG["api"]["llm_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
            temperature=0.4
        )
        
        # Get the response content
        if hasattr(response.choices[0].message, 'content'):
            content = response.choices[0].message.content
        elif hasattr(response.choices[0].message, 'text'):
            content = response.choices[0].message.text
        else:
            raise ValueError("Unexpected response format from LLM")

        # Clean up the response
        content = content.strip()
        
        # Remove markdown code fences if present
        if content.startswith('```'):
            # Find the end of the opening fence
            first_newline = content.find('\n')
            if first_newline != -1:
                content = content[first_newline + 1:]
            # Remove closing fence
            if content.endswith('```'):
                content = content[:-3].strip()
        
        # Parse the frontmatter and content
        if not content.startswith('---'):
            raise ValueError(f"Response does not start with YAML frontmatter for {library_info.name}")
        
        # Find the end of frontmatter
        second_marker = content.find('---', 3)
        if second_marker == -1:
            raise ValueError(f"Invalid frontmatter format for {library_info.name}")
        
        frontmatter = content[3:second_marker].strip()
        markdown_content = content[second_marker + 3:].strip()
        
        # Parse frontmatter (simple YAML parsing)
        description = ""
        glob_pattern = glob_hint
        
        for line in frontmatter.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key == 'description':
                    description = value
                elif key in ['globs', 'glob']:
                    glob_pattern = value
        
        # Create MDCRule instance
        rule = MDCRule(
            name=f"{library_info.name} Best Practices",
            glob_pattern=glob_pattern,
            description=description if description else f"Best practices and coding standards for {library_info.name}",
            content=markdown_content
        )
        
        logger.info(f"Successfully generated rule for {library_info.name}")
        return [rule]
        
    except Exception as e:
        logger.error(f"Error generating MDC rule for {library_info.name}: {str(e)}")
        raise

def create_mdc_file(rule: MDCRule, output_path: Path) -> None:
    """Create a single .mdc file from a rule."""
    # Content should already be clean markdown from the generation function
    content = rule.content.strip()
    
    # Create the complete .mdc file with frontmatter
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

def process_single_library(library_info: Dict[str, Any], output_dir: str, exa_client: Optional[Exa] = None, force_refresh_exa: bool = False) -> Tuple[str, bool]:
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
        
        # Get best practices using Exa (uses cache unless force_refresh_exa is True)
        logger.info(f"Getting best practices for {library_name} using Exa")
        exa_results = get_library_best_practices_exa(library_name, exa_client, library_tags, force_refresh=force_refresh_exa)
        
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
                      retry_failed_only: bool = False, force_refresh_exa: bool = False):
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
            # If retry_failed_only is False (i.e., --regenerate-all was passed), process all libraries
            if not retry_failed_only:
                libraries_to_process.append(library)
            elif not progress_tracker.is_library_processed(library_key) or progress_tracker.is_library_failed(library_key):
                libraries_to_process.append(library)
    else:
        # Normal processing - apply progress tracking rules
        libraries_to_process = []
        for library in libraries:
            library_key = library["name"]
            # If retry_failed_only is False (i.e., --regenerate-all was passed), process all libraries
            if not retry_failed_only:
                libraries_to_process.append(library)
            elif not progress_tracker.is_library_processed(library_key) or progress_tracker.is_library_failed(library_key):
                libraries_to_process.append(library)
    
    # If test mode is enabled, just process one library
    if test_mode:
        if libraries_to_process:
            test_library = libraries_to_process[0]
            library_key, success = process_single_library(test_library, output_dir, exa_client, force_refresh_exa)
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
        process_func = partial(process_single_library, output_dir=output_dir, exa_client=exa_client, force_refresh_exa=force_refresh_exa)
        
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
    parser.add_argument("--refresh-exa", action="store_true", help="Bypass Exa cache and fetch fresh results")
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
            retry_failed_only=not args.regenerate_all,  # Invert regenerate-all flag
            force_refresh_exa=args.refresh_exa
        )
        logger.info("MDC generation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 