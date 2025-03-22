"""
Utils module with helper functions for the Cursor Rules CLI.

This module provides utility functions for file operations, security verification,
and other common tasks.
"""

import os
import re
import hashlib
import logging
import platform
import json
import time
from threading import Lock
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import requests
from functools import lru_cache
from urllib.parse import urlparse
import validators

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter to avoid overwhelming servers."""
    
    def __init__(self, rate_limit: int):
        """
        Initialize rate limiter.
        
        Args:
            rate_limit: Maximum requests per second
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self._lock = Lock()  # Thread-safe locking
    
    def wait(self):
        """
        Wait if necessary to respect the rate limit.
        Thread-safe implementation.
        """
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            
            # If we've made a request recently, wait
            if elapsed < (1.0 / self.rate_limit):
                sleep_time = (1.0 / self.rate_limit) - elapsed
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()

def calculate_content_hash(content: str) -> str:
    """
    Calculate SHA-256 hash of content string.
    
    Args:
        content: Content to hash
        
    Returns:
        SHA-256 hash as hex string
    """
    # Normalize line endings to '\n'
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_cursor_dir() -> Path:
    """
    Get the path to the project's .cursor directory.
    
    Returns:
        Path to the project's .cursor directory
    """
    return Path.cwd() / ".cursor"

def get_rules_dir() -> Path:
    """
    Get the path to the project's .cursor/rules directory.
    
    Returns:
        Path to the project's .cursor/rules directory
    """
    return get_cursor_dir() / "rules"

def get_config_file() -> Path:
    """
    Get the path to the configuration file.
    
    Returns:
        Path to the configuration file in the project's .cursor directory
    """
    return get_cursor_dir() / "rules-cli-config.json"

def load_config() -> Dict[str, Any]:
    """
    Load configuration from the config file.
    
    Returns:
        Configuration dictionary
    """
    config_file = get_config_file()
    
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load config file: {e}")
        return {}

def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to the config file.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    config_file = get_config_file()
    
    try:
        # Ensure the directory exists
        ensure_dir_exists(config_file.parent)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
    except IOError as e:
        logger.error(f"Failed to save config file: {e}")
        return False

def ensure_dir_exists(path: Path) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        True if the directory exists or was created, False otherwise
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def is_url_trusted(url: str) -> Tuple[bool, str]:
    """
    Check if a URL is from a trusted source using proper URL parsing.
    
    Args:
        url: URL to check
        
    Returns:
        Tuple of (is_trusted: bool, error_message: str)
    """
    # First validate URL format
    if not validators.url(url):
        return False, "Invalid URL format"
    
    try:
        parsed_url = urlparse(url)
        
        # Check for HTTPS
        if parsed_url.scheme != "https":
            return False, "URL must use HTTPS"
        
        # List of trusted domains and their subdomains
        trusted_domains = [
            "raw.githubusercontent.com",
            "github.com",
        ]
        
        # Extract domain from URL
        domain = parsed_url.netloc.lower()
        
        # Check if domain exactly matches or is subdomain of trusted domains
        is_trusted = any(
            domain == trusted_domain or domain.endswith(f".{trusted_domain}")
            for trusted_domain in trusted_domains
        )
        
        if not is_trusted:
            return False, f"Domain {domain} is not in trusted list"
        
        # Additional security checks for GitHub URLs
        if "github" in domain:
            # Validate path format for raw.githubusercontent.com
            if domain == "raw.githubusercontent.com":
                path_parts = [p for p in parsed_url.path.split("/") if p]
                if len(path_parts) < 4:  # username/repo/branch/path
                    return False, "Invalid GitHub raw URL format"
            
            # Validate path format for github.com
            elif domain == "github.com":
                path_parts = [p for p in parsed_url.path.split("/") if p]
                if len(path_parts) < 2:  # username/repo
                    return False, "Invalid GitHub repository URL format"
        
        return True, ""
        
    except Exception as e:
        return False, f"URL validation error: {str(e)}"

def validate_mdc_content(content: str) -> Tuple[bool, str]:
    """
    Validate MDC file content more thoroughly.
    
    Args:
        content: Content to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not content:
        return False, "Empty content"
    
    # Check for frontmatter
    if not content.startswith("---"):
        return False, "Missing frontmatter start"
    
    # Find end of frontmatter
    frontmatter_end = content.find("---", 3)
    if frontmatter_end == -1:
        return False, "Missing frontmatter end"
    
    # Extract frontmatter
    frontmatter = content[3:frontmatter_end].strip()
    
    # Required fields in frontmatter
    required_fields = ["description", "globs"]
    
    # Check for required fields
    for field in required_fields:
        if f"{field}:" not in frontmatter:
            return False, f"Missing required field: {field}"
    
    # Check for content after frontmatter
    content_after_frontmatter = content[frontmatter_end + 3:].strip()
    if not content_after_frontmatter:
        return False, "No content after frontmatter"
    
    # Check for potentially malicious content
    suspicious_patterns = [
        r"<script",
        r"javascript:",
        r"data:text/html",
        r"vbscript:",
        r"onload=",
        r"onerror=",
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return False, f"Suspicious content detected: {pattern}"
    
    # Validate globs format
    try:
        globs_line = next(line for line in frontmatter.split("\n") if line.startswith("globs:"))
        globs_value = globs_line.split(":", 1)[1].strip()
        if not globs_value:
            return False, "Empty globs value"
        
        # Basic glob pattern validation
        invalid_chars = '<>|"'
        if any(char in globs_value for char in invalid_chars):
            return False, "Invalid characters in globs pattern"
            
    except StopIteration:
        return False, "Could not parse globs field"
    
    return True, ""

def calculate_file_hash(file_path: Path) -> Optional[str]:
    """
    Calculate the SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash as a hex string, or None if failed
    """
    try:
        # Read the file as text and normalize line endings
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Normalize line endings to '\n'
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Calculate hash from normalized content
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    except IOError as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        return None

def is_valid_mdc_file(content: str) -> bool:
    """
    Check if the content is a valid MDC file.
    
    Args:
        content: File content to check
        
    Returns:
        True if valid, False otherwise
    """
    # Check for frontmatter
    if not content.startswith("---"):
        return False
    
    # Check for description
    if "description:" not in content:
        return False
    
    # Check for globs
    if "globs:" not in content:
        return False
    
    # Check for closing frontmatter
    frontmatter_end = content.find("---", 3)
    if frontmatter_end == -1:
        return False
    
    # Check for content after frontmatter
    if len(content) <= frontmatter_end + 3:
        return False
    
    # Check if there's actual content after the frontmatter
    content_after_frontmatter = content[frontmatter_end + 3:].strip()
    if not content_after_frontmatter:
        return False
    
    return True

def sanitize_filename(name: str) -> str:
    """
    Sanitize a filename to ensure it's valid.
    
    Args:
        name: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscores
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, "_", name)
    
    # Ensure it's not too long
    max_length = 255 if platform.system() != "Windows" else 240
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def preview_content(content: str, max_lines: int = 10) -> str:
    """
    Generate a preview of content.
    
    Args:
        content: Content to preview
        max_lines: Maximum number of lines to include
        
    Returns:
        Preview of the content
    """
    lines = content.split("\n")
    
    if len(lines) <= max_lines:
        return content
    
    # Show first few lines and indicate there's more
    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)
    preview += f"\n... ({len(lines) - max_lines} more lines)"
    
    return preview

def validate_github_repo(repo: str) -> bool:
    """
    Validate a GitHub repository string.
    
    Args:
        repo: GitHub repository string (username/repo)
        
    Returns:
        True if valid, False otherwise
    """
    if not repo:
        return False
    
    # Check format (username/repo)
    if not re.match(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$', repo):
        return False
    
    # Check if the repository exists and has a rules.json file
    try:
        url = f"https://raw.githubusercontent.com/{repo}/main/rules.json"
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def get_project_config_file(project_dir: Path) -> Path:
    """
    Get the path to the project-specific configuration file.
    
    Args:
        project_dir: Path to the project directory
        
    Returns:
        Path to the project configuration file
    """
    return project_dir / ".cursor-rules-cli.json"

def load_project_config(project_dir: Path) -> Dict[str, Any]:
    """
    Load configuration from the project-specific config file.
    
    Args:
        project_dir: Path to the project directory
        
    Returns:
        Project configuration dictionary
    """
    config_file = get_project_config_file(project_dir)
    
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load project config file: {e}")
        return {}

def save_project_config(project_dir: Path, config: Dict[str, Any]) -> bool:
    """
    Save configuration to the project-specific config file.
    
    Args:
        project_dir: Path to the project directory
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    config_file = get_project_config_file(project_dir)
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
    except IOError as e:
        logger.error(f"Failed to save project config file: {e}")
        return False

def merge_configs(global_config: Dict[str, Any], project_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge global and project-specific configurations.
    Project configuration takes precedence over global configuration for project-specific settings.
    CLI-wide settings (like custom_repo and source) are always taken from global config.
    
    Args:
        global_config: Global configuration dictionary
        project_config: Project-specific configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    # CLI-wide settings that should only come from global config
    cli_wide_settings = ["custom_repo", "source"]
    
    # Start with a copy of the global config
    merged = global_config.copy()
    
    # Update with project config, but exclude CLI-wide settings
    project_config_filtered = {k: v for k, v in project_config.items() if k not in cli_wide_settings}
    merged.update(project_config_filtered)
    
    return merged

# Default paths
DEFAULT_RULES_PATH = Path(__file__).parent.parent / "rules.json"
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".cache"

@lru_cache(maxsize=1)
def load_library_data(rules_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and cache library data from rules.json.
    
    Args:
        rules_path: Optional path to rules.json
        
    Returns:
        Dictionary of library data
    """
    # If a specific path is provided, use it first
    if rules_path and Path(rules_path).exists():
        logger.debug(f"Using specified rules.json at {rules_path}")
        path_to_use = Path(rules_path)
    else:
        # Search for rules.json in priority order
        possible_paths = [
            Path(__file__).parent.parent / "rules.json",  # package root rules.json
            Path.cwd() / "rules.json"  # current directory rules.json
        ]
        
        path_to_use = None
        for path in possible_paths:
            if path.exists():
                path_to_use = path
                logger.debug(f"Found rules.json at {path}")
                break
    
    if not path_to_use or not path_to_use.exists():
        logger.warning("rules.json not found in any standard location, using default library detection")
        return {}
    
    try:
        with open(path_to_use, 'r') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded rules.json from {path_to_use}")
            return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error loading rules.json from {path_to_use}: {e}")
        return {}

def normalize_library_name(name: str, library_data: Dict[str, Any]) -> str:
    """
    Normalize a library name to match rules.json conventions.
    
    Args:
        name: Library name to normalize
        library_data: Library data from rules.json
        
    Returns:
        Normalized library name
    """
    if not library_data or "libraries" not in library_data:
        return name.lower()
    
    name_lower = name.lower()
    lib_map = {lib["name"].lower(): lib["name"] for lib in library_data["libraries"]}
    
    # Handle special cases and common aliases
    special_cases = {
        "torch": "pytorch",
        "tf": "tensorflow",
        "bs4": "beautifulsoup4",
        "plt": "matplotlib",
        "np": "numpy",
        "pd": "pandas"
    }
    
    if name_lower in special_cases and special_cases[name_lower] in lib_map:
        return lib_map[special_cases[name_lower]]
    
    return lib_map.get(name_lower, name_lower)

def calculate_library_popularity(lib_name: str, library_data: Dict[str, Any]) -> float:
    """
    Calculate a library's popularity score based on its tags and relationships.
    
    Args:
        lib_name: Library name
        library_data: Library data from rules.json
        
    Returns:
        Popularity score between 0 and 1
    """
    if not library_data or "libraries" not in library_data:
        return 0.5  # Default score
    
    lib_map = {lib["name"].lower(): lib for lib in library_data["libraries"]}
    lib_name_lower = lib_name.lower()
    
    if lib_name_lower not in lib_map:
        return 0.5
    
    lib_info = lib_map[lib_name_lower]
    tags = lib_info.get("tags", [])
    
    # Base score from number of tags (more tags = more versatile)
    tag_score = min(len(tags) / 10, 0.5)  # Up to 0.5 from tags
    
    # Additional score from important tags
    important_tags = {"framework", "language", "major-platform"}
    tag_importance = sum(0.1 for tag in tags if tag in important_tags)
    
    # Calculate related libraries score
    related_count = sum(1 for lib in library_data["libraries"]
                       if any(tag in lib.get("tags", []) for tag in tags))
    relationship_score = min(related_count / len(library_data["libraries"]), 0.3)
    
    total_score = tag_score + tag_importance + relationship_score
    return min(total_score, 1.0)

def get_project_context(detected_libs: Set[str], library_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Determine project context based on detected libraries.
    
    Args:
        detected_libs: Set of detected library names
        library_data: Library data from rules.json
        
    Returns:
        Dictionary of context scores (e.g., {'frontend': 0.8, 'backend': 0.3})
    """
    contexts = {
        "frontend": 0.0,
        "backend": 0.0,
        "data-science": 0.0,
        "devops": 0.0,
        "mobile": 0.0
    }
    
    if not library_data or "libraries" not in library_data:
        return contexts
    
    lib_map = {lib["name"].lower(): lib for lib in library_data["libraries"]}
    total_libs = len(detected_libs)
    
    if total_libs == 0:
        return contexts
    
    # Context indicators in tags
    context_tags = {
        "frontend": {"frontend", "ui", "javascript", "css", "html"},
        "backend": {"backend", "api", "server", "database"},
        "data-science": {"data-science", "machine-learning", "ai", "analytics"},
        "devops": {"devops", "ci-cd", "containerization", "cloud"},
        "mobile": {"mobile", "ios", "android", "cross-platform"}
    }
    
    # Calculate scores based on detected libraries
    for lib in detected_libs:
        lib_lower = lib.lower()
        if lib_lower in lib_map:
            lib_tags = set(lib_map[lib_lower].get("tags", []))
            
            for context, indicators in context_tags.items():
                if lib_tags & indicators:  # If there's any overlap
                    contexts[context] += 1 / total_libs
    
    # Normalize scores
    max_score = max(contexts.values())
    if max_score > 0:
        contexts = {k: v / max_score for k, v in contexts.items()}
    
    return contexts

def create_cache_key(*args) -> str:
    """
    Create a cache key from arguments.
    
    Args:
        *args: Arguments to create key from
        
    Returns:
        Cache key string
    """
    key = ":".join(str(arg) for arg in args)
    
    # If the key is too long, hash it to avoid file name length issues
    if len(key) > 100:
        return hashlib.md5(key.encode()).hexdigest()
    
    return key

def get_cached_data(cache_key: str) -> Optional[Any]:
    """
    Get data from cache.
    
    Args:
        cache_key: Cache key
        
    Returns:
        Cached data or None if not found
    """
    cache_file = DEFAULT_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None

def set_cached_data(cache_key: str, data: Any) -> bool:
    """
    Save data to cache.
    
    Args:
        cache_key: Cache key
        data: Data to cache
        
    Returns:
        True if successful, False otherwise
    """
    try:
        DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = DEFAULT_CACHE_DIR / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        return True
    except (IOError, OSError):
        return False

if __name__ == "__main__":
    # For testing
    logging.basicConfig(level=logging.DEBUG)
    
    cursor_dir = get_cursor_dir()
    rules_dir = get_rules_dir()
    
    print(f"Cursor directory: {cursor_dir}")
    print(f"Rules directory: {rules_dir}")
    
    # Test directory creation
    if ensure_dir_exists(rules_dir):
        print(f"Created rules directory: {rules_dir}")
    
    # Test filename sanitization
    print(f"Sanitized filename: {sanitize_filename('invalid:file*name.txt')}") 