"""
Downloader module for downloading MDC rule files.

This module handles downloading the selected MDC rule files from the repository.
"""

import os
import time
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from cursor_rules_cli import utils

logger = logging.getLogger(__name__)

# Rate limiting settings
DEFAULT_RATE_LIMIT = 10  # requests per second
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # seconds
DEFAULT_TIMEOUT = 10  # seconds

class DownloadError(Exception):
    """Custom exception for download errors."""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def create_session() -> requests.Session:
    """
    Create a requests session with retry configuration.
    
    Returns:
        Configured requests session
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=DEFAULT_MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # Mount the retry adapter
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

def verify_source_url(source_url: str, session: requests.Session = None) -> Tuple[bool, str]:
    """
    Verify that the source URL is accessible.
    
    Args:
        source_url: Base URL to verify
        session: Optional requests session to use
        
    Returns:
        Tuple of (is_accessible: bool, error_message: str)
    """
    if not session:
        session = create_session()
        
    try:
        # Check if the source URL is accessible
        # Try a few different paths to ensure it's a valid repository
        repo_paths = [
            "",  # Base URL
            "/rules.json",  # Rules index file
            "/rules-mdc"  # Rules directory
        ]
        
        for path in repo_paths:
            url = f"{source_url}{path}"
            response = session.head(url, timeout=DEFAULT_TIMEOUT)
            
            if response.status_code >= 400:
                return False, f"Source URL is not accessible: {url} (Status code: {response.status_code})"
                
        return True, ""
    except requests.RequestException as e:
        return False, f"Failed to connect to source URL: {e}"
    except Exception as e:
        return False, f"Unexpected error verifying source URL: {e}"

def download_rules(
    rules: List[Dict[str, Any]],
    source_url: str,
    temp_dir: Optional[Path] = None,
    rate_limit: int = DEFAULT_RATE_LIMIT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Download selected MDC rule files.
    
    Args:
        rules: List of rule metadata to download
        source_url: Base URL for the repository
        temp_dir: Temporary directory to store downloaded files
        rate_limit: Maximum requests per second
        max_retries: Maximum number of retries for failed downloads
        max_workers: Maximum number of concurrent downloads
        
    Returns:
        List of downloaded rule metadata with local file paths
        
    Raises:
        DownloadError: If there are critical download failures
    """
    if not rules:
        logger.warning("No rules to download")
        return []
    
    # Create temporary directory if not provided
    if temp_dir is None:
        temp_dir = Path.home() / ".cursor-rules-cli" / "temp"
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using temporary directory: {temp_dir}")
    
    # Create rate limiter and session
    rate_limiter = utils.RateLimiter(rate_limit)
    session = create_session()
    
    # Verify source URL is accessible
    is_accessible, error_msg = verify_source_url(source_url, session)
    if not is_accessible:
        logger.error(f"Source URL verification failed: {error_msg}")
        logger.error(f"Please check if the source URL is correct: {source_url}")
        raise DownloadError(f"Source URL is not accessible: {error_msg}")
        
    # Download rules in parallel
    downloaded_rules = []
    failed_downloads = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks
        future_to_rule = {
            executor.submit(
                download_rule,
                rule,
                temp_dir,
                rate_limiter,
                session,
                max_retries
            ): rule for rule in rules
        }
        
        # Process results as they complete
        for future in as_completed(future_to_rule):
            rule = future_to_rule[future]
            try:
                result = future.result()
                if result:
                    downloaded_rules.append(result)
                    logger.info(f"Downloaded {rule['name']}")
                else:
                    failed_downloads.append(rule)
                    logger.error(f"Failed to download {rule['name']}")
            except Exception as e:
                failed_downloads.append(rule)
                logger.error(f"Error downloading {rule['name']}: {str(e)}")
    
    # Close the session
    session.close()
    
    # Report download statistics
    total_rules = len(rules)
    success_count = len(downloaded_rules)
    failed_count = len(failed_downloads)
    
    if failed_count > 0:
        failed_names = [rule['name'] for rule in failed_downloads]
        logger.warning(
            f"Downloaded {success_count}/{total_rules} rules. "
            f"Failed to download {failed_count} rules: {', '.join(failed_names)}"
        )
        if failed_count == total_rules:
            logger.error("All downloads failed. Please check your internet connection and the source URL.")
            logger.error(f"Source URL: {source_url}")
            raise DownloadError("All downloads failed. Check internet connection and source URL.")
    else:
        logger.info(f"Successfully downloaded all {success_count} rules")
    
    return downloaded_rules

def download_rule(
    rule: Dict[str, Any],
    temp_dir: Path,
    rate_limiter: utils.RateLimiter,
    session: requests.Session,
    max_retries: int,
) -> Optional[Dict[str, Any]]:
    """
    Download a single MDC rule file with validation.
    
    Args:
        rule: Rule metadata
        temp_dir: Temporary directory to store downloaded file
        rate_limiter: Rate limiter instance
        session: Requests session
        max_retries: Maximum number of retries
        
    Returns:
        Updated rule metadata with local file path or None if failed
        
    Raises:
        ValidationError: If the downloaded content fails validation
    """
    url = rule["url"]
    name = rule["name"]
    
    # Validate URL
    is_trusted, error_msg = utils.is_url_trusted(url)
    if not is_trusted:
        raise ValidationError(f"URL validation failed for {name}: {error_msg}")
    
    # Create local file path
    local_path = temp_dir / f"{name}.mdc"
    
    # Try to download the file
    for attempt in range(max_retries + 1):
        try:
            # Respect rate limit
            rate_limiter.wait()
            
            # Download the file
            response = session.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            content = response.text
            
            # Validate content
            is_valid, error_msg = utils.validate_mdc_content(content)
            if not is_valid:
                logger.error(f"Content validation failed for {name}: {error_msg}")
                raise ValidationError(f"Content validation failed: {error_msg}")
            
            # Calculate content hash before saving
            content_hash = utils.calculate_content_hash(content)
            logger.debug(f"Content hash for {name}: {content_hash}")
            
            # Save the file
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Verify the saved file
            saved_hash = utils.calculate_file_hash(local_path)
            logger.debug(f"Saved file hash for {name}: {saved_hash}")
            
            if saved_hash != content_hash:
                logger.error(f"File integrity check failed for {name}. Content hash: {content_hash}, File hash: {saved_hash}")
                # Fix: Read the file back and compare the content
                with open(local_path, "r", encoding="utf-8") as f:
                    saved_content = f.read()
                if content == saved_content:
                    logger.info(f"Content matches but hashes differ for {name}. This may be due to line ending differences. Proceeding anyway.")
                    # Update rule metadata and continue
                    rule["local_path"] = str(local_path)
                    rule["content"] = content
                    rule["hash"] = saved_hash  # Use the file hash since that's what we'll verify against later
                    return rule
                else:
                    logger.error(f"Content mismatch for {name}")
                    raise ValidationError("File integrity check failed")
            
            # Update rule metadata
            rule["local_path"] = str(local_path)
            rule["content"] = content
            rule["hash"] = content_hash
            
            return rule
            
        except requests.RequestException as e:
            if attempt < max_retries:
                delay = DEFAULT_RETRY_DELAY * (attempt + 1)
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {name}: {e}")
                logger.warning(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to download {name} after {max_retries + 1} attempts: {e}")
                return None
                
        except ValidationError as e:
            logger.error(f"Validation failed for {name}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error downloading {name}: {e}")
            return None
    
    return None

def preview_rule_content(rule: Dict[str, Any], max_lines: int = 10) -> str:
    """
    Generate a preview of the rule content.
    
    Args:
        rule: Rule metadata with content
        max_lines: Maximum number of lines to include
        
    Returns:
        Preview of the rule content
    """
    if "content" not in rule:
        return "Content not available"
    
    content = rule["content"]
    lines = content.split("\n")
    
    if len(lines) <= max_lines:
        return content
    
    # Show first few lines and indicate there's more
    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)
    preview += f"\n... ({len(lines) - max_lines} more lines)"
    
    return preview

if __name__ == "__main__":
    # For testing
    import json
    logging.basicConfig(level=logging.DEBUG)
    
    # Example rule
    test_rule = {
        "name": "react",
        "tags": ["frontend", "framework", "javascript"],
        "path": "rules-mdc/react.mdc",
        "url": "https://raw.githubusercontent.com/sanjeed5/awesome-cursor-rules-mdc/main/rules-mdc/react.mdc",
        "description": "react (frontend, framework, javascript)",
    }
    
    # Test download
    downloaded = download_rules([test_rule], "")
    
    if downloaded:
        print(f"Downloaded rule: {downloaded[0]['name']}")
        print("Preview:")
        print(preview_rule_content(downloaded[0])) 