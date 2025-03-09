#!/usr/bin/env python
"""
Test script for cursor-rules-cli download functionality
"""

import logging
import sys
from pathlib import Path
from cursor_rules_cli.downloader import download_rules, verify_source_url, create_session, extract_repo_info

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test_downloader")

def main():
    """Test the downloader"""
    # Source URL to test - using GitHub repo URL instead of raw content
    source_url = "https://github.com/sanjeed5/awesome-cursor-rules-mdc"
    
    # Extract repo information
    owner, repo, branch = extract_repo_info(source_url)
    logger.info(f"Repository info: owner={owner}, repo={repo}, branch={branch}")
    
    # Test rules to download
    rules = [
        {"name": "python"},
        {"name": "pandas"},
        {"name": "fastapi"},
        {"name": "pydantic"},
        {"name": "asyncio"}
    ]
    
    # Verify source URL
    logger.info(f"Testing source URL: {source_url}")
    session = create_session()
    is_accessible, error_msg = verify_source_url(source_url, session)
    
    if not is_accessible:
        logger.error(f"Source URL verification failed: {error_msg}")
        return 1
    
    logger.info("Source URL verification passed")
    
    # Test downloading rules
    logger.info(f"Testing download of {len(rules)} rules")
    try:
        downloaded_rules = download_rules(rules, source_url)
        logger.info(f"Successfully downloaded {len(downloaded_rules)} rules")
        
        # Print details of downloaded rules
        for rule in downloaded_rules:
            logger.info(f"Rule: {rule['name']}, Local path: {rule.get('local_path')}")
            
        return 0
    except Exception as e:
        logger.error(f"Download test failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 