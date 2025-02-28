"""
Installer module for installing MDC rule files.

This module handles installing the downloaded MDC rule files to the project's
.cursor/rules directory.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def install_rules(
    rules: List[Dict[str, Any]],
    force: bool = False,
    cursor_dir: Optional[Path] = None,
    backup: bool = True,
) -> Dict[str, List]:
    """
    Install downloaded MDC rule files to the project's .cursor/rules directory.
    
    Args:
        rules: List of rule metadata with local file paths
        force: Whether to overwrite existing rules
        cursor_dir: Path to .cursor directory (defaults to ./.cursor in current directory)
        backup: Whether to backup existing rules
        
    Returns:
        Dictionary with 'installed' and 'failed' lists
    """
    result = {
        "installed": [],
        "failed": []
    }
    
    if not rules:
        logger.warning("No rules to install")
        return result
    
    # Determine .cursor directory - use project local directory
    if cursor_dir is None:
        cursor_dir = Path.cwd() / ".cursor"
    
    # Create rules directory if it doesn't exist
    rules_dir = cursor_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Installing rules to {rules_dir}")
    
    # Backup existing rules if needed
    if backup and any(rules_dir.glob("*.mdc")):
        backup_dir = create_backup(rules_dir)
        if backup_dir:
            logger.info(f"Backed up existing rules to {backup_dir}")
    
    # Install each rule
    for rule in rules:
        if "local_path" not in rule:
            failure = {
                "rule": rule,
                "error": "No local file path"
            }
            result["failed"].append(failure)
            logger.warning(f"Skipping {rule['name']}: No local file path")
            continue
        
        # Determine target path
        target_path = rules_dir / f"{rule['name']}.mdc"
        
        # Check if rule already exists
        if target_path.exists() and not force:
            failure = {
                "rule": rule,
                "error": "Rule already exists (use --force to overwrite)"
            }
            result["failed"].append(failure)
            logger.warning(f"Skipping {rule['name']}: Rule already exists (use --force to overwrite)")
            continue
        
        # Copy the rule file
        try:
            shutil.copy2(rule["local_path"], target_path)
            logger.debug(f"Installed {rule['name']} to {target_path}")
            result["installed"].append(rule)
        except IOError as e:
            failure = {
                "rule": rule,
                "error": str(e)
            }
            result["failed"].append(failure)
            logger.error(f"Failed to install {rule['name']}: {e}")
    
    logger.info(f"Installed {len(result['installed'])}/{len(rules)} rules to {rules_dir}")
    return result

def create_backup(rules_dir: Path) -> Optional[Path]:
    """
    Create a backup of existing rules in the project directory.
    
    Args:
        rules_dir: Path to .cursor/rules directory
        
    Returns:
        Path to backup directory or None if failed
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep backups in the project directory under .cursor/backups
    backup_dir = rules_dir.parent / "backups" / f"rules_backup_{timestamp}"
    
    try:
        # Create backup directory
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy existing rules
        for rule_file in rules_dir.glob("*.mdc"):
            shutil.copy2(rule_file, backup_dir / rule_file.name)
        
        return backup_dir
    except IOError as e:
        logger.error(f"Failed to create backup: {e}")
        return None

def list_installed_rules(cursor_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    List installed MDC rule files.
    
    Args:
        cursor_dir: Path to .cursor directory (defaults to ./.cursor in current directory)
        
    Returns:
        List of installed rule metadata
    """
    # Determine .cursor directory - use project local directory
    if cursor_dir is None:
        cursor_dir = Path.cwd() / ".cursor"
    
    rules_dir = cursor_dir / "rules"
    
    if not rules_dir.exists():
        logger.debug(f"Rules directory not found: {rules_dir}")
        return []
    
    installed_rules = []
    for rule_file in rules_dir.glob("*.mdc"):
        # Extract rule name from filename
        name = rule_file.stem
        
        # Read first few lines to extract description
        try:
            with open(rule_file, "r", encoding="utf-8") as f:
                content = f.read(1000)  # Read first 1000 chars
                
                # Try to extract description from frontmatter
                description = name
                if "description:" in content:
                    desc_line = [line for line in content.split("\n") if "description:" in line]
                    if desc_line:
                        description = desc_line[0].split("description:")[1].strip()
        except IOError:
            description = name
        
        installed_rules.append({
            "name": name,
            "path": str(rule_file),
            "description": description,
        })
    
    return installed_rules

if __name__ == "__main__":
    # For testing
    import sys
    logging.basicConfig(level=logging.DEBUG)
    
    # List installed rules
    rules = list_installed_rules()
    print(f"Installed rules: {len(rules)}")
    for rule in rules:
        print(f"  - {rule['name']}: {rule['description']}") 