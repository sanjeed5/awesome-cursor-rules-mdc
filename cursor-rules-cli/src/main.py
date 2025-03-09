#!/usr/bin/env python
"""
cursor-rules-cli: A tool to scan projects and suggest relevant Cursor rules

This module serves as the entry point for the CLI tool.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import json
from colorama import Fore, Style, init as init_colorama

# Initialize colorama
init_colorama()

# Import local modules
from cursor_rules_cli.scanner import scan_project, scan_package_files
from cursor_rules_cli.matcher import match_libraries
from cursor_rules_cli.downloader import download_rules
from cursor_rules_cli.installer import install_rules
from cursor_rules_cli.utils import (
    load_config, save_config, get_config_file, 
    load_project_config, save_project_config, get_project_config_file,
    merge_configs, validate_github_repo, DEFAULT_RULES_PATH
)

# Configure logging with colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            if record.levelno >= logging.WARNING:
                record.msg = f"{self.COLORS[levelname]}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(levelname)s: %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan your project and install relevant Cursor rules (.mdc files)."
    )
    
    parser.add_argument(
        "-d", "--directory",
        default=".",
        help="Project directory to scan (default: current directory)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing rules"
    )
    
    parser.add_argument(
        "--source",
        default="https://github.com/sanjeed5/awesome-cursor-rules-mdc",
        help="GitHub repository URL for downloading rules"
    )
    
    parser.add_argument(
        "--custom-repo",
        default=None,
        help="GitHub username/repo for a forked repository (e.g., 'username/repo')"
    )
    
    parser.add_argument(
        "--set-repo",
        action="store_true",
        help="Set custom repository without running scan"
    )
    
    parser.add_argument(
        "--rules-json",
        default=None,
        help="Path to custom rules.json file"
    )
    
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save current settings as default configuration"
    )
    
    parser.add_argument(
        "--save-project-config",
        action="store_true",
        help="Save current settings as project-specific configuration"
    )
    
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show current configuration"
    )
    
    parser.add_argument(
        "--quick-scan",
        action="store_true",
        help="Perform a quick scan (only check package files, not imports)"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of rules to display (default: 20)"
    )
    
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum relevance score for rules (0-1, default: 0.5)"
    )
    
    parser.add_argument(
        "--libraries",
        type=str,
        help="Comma-separated list of libraries to match directly (e.g., 'react,vue,django')"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def display_config(config: Dict[str, Any], global_config: Dict[str, Any], project_config: Dict[str, Any]):
    """Display the current configuration."""
    print(f"\n{Style.BRIGHT}{Fore.BLUE}Current Configuration:{Style.RESET_ALL}")
    
    # First display CLI-wide settings
    cli_wide_settings = ["custom_repo", "source"]
    if any(setting in config for setting in cli_wide_settings):
        print(f"\n{Fore.BLUE}CLI-wide settings:{Style.RESET_ALL}")
        for key in cli_wide_settings:
            if key in config:
                source = f" {Fore.GREEN}(global){Style.RESET_ALL}" if key in global_config else f" {Fore.YELLOW}(default){Style.RESET_ALL}"
                print(f"  {Fore.BLUE}{key}{Style.RESET_ALL}: {config[key]}{source}")
    
    # Then display project-specific settings
    project_settings = [k for k in config.keys() if k not in cli_wide_settings]
    if project_settings:
        print(f"\n{Fore.BLUE}Project-specific settings:{Style.RESET_ALL}")
        for key in project_settings:
            source = ""
            if key in project_config:
                source = f" {Fore.CYAN}(project){Style.RESET_ALL}"
            elif key in global_config:
                source = f" {Fore.GREEN}(global){Style.RESET_ALL}"
            else:
                source = f" {Fore.YELLOW}(default){Style.RESET_ALL}"
            
            print(f"  {Fore.BLUE}{key}{Style.RESET_ALL}: {config[key]}{source}")
    
    print()

def main():
    """Main entry point for the CLI."""
    # Parse command line arguments
    args = parse_args()
    
    # Convert directory to Path
    project_dir = Path(args.directory).resolve()
    
    # Load configurations
    global_config = load_config()
    project_config = load_project_config(project_dir)
    
    # Merge configurations (project config takes precedence)
    config = merge_configs(global_config, project_config)
    
    # Ensure rules_json is in config
    if "rules_json" not in config:
        config["rules_json"] = str(DEFAULT_RULES_PATH)
    
    # Ensure source is in config
    if "source" not in config:
        config["source"] = "https://raw.githubusercontent.com/sanjeed5/awesome-cursor-rules-mdc/main"
    
    # Handle direct library input if provided
    libraries_directly_provided = False
    if args.libraries:
        libraries = [lib.strip() for lib in args.libraries.split(",") if lib.strip()]
        if not libraries:
            logger.error("No valid libraries provided")
            return 1
        logger.info(f"Using directly provided libraries: {', '.join(libraries)}")
        detected_libraries = libraries
        libraries_directly_provided = True
    else:
        # Run the scanning phase
        try:
            # Use quick scan if requested
            scan_start_msg = "Quick scanning" if args.quick_scan else "Scanning"
            logger.info(f"{scan_start_msg} for libraries and frameworks...")
            
            # Scan project for libraries
            logger.info("Scanning for libraries and frameworks...")
            detected_libraries = scan_project(
                project_dir=project_dir,
                quick_scan=args.quick_scan,
                rules_path=config["rules_json"],
                use_cache=not args.force
            )
            
            # Get direct match libraries from package files
            direct_match_libraries = scan_package_files(Path(project_dir))
            
            logger.info(f"Detected {len(detected_libraries)} libraries/frameworks.")
            
            # Match libraries with rules
            logger.info("Finding relevant rules...")
            matching_rules = match_libraries(
                detected_libraries=detected_libraries,
                source_url=config["source"],
                direct_match_libraries=direct_match_libraries,
                custom_json_path=config["rules_json"],
                max_results=args.max_results,
                min_score=args.min_score
            )
            
            if not matching_rules:
                logger.warning("No matching libraries found for your project.")
                return 0
            
            logger.info(f"Found {Fore.GREEN}{len(matching_rules)}{Style.RESET_ALL} relevant rule files.")
            
            # Display and select rules to download
            selected_rules = display_matched_rules(matching_rules, args.max_results)
            
            if not selected_rules:
                logger.info("No rules selected. Exiting.")
                return 0
            
            # Download selected rules
            if args.dry_run:
                logger.info(f"{Fore.YELLOW}DRY RUN:{Style.RESET_ALL} Would download the following rules:")
                for rule in selected_rules:
                    logger.info(f"  - {Fore.CYAN}{rule}{Style.RESET_ALL}")
            else:
                try:
                    # Normalize source URL if needed (remove trailing slashes)
                    source_url = config["source"].rstrip('/')
                    
                    # Log source information
                    logger.info(f"Using source URL: {source_url}")
                    
                    # Download selected rules
                    downloaded_rules = download_rules(selected_rules, source_url)
                    
                    # Install downloaded rules
                    result = install_rules(downloaded_rules, force=args.force)
                    
                    if result["installed"]:
                        logger.info(f"{Fore.GREEN}✅ Successfully installed {len(result['installed'])} rules!{Style.RESET_ALL}")
                    
                    if result["failed"]:
                        logger.warning(f"{Fore.YELLOW}⚠️ Failed to install {len(result['failed'])} rules:{Style.RESET_ALL}")
                        for rule in result["failed"]:
                            logger.warning(f"  - {Fore.CYAN}{rule}{Style.RESET_ALL}")
                except Exception as e:
                    logger.error(f"An error occurred: {str(e)}")
                    return 1
        
        except KeyboardInterrupt:
            logger.info(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
            return 130
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    # Override config with command line arguments
    if args.custom_repo is not None:
        # Validate custom repo if provided
        if args.custom_repo and not validate_github_repo(args.custom_repo):
            logger.error(f"{Fore.RED}Invalid GitHub repository: {args.custom_repo}{Style.RESET_ALL}")
            logger.error(f"{Fore.RED}Repository must exist and contain a rules.json file.{Style.RESET_ALL}")
            return 1
        config["custom_repo"] = args.custom_repo
    elif "custom_repo" not in config:
        config["custom_repo"] = None
        
    if args.rules_json is not None:
        config["rules_json"] = args.rules_json
    elif "rules_json" not in config:
        config["rules_json"] = str(DEFAULT_RULES_PATH)
        
    if args.source != "https://raw.githubusercontent.com/sanjeed5/awesome-cursor-rules-mdc/main":
        config["source"] = args.source
    elif "source" not in config:
        config["source"] = "https://raw.githubusercontent.com/sanjeed5/awesome-cursor-rules-mdc/main"
    
    # Set custom repository without running scan if requested
    if args.set_repo:
        if args.custom_repo is None:
            logger.error(f"{Fore.RED}Please specify a custom repository with --custom-repo.{Style.RESET_ALL}")
            return 1
        
        global_config["custom_repo"] = config["custom_repo"]
        save_config(global_config)
        logger.info(f"{Fore.GREEN}Custom repository set to: {config['custom_repo']}{Style.RESET_ALL}")
        return 0
    
    # Show configuration if requested
    if args.show_config:
        display_config(config, global_config, project_config)
        return 0
    
    # Save configuration if requested
    if args.save_config:
        # For custom repo, only save to global config, not project config
        global_config_to_save = global_config.copy()
        if "custom_repo" in config:
            global_config_to_save["custom_repo"] = config["custom_repo"]
        if "source" in config:
            global_config_to_save["source"] = config["source"]
        
        save_config(global_config_to_save)
        logger.info(f"{Fore.GREEN}Global configuration saved successfully.{Style.RESET_ALL}")
        if not args.directory or args.directory == ".":
            return 0
    
    if args.save_project_config:
        # Don't include custom_repo in project config
        project_config_to_save = {k: v for k, v in config.items() if k not in ["custom_repo", "source"]}
        save_project_config(project_dir, project_config_to_save)
        logger.info(f"{Fore.GREEN}Project configuration saved to {project_dir / '.cursor-rules-cli.json'}{Style.RESET_ALL}")
        if not args.directory or args.directory == ".":
            return 0
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"{Style.BRIGHT}{Fore.BLUE}Cursor Rules CLI{Style.RESET_ALL}")
    logger.info(f"Scanning project directory: {Fore.CYAN}{os.path.abspath(args.directory)}{Style.RESET_ALL}")
    
    # Handle custom repository if specified
    source_url = config["source"]
    if config["custom_repo"]:
        source_url = f"https://raw.githubusercontent.com/{config['custom_repo']}/main"
        logger.info(f"Using custom repository: {Fore.CYAN}{config['custom_repo']}{Style.RESET_ALL}")
    
    # Run the scanning phase only if libraries were not directly provided
    try:
        # Skip scanning if libraries were directly provided
        if not libraries_directly_provided:
            # Use quick scan if requested
            scan_start_msg = "Quick scanning" if args.quick_scan else "Scanning"
            logger.info(f"{scan_start_msg} for libraries and frameworks...")
            
            # Scan project for libraries
            logger.info("Scanning for libraries and frameworks...")
            detected_libraries = scan_project(
                project_dir=project_dir,
                quick_scan=args.quick_scan,
                rules_path=config["rules_json"],
                use_cache=not args.force
            )
            
            # Get direct match libraries from package files
            direct_match_libraries = scan_package_files(Path(project_dir))
            
            logger.info(f"Detected {len(detected_libraries)} libraries/frameworks.")
        else:
            # For directly provided libraries, we don't need to scan package files
            direct_match_libraries = set(detected_libraries)
            logger.info(f"{Fore.CYAN}Skipping project scan - using directly provided libraries only{Style.RESET_ALL}")
        
        # Match libraries with rules
        logger.info("Finding relevant rules...")
        matching_rules = match_libraries(
            detected_libraries=detected_libraries,
            source_url=source_url,
            direct_match_libraries=direct_match_libraries,
            custom_json_path=config["rules_json"],
            max_results=args.max_results,
            min_score=args.min_score
        )
        
        if not matching_rules:
            logger.warning("No matching libraries found for your project.")
            return 0
        
        logger.info(f"Found {Fore.GREEN}{len(matching_rules)}{Style.RESET_ALL} relevant rule files.")
        
        # Display and select rules to download
        selected_rules = display_matched_rules(matching_rules, args.max_results)
        
        if not selected_rules:
            logger.info("No rules selected. Exiting.")
            return 0
        
        # Download selected rules
        if args.dry_run:
            logger.info(f"{Fore.YELLOW}DRY RUN:{Style.RESET_ALL} Would download the following rules:")
            for rule in selected_rules:
                logger.info(f"  - {Fore.CYAN}{rule}{Style.RESET_ALL}")
        else:
            downloaded_rules = download_rules(selected_rules, source_url)
            
            # Install downloaded rules
            result = install_rules(downloaded_rules, force=args.force)
            
            if result["installed"]:
                logger.info(f"{Fore.GREEN}✅ Successfully installed {len(result['installed'])} rules!{Style.RESET_ALL}")
            
            if result["failed"]:
                logger.warning(f"{Fore.YELLOW}⚠️ Failed to install {len(result['failed'])} rules:{Style.RESET_ALL}")
                for rule in result["failed"]:
                    logger.warning(f"  - {Fore.CYAN}{rule}{Style.RESET_ALL}")
            
        return 0
    
    except KeyboardInterrupt:
        logger.info(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
        return 130
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def group_rules_by_category(rules: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group rules by their category.
    
    Args:
        rules: List of rule dictionaries with category information
        
    Returns:
        Dictionary mapping categories to lists of rules
    """
    categories = {}
    
    for rule in rules:
        category = rule.get("category", "other")
        if category not in categories:
            categories[category] = []
        categories[category].append(rule)
    
    return categories

def get_category_display_name(category: str) -> str:
    """
    Get a display name for a category.
    
    Args:
        category: Category key
        
    Returns:
        Display name for the category
    """
    category_names = {
        "development": "Development Tools",
        "frontend": "Frontend Frameworks & Libraries",
        "backend": "Backend Frameworks & Libraries",
        "database": "Database & ORM",
        "ai_ml": "AI & Machine Learning",
        "devops": "DevOps & Cloud",
        "utilities": "Utilities & CLI Tools",
        "other": "Other Libraries",
    }
    
    return category_names.get(category, category.title())

def display_matched_rules(matched_rules: List[Dict[str, Any]], max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Display matched rules and return selected rule objects.
    
    Args:
        matched_rules: List of matched rules
        max_results: Maximum number of rules to display
        
    Returns:
        List of selected rule objects
    """
    if not matched_rules:
        logger.info("No relevant rules found for your project.")
        return []
    
    # Group rules by category (direct_match vs others)
    direct_matches = []
    other_matches = []
    
    for rule in matched_rules:
        # Check if this is a direct match from package files
        if rule.get("is_direct_match", False):
            direct_matches.append(rule)
        else:
            other_matches.append(rule)
    
    # Sort each group by relevance score
    direct_matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    other_matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Combine the lists with direct matches first
    sorted_rules = direct_matches + other_matches
    
    # Limit to max_results
    display_rules = sorted_rules[:max_results]
    
    # Display rules
    print(f"\n{Style.BRIGHT}{Fore.BLUE}Available Cursor rules for your project:{Style.RESET_ALL}\n")
    
    # Display direct matches first
    if direct_matches:
        print(f"{Style.BRIGHT}{Fore.GREEN}Direct Dependencies:{Style.RESET_ALL}")
        for i, rule in enumerate([r for r in display_rules if r.get("is_direct_match", False)], 1):
            tags = f"[{', '.join(rule.get('tags', []))}]" if rule.get('tags') else ""
            score = f"({rule['relevance_score']:.2f})"
            print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} {Fore.CYAN}{rule['rule']}{Style.RESET_ALL} {tags} {score}")
    
    # Display other matches
    if other_matches and any(not r.get("is_direct_match", False) for r in display_rules):
        print(f"\n{Style.BRIGHT}{Fore.YELLOW}Other Relevant Rules:{Style.RESET_ALL}")
        # Continue numbering from where direct matches left off
        start_idx = len([r for r in display_rules if r.get("is_direct_match", False)]) + 1
        for i, rule in enumerate([r for r in display_rules if not r.get("is_direct_match", False)], start_idx):
            tags = f"[{', '.join(rule.get('tags', []))}]" if rule.get('tags') else ""
            score = f"({rule['relevance_score']:.2f})"
            print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} {Fore.CYAN}{rule['rule']}{Style.RESET_ALL} {tags} {score}")
    
    # Get user selection
    print(f"\n{Style.BRIGHT}Select rules to install:{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}* Enter comma-separated numbers (e.g., 1,3,5){Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}* Type 'all' to select all rules{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}* Type 'category:name' to select all rules in a category (e.g., 'category:development'){Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}* Type 'none' to cancel{Style.RESET_ALL}")
    
    selection = input(f"{Fore.GREEN}> {Style.RESET_ALL}").strip().lower()
    
    if selection == "none":
        logger.info("No rules selected. Exiting.")
        return []
    
    if selection == "all":
        return display_rules
    
    if selection.startswith("category:"):
        category = selection.split(":", 1)[1]
        return [
            rule for rule in display_rules
            if category in rule.get("tags", [])
        ]
    
    try:
        indices = [int(idx.strip()) for idx in selection.split(",") if idx.strip()]
        return [display_rules[idx - 1] for idx in indices if 1 <= idx <= len(display_rules)]
    except (ValueError, IndexError):
        logger.error(f"{Fore.RED}Invalid selection. Please try again.{Style.RESET_ALL}")
        return display_matched_rules(matched_rules, max_results)

if __name__ == "__main__":
    sys.exit(main()) 