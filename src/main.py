import argparse
import os
import shutil
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv

from config import CONFIG
from constants import SCRIPT_DIR
from logger import logger
from mdc_generator import process_rules_json

# Enable JSON schema validation in LiteLLM
litellm.enable_json_schema_validation = True

# Load environment variables
load_dotenv()


def validate_environment_variables() -> bool:
    """Validate that all required environment variables are set."""
    required_vars = []

    # Check for LiteLLM API key (depends on the model being used)
    model = CONFIG.api.llm_model
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
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        return False

    return True


def main():
    """Main function to run the MDC generation process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate MDC files from rules.json")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (process only one library)",
    )
    parser.add_argument(
        "--tag", type=str, help="Process only libraries with a specific tag"
    )
    parser.add_argument("--library", type=str, help="Process only a specific library")
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory for MDC files"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--workers",
        type=int,
        help=f"Number of parallel workers (default: {CONFIG.processing.max_workers})",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        help=f"API rate limit calls per minute (default: {CONFIG.api.rate_limit_calls})",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to save log files"
    )
    parser.add_argument(
        "--exa-results-dir",
        type=str,
        default=None,
        help="Directory to save Exa results",
    )
    parser.add_argument(
        "--regenerate-all",
        action="store_true",
        help="Regenerate all libraries, including previously completed ones",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable extra debug output"
    )
    args = parser.parse_args()

    # Set up verbose logging if requested
    if args.verbose or args.debug:
        logger.getLogger().setLevel(logger.DEBUG)
        logger.setLevel(logger.DEBUG)

    # Override config with command line arguments
    if args.workers:
        CONFIG.processing.max_workers = args.workers

    if args.rate_limit:
        CONFIG.api.rate_limit_calls = args.rate_limit

    if args.exa_results_dir:
        CONFIG.paths.exa_results_dir = args.exa_results_dir

    # Set paths with absolute paths
    rules_json_path = SCRIPT_DIR / Path(CONFIG.paths.rules_json)
    output_dir = (
        args.output if args.output else SCRIPT_DIR / Path(CONFIG.paths.output_dir)
    )

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
        )
        logger.info("MDC generation completed successfully!")

        # Copy rules.json to cursor-rules-cli folder
        # This is the source of truth - the CLI package will use this during setup
        cursor_rules_cli_dir = Path(SCRIPT_DIR).parent / "cursor-rules-cli"
        cursor_rules_cli_dir.mkdir(exist_ok=True)

        target_rules_json = cursor_rules_cli_dir / "rules.json"

        try:
            shutil.copy2(rules_json_path, target_rules_json)
            logger.info(
                f"Successfully copied rules.json (source of truth) to {target_rules_json}"
            )

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
            logger.error(
                f"Failed to copy rules.json to cursor-rules-cli folder: {str(e)}"
            )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
