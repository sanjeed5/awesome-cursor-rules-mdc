import os
import json
from pathlib import Path
from typing import List, Dict
import logging
from dotenv import load_dotenv
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
from pydantic import BaseModel, Field
import re
# Enable JSON schema validation in LiteLLM
import litellm
litellm.enable_json_schema_validation = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Rate limit configuration - 2000 requests per minute
CALLS_PER_MINUTE = 2000
ONE_MINUTE = 60

# Load MDC instructions
MDC_INSTRUCTIONS = Path("mdc-instructions.txt").read_text()

class MDCRule(BaseModel):
    name: str = Field(..., description="A meaningful name for the rule that reflects its purpose")
    glob_pattern: str = Field(..., description="Glob pattern to specify which files/folders the rule applies to")
    description: str = Field(..., description="A clear description of what the rule does and when it should be applied")
    content: str = Field(..., description="The actual rule content")

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=ONE_MINUTE)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_file_content(file_content: str, file_name: str) -> List[MDCRule]:
    """Analyze file content and split into multiple rules if needed."""
    try:
        prompt = f"""Create Cursor rule files (.mdc) by following these guidelines:

{MDC_INSTRUCTIONS}

Key requirements:
- Split into multiple distinct rules when different glob patterns or contexts are present
- Each rule must have a unique glob pattern targeting specific files/folders
- Create separate rules for different contexts (e.g., different frameworks, file types, or project areas)
- Maintain atomic rules - one clear purpose per rule
- Preserve all relevant context from the original content

Original content from '{file_name}':
{file_content}

Analyze the content and split it into logical rules based on:
1. Different target file patterns (globs)
2. Different frameworks/technologies mentioned
3. Different directories or project areas
4. Distinct functionality or purpose

Provide each rule with:
- A specific glob pattern
- Clear context description
- Only the relevant instructions for that pattern/context

Return a valid JSON array where each rule is an object with exactly the following keys:
  - name: a short descriptive name for the rule
  - glob_pattern: a valid glob pattern as a string
  - description: a 1-2 sentence description of what the rule does
  - content: the formatted rule content using yaml frontmatter
Do not include any other keys."""

        response = completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "array",
                    "items": {
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
                                "description": "Formatted rule content using yaml frontmatter"
                            }
                        },
                        "required": ["name", "glob_pattern", "description", "content"]
                    }
                }
            }
        )
        
        # Parse the JSON response
        try:
            json_response = json.loads(response.choices[0].message.content)
            # Create MDCRule instances from the JSON
            rules = [MDCRule(**rule) for rule in json_response]
            return rules
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid JSON response: {response.choices[0].message.content}")
            raise ValueError(f"Invalid response format: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {str(e)}")
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
        except Exception:
            pass
    
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

class ProgressTracker:
    def __init__(self, tracking_file: str = "conversion_progress.json"):
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
    
    def is_folder_processed(self, folder_path: str) -> bool:
        """Check if a folder has been successfully processed."""
        return self.progress.get(str(folder_path)) == "completed"
    
    def mark_folder_completed(self, folder_path: str):
        """Mark a folder as successfully processed."""
        self.progress[str(folder_path)] = "completed"
        self.save_progress()
    
    def mark_folder_failed(self, folder_path: str):
        """Mark a folder as failed."""
        self.progress[str(folder_path)] = "failed"
        self.save_progress()

def process_folder(
    source_folder: str,
    output_base_folder: str,
    same_folder: bool = True,
    test_mode: bool = False
) -> None:
    """
    Process all .cursorrules files in the source folder and create .mdc versions.
    
    Args:
        source_folder: Path to the source folder containing rules
        output_base_folder: Path to the output folder for new structure
        same_folder: Whether to also save .mdc files in the same folder as source
        test_mode: If True, only process 2 folders
    """
    source_path = Path(source_folder)
    output_base_path = Path(output_base_folder)
    progress_tracker = ProgressTracker()
    
    if not source_path.exists():
        raise ValueError(f"Source folder {source_folder} does not exist")
    
    # Create output base folder if it doesn't exist
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # Get all subdirectories
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    if test_mode:
        subdirs = subdirs[:2]  # Only process first 2 folders in test mode
        logger.info("Running in test mode - processing only 2 folders")
    
    total_folders = len(subdirs)
    processed_count = 0
    
    for subdir in subdirs:
        if progress_tracker.is_folder_processed(str(subdir)):
            logger.info(f"Skipping already processed folder: {subdir}")
            processed_count += 1
            continue
            
        logger.info(f"Processing folder: {subdir} ({processed_count + 1}/{total_folders})")
        
        try:
            # Create corresponding output subdirectory
            output_subdir = output_base_path / subdir.name
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Process only .cursorrules files in the subdirectory
            cursorrules_file = subdir / ".cursorrules"
            if cursorrules_file.is_file():
                # Read original content
                with open(cursorrules_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get rules from LLM
                rules = analyze_file_content(content, cursorrules_file.name)
                
                # Create .mdc files for each rule
                for i, rule in enumerate(rules):
                    # Create filename based on rule name
                    safe_name = re.sub('[^a-z0-9-]+', '-', rule.name.lower()).strip('-')
                    mdc_filename = f"{safe_name}.mdc"
                    
                    # Save to separate folder
                    output_path = output_subdir / mdc_filename
                    create_mdc_file(rule, output_path)
                    
                    # Optionally save to same folder
                    if same_folder:
                        same_folder_path = cursorrules_file.parent / mdc_filename
                        create_mdc_file(rule, same_folder_path)
                
                progress_tracker.mark_folder_completed(str(subdir))
                processed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing {cursorrules_file}: {str(e)}")
            progress_tracker.mark_folder_failed(str(subdir))
            continue

def main():
    """Main function to run the conversion process."""
    source_folder = "awesome-cursorrules/rules"
    output_folder = "awesome-cursor-rules-mdc"
    
    # Check if source folder exists
    if not os.path.exists(source_folder):
        logger.error(f"Source folder {source_folder} does not exist")
        return
    
    # Process with test mode first
    test_mode = False  # Change to False for processing all folders
    try:
        process_folder(source_folder, output_folder, same_folder=True, test_mode=test_mode)
        logger.info("Processing completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 