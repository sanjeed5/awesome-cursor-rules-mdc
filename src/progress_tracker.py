import json
from logger import logger
from constants import SCRIPT_DIR
from typing import Dict, List

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
