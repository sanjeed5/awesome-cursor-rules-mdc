"""
Logging configuration for the MDC generation process.
"""

import logging
import datetime
from pathlib import Path

# Get the script directory for absolute path resolution
SCRIPT_DIR = Path(__file__).parent.absolute()

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = Path(SCRIPT_DIR).parent / "logs"
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