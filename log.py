import logging
import sys
from datetime import datetime
import os

# Create log directory if it doesn't exist
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)

# Generate logging file path with current date
current_date = datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD, e.g., 20250407
logging_file_path = os.path.join(log_dir, f"search_pipe_{current_date}.log")

# logging_file_path = os.path.join(log_dir, f"server_pipe_test.log")

# Configure handlers
handlers = [logging.FileHandler(logging_file_path), logging.StreamHandler(sys.stdout)]

# Set logging level (DEBUG overrides INFO)
level = logging.INFO
# level = logging.DEBUG

# Configure basic logging
logging.basicConfig(
    level=level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
)

# Create logger
logger = logging.getLogger(__name__)

# Example usage
logger.debug("This is a debug message")
logger.info("This is an info message")
