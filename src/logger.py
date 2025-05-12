import os
import logging
from datetime import datetime

# Directory to store log files
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Create the log directory if it doesn't exist

# Define the log filename with the current timestamp for uniqueness
log_filename = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Initialize the logger with the name "ml_pipeline_logger"
logger = logging.getLogger("ml_pipeline_logger")
logger.setLevel(logging.INFO)  # Set the logging level to INFO

# Prevent adding duplicate handlers if the logger is re-imported
if not logger.hasHandlers():
    # Create file handler to write logs to a file
    file_handler = logging.FileHandler(log_filename)
    
    # Create stream handler to output logs to the console (stdout)
    stream_handler = logging.StreamHandler()

    # Define the log message format including time, level, filename, line number, and message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
    )
    
    # Set the formatter for both handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add both handlers to the logger (file and stream)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
