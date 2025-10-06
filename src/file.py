import sys
import traceback
from logger_config import get_logger

logger = get_logger(__name__)

class ExceptionLogger:
    """
    Logs exceptions with detailed information.
    Does not raise, return, or exit — program continues.
    """

    def __init__(self, error_message, exc_info=None, log_traceback=True):
        """
        :param error_message: Description of the error
        :param exc_info: Optional exception info tuple (default: sys.exc_info())
        :param log_traceback: If True, logs full traceback
        """
        self.exc_info = exc_info or sys.exc_info()
        self.error_message = self.format_error(error_message)

        # Log main error
        logger.error(self.error_message)

        # Optionally log full traceback
        if log_traceback and self.exc_info[2]:
            formatted_traceback = "".join(traceback.format_exception(*self.exc_info))
            logger.error("Full traceback:\n%s", formatted_traceback)

    def format_error(self, error_message):
        exc_type, _, exc_tb = self.exc_info
        exc_name = exc_type.__name__ if exc_type else "Exception"

        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "unknown"
            line_number = "unknown"

        return f"{exc_name}: {error_message} | File: {file_name} | Line: {line_number}"


import os
import sys
import logging

def get_logger(name="ml_pipeline_logger"):
    """
    Returns a configured logger object.

    Features:
    - Logs to both file and console
    - Each process writes to its own file using PID to avoid conflicts
    - Prevents duplicate handlers
    """
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    pid = os.getpid()
    log_file_path = os.path.join(LOG_DIR, f"{name}_{pid}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # Handlers
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | PID:%(process)d | %(filename)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
