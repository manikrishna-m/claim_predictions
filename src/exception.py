import sys
from src.logger import logger

class CustomException(Exception):
    """
    A custom exception class for handling application-specific errors.
    This class automatically logs the error along with detailed traceback information.

    Attributes:
        error_message (str): The formatted error message to be logged and raised.
    """

    def __init__(self, error_message, exc_info=None):
        """
        Initializes the custom exception with a provided error message and optional exception information.
        It logs the error message with a detailed traceback.

        :param error_message: The main error message that describes the error.
        :param exc_info: Exception information (default is None). This is used to extract traceback details.
        """
        super().__init__(error_message)
        self.error_message = self.get_detailed_error(error_message, exc_info)
        logger.error(self.error_message)

    @staticmethod
    def get_detailed_error(error_message, exc_info):
        """
        Extracts and formats the traceback information from the exception.
        If no exception information is provided, it uses `sys.exc_info()` to retrieve the current exception info.

        :param error_message: The main error message to include in the detailed error.
        :param exc_info: Exception information (default is None).
        :returns: A formatted string that includes the error message, file name, and line number.
        """
        if exc_info is None:
            exc_info = sys.exc_info()

        _, _, exc_tb = exc_info
        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "unknown"
            line_number = "unknown"

        return f"Error: {error_message} | File: {file_name} | Line: {line_number}"

    def __str__(self):
        """
        Returns the formatted error message when the exception is raised.
        """
        return self.error_message
