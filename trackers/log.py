import logging
import os
import sys
from typing import ClassVar

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LogFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    blue = "\x1b[34;1m"  # Brighter blue
    yellow = "\x1b[33;1m"  # Brighter yellow
    red = "\x1b[31;1m"  # Brighter red
    bold_red = "\x1b[31;1m"  # Same as red for consistency
    reset = "\x1b[0m"

    base_format = "%(asctime)s - %(name)s - "
    level_message_format = "%(levelname)s: %(message)s"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: grey + base_format + level_message_format + reset,
        logging.INFO: blue + base_format + level_message_format + reset,
        logging.WARNING: yellow + base_format + level_message_format + reset,
        logging.ERROR: red + base_format + level_message_format + reset,
        logging.CRITICAL: bold_red + base_format + level_message_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


log_filename = os.environ.get("TRACKERS_LOG_FILENAME", "trackers.log")
log_level_name = os.environ.get("TRACKERS_LOG_LEVEL", "INFO").upper()
log_output_type = os.environ.get("TRACKERS_LOG_OUTPUT", "stderr").lower()

log_level = LOG_LEVELS.get(log_level_name, logging.INFO)

if log_output_type == "file":
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_filename,
        filemode="a",
    )
else:
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(LogFormatter())
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance with the specified name.

    Args:
        name (str): The name for the logger, typically __name__.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
