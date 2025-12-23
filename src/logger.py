import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Initializes a logger with a standard format and console output.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(level)

        # Format: Time - Name - Level - Message
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
