import os
import logging

from datetime import datetime
from rich.logging import RichHandler


def setup_logger(name="main", log_dir="logs", log_prefix="train", use_file=True):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime(f"{log_prefix}_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    # Avoid duplicate logs if logger is re-instantiated
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # Rich for console logging
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setLevel(logging.INFO)

    # Optional file logging
    handlers = [rich_handler]
    if use_file:
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Set handlers
    for handler in handlers:
        logger.addHandler(handler)

    return logger


logger = setup_logger(log_prefix="vivqa_co_training", 
                      use_file=False)