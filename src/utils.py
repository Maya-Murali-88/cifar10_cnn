import logging
from logging.handlers import RotatingFileHandler
import os
import torch

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    Creates a logger that logs to BOTH console and a rotating file.
    Safe to use in FastAPI (avoids duplicate handlers).
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers (important for FastAPI reload)
    if logger.handlers:
        return logger

    # Log format
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Rotating file handler (max ~2MB / keeps 3 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_device():
    """Returns GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
