"""Structured logging utilities for AdaMule."""

import logging
import sys
from pathlib import Path
from typing import Optional
from adamule.utils.config import get_project_root


def get_logger(name: str = "adamule", log_file: Optional[str | Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a structured logger.
    
    Args:
        name: Name of the logger.
        log_file: Optional path to output log file.
        level: Logging level (default: logging.INFO).
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            path = Path(log_file)
            if not path.is_absolute():
                path = get_project_root() / path
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(path), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger
