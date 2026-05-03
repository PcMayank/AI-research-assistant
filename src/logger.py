"""
logger.py — Loguru-based logger setup for the whole project.
"""
import sys
from pathlib import Path
from loguru import logger
from src.config import settings

Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)

# Remove default handler
logger.remove()

# Console — pretty coloured output
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    colorize=True,
)

# File — full structured log
logger.add(
    settings.log_file,
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    enqueue=True,
)

__all__ = ["logger"]
