"""
ViFinQA Structured Logger
──────────────────────────────────────────────────────────────
JSON-formatted structured logging for pipeline observability.
"""

import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach extra structured fields
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(
    name: str,
    level: int = logging.INFO,
    json_format: bool = True,
) -> logging.Logger:
    """Create a structured logger.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        json_format: If True, output JSON lines; otherwise human-readable.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def log_with_data(
    logger: logging.Logger,
    level: int,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a message with attached structured data."""
    record = logger.makeRecord(
        logger.name, level, "(unknown)", 0, message, (), None
    )
    if data:
        record.extra_data = data  # type: ignore[attr-defined]
    logger.handle(record)
