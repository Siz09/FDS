"""Structured logging configuration using structlog."""
import logging
import os
import sys

import structlog


def setup_logging() -> None:
    """Configure structlog for dev (ConsoleRenderer) or prod (JSONRenderer).

    Reads APP_ENV env var. Anything other than 'production' uses colored
    console output. Falls back to production JSON if unset.
    """
    env = os.getenv("APP_ENV", "production")

    shared_processors: list = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]

    renderer = (
        structlog.dev.ConsoleRenderer()
        if env != "production"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
