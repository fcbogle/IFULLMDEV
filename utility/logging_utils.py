# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-07
# Description: logging_utils.py
# -----------------------------------------------------------------------------

import logging
import colorlog

def get_logger(name: str | None = None) -> logging.Logger:
    """
    Returns a colourised logger using colorlog for better readability.
    """
    base_name = "ifu_rag_llm"
    full_name = f"{base_name}.{name}" if name else base_name
    logger = logging.getLogger(full_name)

    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = colorlog.ColoredFormatter(
            fmt=(
                "%(log_color)s%(asctime)s [%(levelname)s] "
                "%(name)s:%(lineno)d:%(reset)s %(message_log_color)s%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            secondary_log_colors={
                "message": {
                    "INFO": "white",
                    "WARNING": "yellow",
                    "ERROR": "light_red",
                    "CRITICAL": "red",
                }
            },
            style="%",
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger