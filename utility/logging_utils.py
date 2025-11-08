# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-07
# Description: logging_utils.py
# -----------------------------------------------------------------------------

# logging_utils.py
import logging
import colorlog


def _create_logger(full_name: str) -> logging.Logger:
    """
    Internal helper to create/configure a logger with a given full name.
    """
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


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Generic logger that does not include a class name.
    """
    base_name = "ifu_rag_llm"
    full_name = f"{base_name}.{name}" if name else base_name
    return _create_logger(full_name)


def get_class_logger(cls: type) -> logging.Logger:
    """
    Returns a logger whose name includes module + class, e.g.:

      ifu_rag_llm.health.OpenAIHealth.OpenAIHealth
      ifu_rag_llm.health.EmbeddingHealth.EmbeddingHealth
    """
    base_name = "ifu_rag_llm"
    module = getattr(cls, "__module__", "unknown_module")
    classname = getattr(cls, "__name__", "UnknownClass")

    full_name = f"{base_name}.{module}.{classname}"
    return _create_logger(full_name)

