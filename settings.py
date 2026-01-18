# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2026-01-09
# Updated: 2026-01-18
# Description: settings.py
# -----------------------------------------------------------------------------
import os
from typing import Any, Dict, Optional


def _env(name: str, default: str = "") -> str:
    """Read env var safely and strip whitespace."""
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name, "")
    if v == "":
        return default
    try:
        return int(v)
    except ValueError as e:
        raise RuntimeError(f"Env var {name} must be an int, got {v!r}") from e


def _env_float(name: str, default: float) -> float:
    v = _env(name, "")
    if v == "":
        return default
    try:
        return float(v)
    except ValueError as e:
        raise RuntimeError(f"Env var {name} must be a float, got {v!r}") from e


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name, "")
    if v == "":
        return default
    v = v.lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise RuntimeError(f"Env var {name} must be a boolean, got {v!r}")


# -----------------------------------------------------------------------------
# Corpus
# -----------------------------------------------------------------------------
# New preferred env var:
ACTIVE_CORPUS_ID = _env("IFU_ACTIVE_CORPUS_ID", "")

# Backwards-compatible fallback to older env var name:
if not ACTIVE_CORPUS_ID:
    ACTIVE_CORPUS_ID = _env("IFU_CORPUS_ID", "QADocuments")


# -----------------------------------------------------------------------------
# Blob storage (Azure Blob) - optional for vector-only deployments
# -----------------------------------------------------------------------------
BLOB_CONTAINER_DEFAULT = _env("IFU_BLOB_CONTAINER_DEFAULT", "")

# Backwards-compatible fallback to your existing UI env var name if you used it:
if not BLOB_CONTAINER_DEFAULT:
    BLOB_CONTAINER_DEFAULT = _env("IFU_DEFAULT_CONTAINER", "ifu-docs-test")


# -----------------------------------------------------------------------------
# Vector storage (Chroma collection name)
# -----------------------------------------------------------------------------
# New preferred env var:
VECTOR_COLLECTION_DEFAULT = _env("IFU_VECTOR_COLLECTION", "")

# Backwards-compatible env var name (if you used it previously):
if not VECTOR_COLLECTION_DEFAULT:
    VECTOR_COLLECTION_DEFAULT = _env("IFU_VECTOR_COLLECTION_DEFAULT", "")

# Backwards-compatible default: historically you reused the blob container name
if not VECTOR_COLLECTION_DEFAULT:
    VECTOR_COLLECTION_DEFAULT = BLOB_CONTAINER_DEFAULT


# -----------------------------------------------------------------------------
# Ask() defaults (env-controlled)
# -----------------------------------------------------------------------------
ASK_DEFAULTS: Dict[str, Any] = {
    "n_results": _env_int("IFU_DEFAULT_N_RESULTS", 5),
    "temperature": _env_float("IFU_DEFAULT_TEMPERATURE", 0.0),
    "max_tokens": _env_int("IFU_DEFAULT_MAX_TOKENS", 512),
    "tone": _env("IFU_DEFAULT_TONE", "neutral"),
    "language": _env("IFU_DEFAULT_LANGUAGE", "en"),
    # blank/empty means "auto" (let IFUChatService resolve mode from triggers)
    "mode": _env("IFU_DEFAULT_MODE", ""),
}

# Optional: allow you to enforce that a non-empty "where" must be provided
ASK_FORCE_WHERE = _env_bool("IFU_ASK_FORCE_WHERE", False)

# Optional: cap context length if you want this centrally controlled later
MAX_CONTEXT_CHARS = _env_int("IFU_MAX_CONTEXT_CHARS", 12000)


# -----------------------------------------------------------------------------
# Sanity checks (tunable)
# -----------------------------------------------------------------------------
if not ACTIVE_CORPUS_ID:
    raise RuntimeError("ACTIVE_CORPUS_ID resolved to empty value")

# Blob container may be unused in vector-only mode, so we do not hard-fail by default.
# If you want to enforce it, set IFU_REQUIRE_BLOB_CONTAINER=true
REQUIRE_BLOB_CONTAINER = _env_bool("IFU_REQUIRE_BLOB_CONTAINER", False)
if REQUIRE_BLOB_CONTAINER and not BLOB_CONTAINER_DEFAULT:
    raise RuntimeError("BLOB_CONTAINER_DEFAULT resolved to empty value but is required")

if not VECTOR_COLLECTION_DEFAULT:
    raise RuntimeError("VECTOR_COLLECTION_DEFAULT resolved to empty value")
