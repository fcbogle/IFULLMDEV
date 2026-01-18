# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2026-01-09
# Description: settings.py
# -----------------------------------------------------------------------------
import os

def _env(name: str, default: str) -> str:
    """Read env var safely and strip whitespace."""
    return (os.getenv(name) or default).strip()

# ACTIVE_CORPUS_ID = os.getenv("IFU_CORPUS_ID", "QADocuments")

# ---- Corpus ----
# New preferred env var:
ACTIVE_CORPUS_ID = _env("IFU_ACTIVE_CORPUS_ID", "")

# Backwards-compatible fallback to your existing env var name:
if not ACTIVE_CORPUS_ID:
    ACTIVE_CORPUS_ID = _env("IFU_CORPUS_ID", "QADocuments")


# ---- Blob storage (Azure Blob) ----
BLOB_CONTAINER_DEFAULT = _env("IFU_BLOB_CONTAINER_DEFAULT", "")

# Backwards-compatible fallback to your existing UI env var name if you used it:
if not BLOB_CONTAINER_DEFAULT:
    BLOB_CONTAINER_DEFAULT = _env("IFU_DEFAULT_CONTAINER", "ifu-docs-test")


# ---- Vector storage (Chroma collection) ----
VECTOR_COLLECTION_DEFAULT = _env("IFU_VECTOR_COLLECTION_DEFAULT", "")

# Backwards-compatible default: if you currently reuse the container name
if not VECTOR_COLLECTION_DEFAULT:
    VECTOR_COLLECTION_DEFAULT = BLOB_CONTAINER_DEFAULT