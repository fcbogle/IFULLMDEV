# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: ifu_stats.py
# -----------------------------------------------------------------------------
from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel, Field


class DocumentStats(BaseModel):
    doc_id: str

    doc_name: Optional[str] = None

    chunk_count: int
    page_count: int | None = None
    last_modified: datetime | None = None
    document_type: str | None = None

    primary_lang: Optional[str] = None
    lang_counts: Optional[Dict[str, int]] = None


class BlobStats(BaseModel):
    blob_name: str
    size: int | None = None
    content_type: str | None = None
    last_modified: datetime | None = None  # keep consistent


class IFUStatsResponse(BaseModel):
    collection_name: str

    corpus_id: Optional[str] = None

    total_chunks: int
    total_documents: int
    documents: List[DocumentStats]

    # blob stats
    blob_container: str | None = None
    total_blobs: int = 0
    blobs: List[BlobStats] = Field(default_factory=list)


