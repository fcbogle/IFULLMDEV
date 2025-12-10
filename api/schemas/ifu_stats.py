# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: ifu_stats.py
# -----------------------------------------------------------------------------
from typing import Optional, List

from pydantic import BaseModel


class DocumentStats(BaseModel):
    doc_id: str
    chunk_count: int
    page_count: int | None = None
class BlobStats(BaseModel):
    blob_name: str
    size: Optional[int] = None
    content_type: Optional[str] = None
    last_modified: Optional[str] = None

class IFUStatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    total_documents: int
    documents: List[DocumentStats]

    # blob stats
    blob_container: Optional[str] = None
    total_blobs: int = 0
    blobs: List[BlobStats] = []

