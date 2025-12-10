# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: types.py
# -----------------------------------------------------------------------------
from typing import TypedDict, List

class DocumentEntry(TypedDict):
    ddoc_id: str
    chunk_count: int

class BlobEntry(TypedDict):
    blob_name: str
    size_bytes: int

class IFUStatsDict(TypedDict):
    collection_name: str
    total_chunks: int
    total_documents: int
    documents: List[DocumentEntry]
    blob_container: str
    blobs: List[BlobEntry]