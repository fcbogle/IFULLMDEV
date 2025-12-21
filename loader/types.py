# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: types.py
# -----------------------------------------------------------------------------
from dataclasses import dataclass

from pydantic import BaseModel
from typing import List, Optional, TypedDict


class DocumentEntryModel(BaseModel):
    doc_id: str
    chunk_count: int
    page_count: Optional[int] = None
    last_modified: Optional[str] = None
    document_type: Optional[str] = None

class BlobEntryModel(BaseModel):
    blob_name: str

class IFUStatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    total_documents: int  # <- required
    documents: List[DocumentEntryModel]
    blob_container: str
    blobs: List[BlobEntryModel]

@dataclass(frozen=True)
class DocumentEntry:
    doc_id: str
    chunk_count: int
    page_count: Optional[int] = None
    last_modified: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None  # optional if you want to expose


@dataclass(frozen=True)
class BlobEntry(TypedDict):
    blob_name: str
    size: Optional[int]
    content_type: Optional[str]
    last_modified: Optional[str]


class IFUStatsDict(TypedDict):
    collection_name: str
    total_chunks: int
    total_documents: int
    total_blobs: int
    documents: List[DocumentEntry]
    blob_container: str
    blobs: List[BlobEntry]