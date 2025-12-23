# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-22
# Description: documents.py
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    container: str

    # Blob-side
    blob_name: str
    size: Optional[int] = None
    content_type: Optional[str] = None
    blob_last_modified: Optional[str] = None
    blob_metadata: Optional[Dict[str, str]] = None

    # Vector-side (Chroma-side)
    chunk_count: Optional[int] = None
    page_count: Optional[int] = None
    document_type: Optional[str] = None
    indexed_last_modified: Optional[str] = None  # or reuse vector doc's last_modified

    # Optional: “is this indexed?”
    is_indexed: bool = Field(False)


class DocumentIndexingInfo(BaseModel):
    # Placeholder for future store/indexing hints (chunk count, last indexed, etc.)
    chunk_count: Optional[int] = None
    last_indexed: Optional[str] = None
    collection_name: Optional[str] = None


class GetDocumentResponse(BaseModel):
    document: DocumentInfo
    indexing: Optional[DocumentIndexingInfo] = None


class ListDocumentsResponse(BaseModel):
    container: str
    count: int
    documents: List[DocumentInfo]


class ListDocumentIdsResponse(BaseModel):
    container: str
    count: int
    doc_ids: List[str]


class IngestDocumentsRequest(BaseModel):
    container: str = Field(..., min_length=1)
    doc_ids: List[str] = Field(..., min_length=1)
    document_type: str = Field("IFU", min_length=1)


class IngestDocumentsResponse(BaseModel):
    container: str
    requested: int
    ingested: int
    document_type: str


class DeleteVectorsResponse(BaseModel):
    doc_id: str
    deleted: int


class UploadDocumentsResponseItem(BaseModel):
    local_path: str
    blob_name: str


class UploadDocumentsResponse(BaseModel):
    container: str
    blob_prefix: str
    uploaded: int
    results: List[UploadDocumentsResponseItem]
