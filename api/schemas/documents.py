# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-22
# Description: documents.py
# -----------------------------------------------------------------------------
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, ConfigDict, Field


# Core models
class DocumentItem(BaseModel):
    """
    Represents a single blob/document in storage.
    Matches your IFUDocumentLoader.get_blob_details() output,
    plus enforced facade fields (doc_id + container).
    """
    model_config = ConfigDict(extra="allow")  # allow additional metadata fields over time

    doc_id: str = Field(..., min_length=1)
    container: str = Field(..., min_length=1)

    blob_name: str = Field(..., min_length=1)
    size: Optional[int] = Field(None, ge=0)
    content_type: Optional[str] = None
    last_modified: Optional[datetime] = None  # accept ISO string; serialize as ISO

class DocumentIndexingInfo(BaseModel):
    """
    Optional placeholder for future: could include collection name,
    chunk count, last indexed time, etc.
    Keep flexible to avoid tight coupling until your store exposes it.
    """
    model_config = ConfigDict(extra="allow")

    indexed: Optional[bool] = None
    collection: Optional[str] = None
    chunk_count: Optional[int] = Field(None, ge=0)
    last_indexed: Optional[datetime] = None

# List and Get Response models
class ListDocumentsResponse(BaseModel):
    container: str
    count: int
    documents: List[DocumentItem]


class GetDocumentResponse(BaseModel):
    document: DocumentItem
    indexing: Optional[DocumentIndexingInfo] = None


class ListDocumentIdsResponse(BaseModel):
    container: str
    count: int
    doc_ids: List[str]

# Upload models
class UploadDocumentsResponseItem(BaseModel):
    """
    For your upload_documents() which returns Dict[Path, str].
    The dict is awkward in OpenAPI, so we model as a list of items.
    """
    local_path: str
    blob_name: str


class UploadDocumentsResponse(BaseModel):
    container: str
    blob_prefix: str = ""
    count: int
    uploads: List[UploadDocumentsResponseItem]

# Index Reindex and Delete models
class IngestDocumentsRequest(BaseModel):
    doc_ids: List[str] = Field(..., min_length=1)
    document_type: str = Field("IFU", min_length=1)


class IngestDocumentsResponse(BaseModel):
    container: str
    requested: int
    ingested: int
    document_type: str


class ReindexDocumentResponse(BaseModel):
    container: str
    doc_id: str
    document_type: str
    ingested: int


class DeleteDocumentVectorsResponse(BaseModel):
    doc_id: str
    deleted: int

# Bytes downloaded models
class GetDocumentBytesResponse(BaseModel):
    """
    Only if you decide to return metadata about the download.
    Usually you'd return StreamingResponse from the router instead.
    """
    container: str
    doc_id: str
    size_bytes: int
    content_type: Optional[str] = None