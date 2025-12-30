from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BlobInfo(BaseModel):
    container: str
    blob_name: str
    size: Optional[int] = None
    content_type: Optional[str] = None
    last_modified: Optional[str] = None
    blob_metadata: Dict[str, str] = Field(default_factory=dict)


class ListBlobsResponse(BaseModel):
    container: str
    prefix: str = ""
    count: int
    blobs: List[BlobInfo]


class GetBlobResponse(BaseModel):
    blob: BlobInfo


class DeleteBlobResponse(BaseModel):
    container: str
    blob_name: str
    deleted: bool


class SetBlobMetadataRequest(BaseModel):
    container: str = Field(..., min_length=1)
    blob_name: str = Field(..., min_length=1)
    # overwrite semantics (Azure). If you want merge, we can add merge=true later.
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SetBlobMetadataResponse(BaseModel):
    container: str
    blob_name: str
    blob_metadata: Dict[str, str]


class UploadBlobsResponseItem(BaseModel):
    filename: str
    blob_name: str


class UploadBlobsResponse(BaseModel):
    container: str
    uploaded: int
    results: List[UploadBlobsResponseItem]
