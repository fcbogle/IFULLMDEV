from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class BlobInfo(BaseModel):
    container: str
    blob_name: str
    size: Optional[int] = None
    content_type: Optional[str] = None
    last_modified: Optional[str] = None
    blob_metadata: Dict[str, str] = Field(default_factory=dict)

    status_code: Literal["ready", "needs_metadata", "not_ingestible"] = "needs_metadata"
    issues: List[str] = Field(default_factory=list)
    status: str = ""


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
    ok: bool

    # present on success
    bytes: Optional[int] = None
    content_type: Optional[str] = None

    # present on reject/error
    error: Optional[str] = None
    rejected: Optional[str] = None
    message: Optional[str] = None
    status: Optional[str] = None

    class Config:
        extra = "allow"


class UploadBlobsResponse(BaseModel):
    container: str
    blob_prefix: str = ""
    uploaded: int
    errors: int = 0
    results: List[UploadBlobsResponseItem]

    # optional if you decide to track it explicitly
    rejected: Optional[int] = None

    class Config:
        extra = "allow"