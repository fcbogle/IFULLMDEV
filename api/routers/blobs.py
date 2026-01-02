# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-29
# Description: api/routers/blobs.py
# -----------------------------------------------------------------------------
import logging
from typing import List, Dict, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import Response

from api.dependencies import get_blob_service
from api.schemas.blobs import (
    BlobInfo,
    ListBlobsResponse,
    GetBlobResponse,
    UploadBlobsResponse,
    SetBlobMetadataResponse, SetBlobMetadataRequest, DeleteBlobResponse,
)
from services.IFUBlobService import IFUBlobService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/blobs", tags=["blobs"])

@router.get("", response_model=ListBlobsResponse)
def get_blobs(container: str, prefix: str = "", svc: IFUBlobService = Depends(get_blob_service)) -> ListBlobsResponse:
    container = (container or "").strip()
    prefix = (prefix or "").strip()
    if not container:
        raise HTTPException(status_code=400, detail="container must not be empty")

    logger.info("GET /blobs (start) container='%s' prefix='%s'", container, prefix)

    try:
        raw = svc.get_blob_details(container=container, prefix=prefix)

        blobs: List[BlobInfo] = []
        for b in raw:
            enriched = {**b, "container": container, **_blob_status(b)}
            blobs.append(BlobInfo(**enriched))

        logger.info("GET /blobs (done) container='%s' count=%d", container, len(blobs))
        return ListBlobsResponse(container=container, prefix=prefix, count=len(blobs), blobs=blobs)

    except Exception as e:
        logger.exception("GET /blobs failed: %s", e)
        raise HTTPException(status_code=500, detail=f"get_blobs failed: {e}")



@router.get("/{blob_name}", response_model=GetBlobResponse)
def get_blob(
    blob_name: str,
    container: str,
    svc: IFUBlobService = Depends(get_blob_service),
) -> GetBlobResponse:
    blob_name = (blob_name or "").strip()
    container = (container or "").strip()
    if not blob_name or not container:
        raise HTTPException(status_code=400, detail="blob_name and container are required")

    logger.info("GET /blobs/%s (start) container='%s'", blob_name, container)
    try:
        raw = svc.get_blob(container=container, blob_name=blob_name)
        logger.info("GET /blobs/%s (done) container='%s'", blob_name, container)
        return GetBlobResponse(container=container, blob=BlobInfo(**raw))
    except Exception as e:
        logger.exception("GET /blobs/%s failed: %s", blob_name, e)
        raise HTTPException(status_code=500, detail=f"get_blob failed: {e}")


@router.head("/{blob_name}")
def head_blob(
    blob_name: str,
    container: str,
    svc: IFUBlobService = Depends(get_blob_service),
) -> Response:
    blob_name = (blob_name or "").strip()
    container = (container or "").strip()
    if not blob_name or not container:
        raise HTTPException(status_code=400, detail="blob_name and container are required")

    try:
        exists = svc.blob_exists(container=container, blob_name=blob_name)
        if not exists:
            raise HTTPException(status_code=404, detail="not found")
        return Response(status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("HEAD /blobs/%s failed: %s", blob_name, e)
        raise HTTPException(status_code=500, detail=f"head_blob failed: {e}")


@router.get("/{blob_name}/download")
def download_blob(
    blob_name: str,
    container: str,
    svc: IFUBlobService = Depends(get_blob_service),
) -> Response:
    blob_name = (blob_name or "").strip()
    container = (container or "").strip()
    if not blob_name or not container:
        raise HTTPException(status_code=400, detail="blob_name and container are required")

    try:
        data = svc.download_blob_bytes(container=container, blob_name=blob_name)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{blob_name}"'},
        )
    except Exception as e:
        logger.exception("download_blob failed: %s", e)
        raise HTTPException(status_code=500, detail=f"download_blob failed: {e}")


@router.post("/upload", response_model=UploadBlobsResponse)
async def upload_blobs(
    container: str,
    blob_prefix: str = "",
    files: List[UploadFile] = File(...),
    svc: IFUBlobService = Depends(get_blob_service),
):
    container = (container or "").strip()
    blob_prefix = (blob_prefix or "").strip()

    if not container:
        raise HTTPException(status_code=400, detail="container must not be empty")
    if not files:
        raise HTTPException(status_code=400, detail="files must not be empty")

    logger.info("POST /blobs/upload (start) container='%s' prefix='%s' files=%d",
                container, blob_prefix, len(files))

    try:
        out = await svc.upload_files(container=container, blob_prefix=blob_prefix, files=files)
        logger.info("POST /blobs/upload (done) uploaded=%d", out.get("uploaded", 0))
        return out
    except Exception as e:
        logger.exception("upload_blobs failed: %s", e)
        raise HTTPException(status_code=500, detail=f"upload failed: {e}")

@router.post("/metadata", response_model=SetBlobMetadataResponse)
def set_blob_metadata(
    req: SetBlobMetadataRequest,
    svc: IFUBlobService = Depends(get_blob_service),
) -> SetBlobMetadataResponse:
    container = (req.container or "").strip()
    blob_name = (req.blob_name or "").strip()
    if not container or not blob_name:
        raise HTTPException(status_code=400, detail="container and blob_name are required")

    logger.info("POST /blobs/metadata (start) container='%s' blob='%s'", container, blob_name)
    try:
        out = svc.set_blob_metadata(container=container, blob_name=blob_name, metadata=req.metadata)
        logger.info("POST /blobs/metadata (done) container='%s' blob='%s' keys=%d", container, blob_name, len(out))
        return SetBlobMetadataResponse(container=container, blob_name=blob_name, blob_metadata=out)
    except Exception as e:
        logger.exception("POST /blobs/metadata failed: %s", e)
        raise HTTPException(status_code=500, detail=f"set metadata failed: {e}")


@router.delete("/{blob_name}", response_model=DeleteBlobResponse)
def delete_blob(
    blob_name: str,
    container: str,
    svc: IFUBlobService = Depends(get_blob_service),
) -> DeleteBlobResponse:
    blob_name = (blob_name or "").strip()
    container = (container or "").strip()
    if not blob_name or not container:
        raise HTTPException(status_code=400, detail="container and blob_name are required")

    logger.info("DELETE /blobs/{blob} (start) container='%s' blob='%s'", container, blob_name)
    try:
        deleted = svc.delete_blob(container=container, blob_name=blob_name)
        logger.info("DELETE /blobs/{blob} (done) container='%s' blob='%s' deleted=%s", container, blob_name, deleted)
        return DeleteBlobResponse(container=container, blob_name=blob_name, deleted=deleted)
    except Exception as e:
        logger.exception("DELETE /blobs/{blob} failed: %s", e)
        raise HTTPException(status_code=500, detail=f"delete blob failed: {e}")

StatusCode = Literal["ready", "not_ingestible", "warning"]

REQUIRED_META_KEYS = {"source", "filename"}  # blob-level requirements only
def _blob_status(b: Dict[str, Any]) -> Dict[str, Any]:
    name = (b.get("blob_name") or "").strip()
    content_type = (b.get("content_type") or "").strip().lower()
    meta = b.get("blob_metadata") or {}
    issues: List[str] = []

    # --- PDF check (strict) ---
    is_pdf_ext = name.lower().endswith(".pdf")
    is_pdf_ct = content_type in ("application/pdf", "application/x-pdf")
    is_pdf = is_pdf_ext or is_pdf_ct

    if not is_pdf:
        issues.append("not_pdf")

    # --- required blob metadata ---
    missing = [
        k for k in REQUIRED_META_KEYS
        if not str(meta.get(k, "")).strip()
    ]
    if missing:
        issues.append("missing_metadata:" + ",".join(missing))

    # --- derive status_code ---
    if "not_pdf" in issues:
        status_code: StatusCode = "not_ingestible"
    elif any(i.startswith("missing_metadata:") for i in issues):
        status_code = "needs_metadata"
    else:
        status_code = "ready"

    status_label = {
        "ready": "🟢 READY",
        "needs_metadata": "🟠 NEEDS METADATA",
        "not_ingestible": "🔴 NOT SUPPORTED",
    }[status_code]

    return {
        "status_code": status_code,
        "issues": issues,
        "status": status_label,
    }


