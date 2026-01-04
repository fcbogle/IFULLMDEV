# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-23
# Description: documents.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response

from api.dependencies import get_document_service
from api.schemas.documents import (
    DeleteVectorsResponse,
    DocumentInfo,
    GetDocumentResponse,
    IngestDocumentsRequest,
    IngestDocumentsResponse,
    ListDocumentIdsResponse,
    ListDocumentsResponse,
)
from services.IFUDocumentService import IFUDocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# -----------------------------------------------------------------------------
# In-memory ingest job store (single-process only)
# -----------------------------------------------------------------------------
INGEST_JOBS: Dict[str, Dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_ingest_job(
    *,
    job_id: str,
    container: str,
    doc_ids: List[str],
    document_type: str,
    svc: IFUDocumentService,
) -> None:
    """
    Background runner. Ingests doc_ids one-by-one so we can report progress.

    NOTE:
      - In-memory job store only works reliably with a single uvicorn worker.
      - If you run multiple workers, move INGEST_JOBS to Redis/DB.
    """
    job = INGEST_JOBS.get(job_id)
    if not job:
        logger.warning("ingest job missing job_id='%s' (possibly restarted process)", job_id)
        return

    job["status"] = "running"
    job["started_at"] = _utc_now()
    job["updated_at"] = _utc_now()

    for doc_id in doc_ids:
        job["items"][doc_id] = {"status": "running", "updated_at": _utc_now()}

        try:
            # Use existing synchronous service (kept unchanged)
            svc.ingest_documents(container=container, doc_ids=[doc_id], document_type=document_type)

            job["items"][doc_id] = {"status": "done", "updated_at": _utc_now()}
            job["done"] += 1

        except Exception as e:
            job["items"][doc_id] = {
                "status": "failed",
                "error": str(e),
                "updated_at": _utc_now(),
            }
            job["failed"] += 1
            logger.exception("ingest job failed job_id='%s' doc_id='%s'", job_id, doc_id)

        job["updated_at"] = _utc_now()

    job["status"] = "done_with_errors" if job["failed"] else "done"
    job["finished_at"] = _utc_now()
    job["updated_at"] = _utc_now()


# -----------------------------------------------------------------------------
# Existing endpoints
# -----------------------------------------------------------------------------
@router.get("", response_model=ListDocumentsResponse)
def get_documents(
    container: str,
    svc: IFUDocumentService = Depends(get_document_service),
) -> ListDocumentsResponse:
    container = (container or "").strip()
    logger.info("GET /documents (start) container='%s'", container)
    if not container:
        logger.warning("GET /documents -> 400 (container empty)")
        raise HTTPException(status_code=400, detail="container must not be empty")
    try:
        raw: List[Dict[str, Any]] = svc.list_documents(container=container)
        docs = [DocumentInfo(**d) for d in raw]
        resp = ListDocumentsResponse(container=container, count=len(docs), documents=docs)
        logger.info("GET /documents (done) container='%s' count=%d", container, resp.count)
        return resp
    except Exception as e:
        logger.exception("GET /documents -> 500 container='%s': %s", container, e)
        raise HTTPException(status_code=500, detail=f"get_documents failed: {e}")


@router.get("/ids", response_model=ListDocumentIdsResponse)
def get_document_ids(
    container: str,
    svc: IFUDocumentService = Depends(get_document_service),
) -> ListDocumentIdsResponse:
    container = (container or "").strip()
    logger.info("GET /documents/ids (start) container='%s'", container)

    if not container:
        logger.warning("GET /documents/ids -> 400 (container empty)")
        raise HTTPException(status_code=400, detail="container must not be empty")

    try:
        ids = svc.list_document_ids(container=container)
        resp = ListDocumentIdsResponse(container=container, count=len(ids), doc_ids=ids)
        logger.info("GET /documents/ids (done) container='%s' count=%d", container, resp.count)
        return resp
    except Exception as e:
        logger.exception("GET /documents/ids -> 500 container='%s': %s", container, e)
        raise HTTPException(status_code=500, detail=f"get_document_ids failed: {e}")


@router.get("/{doc_id}", response_model=GetDocumentResponse)
def get_document(
    doc_id: str,
    container: str,
    svc: IFUDocumentService = Depends(get_document_service),
) -> GetDocumentResponse:
    doc_id = (doc_id or "").strip()
    container = (container or "").strip()
    logger.info("GET /documents/{doc_id} (start) container='%s' doc_id='%s'", container, doc_id)

    if not doc_id:
        logger.warning("GET /documents/{doc_id} -> 400 (doc_id empty)")
        raise HTTPException(status_code=400, detail="doc_id must not be empty")
    if not container:
        logger.warning("GET /documents/{doc_id} -> 400 (container empty) doc_id='%s'", doc_id)
        raise HTTPException(status_code=400, detail="container must not be empty")

    try:
        raw: Dict[str, Any] = svc.get_document(container=container, doc_id=doc_id)
        resp = GetDocumentResponse(
            document=DocumentInfo(**raw["document"]),
            indexing=None,
        )
        logger.info("GET /documents/{doc_id} (done) container='%s' doc_id='%s'", container, doc_id)
        return resp
    except KeyError as e:
        logger.warning("GET /documents/{doc_id} -> 404 container='%s' doc_id='%s': %s", container, doc_id, e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("GET /documents/{doc_id} -> 500 container='%s' doc_id='%s': %s", container, doc_id, e)
        raise HTTPException(status_code=500, detail=f"get_document failed: {e}")


@router.head("/{doc_id}")
def head_document(
    doc_id: str,
    container: str,
    svc: IFUDocumentService = Depends(get_document_service),
) -> Response:
    doc_id = (doc_id or "").strip()
    container = (container or "").strip()
    logger.info("HEAD /documents/{doc_id} (start) container='%s' doc_id='%s'", container, doc_id)

    if not doc_id or not container:
        logger.warning(
            "HEAD /documents/{doc_id} -> 400 (missing params) container='%s' doc_id='%s'",
            container,
            doc_id,
        )
        raise HTTPException(status_code=400, detail="doc_id and container are required")

    try:
        exists = svc.document_exists(container=container, doc_id=doc_id)
        if not exists:
            logger.info("HEAD /documents/{doc_id} -> 404 container='%s' doc_id='%s'", container, doc_id)
            raise HTTPException(status_code=404, detail="not found")

        logger.info("HEAD /documents/{doc_id} -> 200 container='%s' doc_id='%s'", container, doc_id)
        return Response(status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("HEAD /documents/{doc_id} -> 500 container='%s' doc_id='%s': %s", container, doc_id, e)
        raise HTTPException(status_code=500, detail=f"head_document failed: {e}")


# -----------------------------------------------------------------------------
# NEW: Start ingest as background job (bulk)
# -----------------------------------------------------------------------------
@router.post("/ingest", status_code=202)
def post_ingest_documents(
    req: IngestDocumentsRequest,
    background: BackgroundTasks,
    svc: IFUDocumentService = Depends(get_document_service),
) -> Dict[str, Any]:
    container = (req.container or "").strip()
    document_type = (req.document_type or "IFU").strip() or "IFU"
    doc_ids = [x.strip() for x in (req.doc_ids or []) if isinstance(x, str) and x.strip()]

    logger.info(
        "POST /documents/ingest (queue) container='%s' document_type='%s' requested=%d",
        container,
        document_type,
        len(doc_ids),
    )

    if not container:
        raise HTTPException(status_code=400, detail="container must not be empty")
    if not doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids must not be empty")

    job_id = str(uuid4())
    INGEST_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "container": container,
        "document_type": document_type,
        "requested": len(doc_ids),
        "done": 0,
        "failed": 0,
        "items": {},  # doc_id -> {status, error?, updated_at}
        "created_at": _utc_now(),
        "started_at": None,
        "finished_at": None,
        "updated_at": _utc_now(),
    }

    background.add_task(
        _run_ingest_job,
        job_id=job_id,
        container=container,
        doc_ids=doc_ids,
        document_type=document_type,
        svc=svc,
    )

    return {"job_id": job_id, "status": "queued", "requested": len(doc_ids)}


# -----------------------------------------------------------------------------
# NEW: Start ingest as background job (single doc)
# -----------------------------------------------------------------------------
@router.post("/{doc_id}/ingest", status_code=202)
def post_ingest_document(
    doc_id: str,
    container: str,
    background: BackgroundTasks,
    document_type: str = "IFU",
    svc: IFUDocumentService = Depends(get_document_service),
) -> Dict[str, Any]:
    doc_id = (doc_id or "").strip()
    container = (container or "").strip()
    document_type = (document_type or "IFU").strip() or "IFU"

    logger.info(
        "POST /documents/{doc_id}/ingest (queue) container='%s' doc_id='%s' document_type='%s'",
        container,
        doc_id,
        document_type,
    )

    if not container:
        raise HTTPException(status_code=400, detail="container must not be empty")
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id must not be empty")

    # Optional: validate that the blob exists before queueing
    try:
        if not svc.document_exists(container=container, doc_id=doc_id):
            raise HTTPException(status_code=404, detail="document not found in blob store")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("POST /documents/{doc_id}/ingest -> 500 validating existence: %s", e)
        raise HTTPException(status_code=500, detail=f"existence check failed: {e}")

    job_id = str(uuid4())
    INGEST_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "container": container,
        "document_type": document_type,
        "requested": 1,
        "done": 0,
        "failed": 0,
        "items": {},  # doc_id -> {status, error?, updated_at}
        "created_at": _utc_now(),
        "started_at": None,
        "finished_at": None,
        "updated_at": _utc_now(),
    }

    background.add_task(
        _run_ingest_job,
        job_id=job_id,
        container=container,
        doc_ids=[doc_id],
        document_type=document_type,
        svc=svc,
    )

    return {"job_id": job_id, "status": "queued", "requested": 1}


# -----------------------------------------------------------------------------
# NEW: Poll ingest job status
# -----------------------------------------------------------------------------
@router.get("/ingest/jobs/{job_id}")
def get_ingest_job(job_id: str) -> Dict[str, Any]:
    job_id = (job_id or "").strip()
    job = INGEST_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


# -----------------------------------------------------------------------------
# Existing endpoints (still synchronous)
# -----------------------------------------------------------------------------
@router.post("/{doc_id}/reindex", response_model=IngestDocumentsResponse)
def post_reindex_document(
    doc_id: str,
    container: str,
    document_type: str = "IFU",
    svc: IFUDocumentService = Depends(get_document_service),
) -> IngestDocumentsResponse:
    doc_id = (doc_id or "").strip()
    container = (container or "").strip()
    logger.info(
        "POST /documents/{doc_id}/reindex (start) container='%s' doc_id='%s' document_type='%s'",
        container,
        doc_id,
        document_type,
    )

    if not doc_id or not container:
        raise HTTPException(status_code=400, detail="doc_id and container are required")

    try:
        ingested = svc.reindex_document(
            container=container,
            doc_id=doc_id,
            document_type=document_type,
        )
        resp = IngestDocumentsResponse(
            container=container,
            requested=1,
            ingested=ingested,
            document_type=document_type,
        )
        logger.info(
            "POST /documents/{doc_id}/reindex (done) container='%s' doc_id='%s' ingested=%d",
            container,
            doc_id,
            resp.ingested,
        )
        return resp
    except Exception as e:
        logger.exception(
            "POST /documents/{doc_id}/reindex -> 500 container='%s' doc_id='%s': %s",
            container,
            doc_id,
            e,
        )
        raise HTTPException(status_code=500, detail=f"reindex failed: {e}")


@router.delete("/{doc_id}/vectors", response_model=DeleteVectorsResponse)
def delete_document_vectors(
    doc_id: str,
    svc: IFUDocumentService = Depends(get_document_service),
) -> DeleteVectorsResponse:
    doc_id = (doc_id or "").strip()
    logger.info("DELETE /documents/{doc_id}/vectors (start) doc_id='%s'", doc_id)

    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id must not be empty")

    try:
        deleted = svc.delete_document_vectors(doc_id=doc_id)
        resp = DeleteVectorsResponse(doc_id=doc_id, deleted=deleted)
        logger.info("DELETE /documents/{doc_id}/vectors (done) doc_id='%s' deleted=%d", doc_id, deleted)
        return resp
    except Exception as e:
        logger.exception("DELETE /documents/{doc_id}/vectors -> 500 doc_id='%s': %s", doc_id, e)
        raise HTTPException(status_code=500, detail=f"delete vectors failed: {e}")


