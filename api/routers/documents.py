# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-23
# Description: documents.py
# -----------------------------------------------------------------------------
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Response

from api.dependencies import get_document_service

from api.schemas.documents import (
    DocumentInfo,
    GetDocumentResponse,
    ListDocumentsResponse,
    ListDocumentIdsResponse,
    IngestDocumentsRequest,
    IngestDocumentsResponse,
    DeleteVectorsResponse,
)

from services.IFUDocumentService import IFUDocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

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
        logger.warning("HEAD /documents/{doc_id} -> 400 (missing params) container='%s' doc_id='%s'", container, doc_id)
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


@router.post("/ingest", response_model=IngestDocumentsResponse)
def post_ingest_documents(
    req: IngestDocumentsRequest,
    svc: IFUDocumentService = Depends(get_document_service),
) -> IngestDocumentsResponse:
    container = (req.container or "").strip()
    logger.info(
        "POST /documents/ingest (start) container='%s' document_type='%s' requested=%d",
        container,
        req.document_type,
        len(req.doc_ids or []),
    )

    if not container:
        logger.warning("POST /documents/ingest -> 400 (container empty)")
        raise HTTPException(status_code=400, detail="container must not be empty")
    if not req.doc_ids:
        logger.warning("POST /documents/ingest -> 400 (doc_ids empty) container='%s'", container)
        raise HTTPException(status_code=400, detail="doc_ids must not be empty")

    try:
        ingested = svc.ingest_documents(
            container=container,
            doc_ids=req.doc_ids,
            document_type=req.document_type,
        )
        resp = IngestDocumentsResponse(
            container=container,
            requested=len(req.doc_ids),
            ingested=ingested,
            document_type=req.document_type,
        )
        logger.info(
            "POST /documents/ingest (done) container='%s' document_type='%s' requested=%d ingested=%d",
            container,
            req.document_type,
            resp.requested,
            resp.ingested,
        )
        return resp
    except Exception as e:
        logger.exception("POST /documents/ingest -> 500 container='%s': %s", container, e)
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}")

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
        logger.warning(
            "POST /documents/{doc_id}/reindex -> 400 (missing params) container='%s' doc_id='%s'",
            container,
            doc_id,
        )
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
        logger.warning("DELETE /documents/{doc_id}/vectors -> 400 (doc_id empty)")
        raise HTTPException(status_code=400, detail="doc_id must not be empty")

    try:
        deleted = svc.delete_document_vectors(doc_id=doc_id)
        resp = DeleteVectorsResponse(doc_id=doc_id, deleted=deleted)
        logger.info("DELETE /documents/{doc_id}/vectors (done) doc_id='%s' deleted=%d", doc_id, deleted)
        return resp
    except Exception as e:
        logger.exception("DELETE /documents/{doc_id}/vectors -> 500 doc_id='%s': %s", doc_id, e)
        raise HTTPException(status_code=500, detail=f"delete vectors failed: {e}")

