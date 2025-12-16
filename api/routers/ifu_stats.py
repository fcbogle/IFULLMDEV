# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: ifu_stats.py
# -----------------------------------------------------------------------------
import logging

from fastapi import APIRouter, Depends, Query

from api.schemas.ifu_stats import IFUStatsResponse, DocumentStats, BlobStats
from api.dependencies import get_multi_doc_loader

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/stats",
    tags=["stats"]
)

@router.get("", response_model=IFUStatsResponse)
def get_ifu_stats(
    loader = Depends(get_multi_doc_loader),
    blob_container: str = Query("ifu-docs-test", description="Blob container to inspect"),
) -> IFUStatsResponse:

    stats = loader.get_stats(blob_container=blob_container)

    documents = [
        DocumentStats(
            doc_id=d["doc_id"],
            chunk_count=d["chunk_count"],
            page_count=d.get("page_count"),
            last_modified=d.get("last_modified"),
            document_type=d.get("document_type"),
        )
        for d in stats["documents"]
    ]

    try:
        blob_entries = loader.get_blob_details(blob_container)
        logger.info("Blob entries: %r", blob_entries)
    except Exception as e:
        logger.exception("Failed to get blob details: %s", e)
        raise

    blobs = [
        BlobStats(
            blob_name=b.get("blob_name") or b.get("name"),
            size=b.get("size"),
            content_type=b.get("content_type"),
            last_modified=b.get("last_modified"),
        )
        for b in blob_entries
    ]

    return IFUStatsResponse(
        collection_name=stats["collection_name"],
        total_chunks=stats["total_chunks"],
        total_documents=len(documents),
        documents=documents,
        blob_container=blob_container,
        total_blobs=len(blobs),
        blobs=blobs,
    )

