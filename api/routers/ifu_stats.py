# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: ifu_stats.py
# -----------------------------------------------------------------------------
import logging

from fastapi import APIRouter, Depends, Query

from api.schemas.ifu_stats import IFUStatsResponse, DocumentStats, BlobStats
from api.dependencies import get_multi_doc_loader

logger = logging.getLogger("ifu_stats")

router = APIRouter(
    prefix="/ifu",
    tags=["stats"]
)

@router.get("/stats", response_model=IFUStatsResponse)
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

    blob_entries = loader.get_blob_details(blob_container)

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

