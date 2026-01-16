# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: ifu_stats.py
# -----------------------------------------------------------------------------
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_stats_service
from api.schemas.ifu_stats import IFUStatsResponse
from services.IFUStatsService import IFUStatsService
from settings import ACTIVE_CORPUS_ID

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("", response_model=IFUStatsResponse)
def get_ifu_stats(
    svc: IFUStatsService = Depends(get_stats_service),
    blob_container: str = Query("ifu-docs-test"),
    corpus_id: Optional[str] = Query(default=None),
) -> IFUStatsResponse:
    effective_corpus = corpus_id or ACTIVE_CORPUS_ID
    logger.info(
        "Getting stats for blob_container='%s' corpus_id='%s'",
        blob_container,
        effective_corpus,
    )
    return svc.get_stats(blob_container=blob_container, corpus_id=effective_corpus)


@router.get("/samples")
def get_ifu_samples(
    svc: IFUStatsService = Depends(get_stats_service),
    blob_container: str = Query("ifu-docs-test"),
    corpus_id: Optional[str] = Query(default=None),
    lang: Optional[str] = Query(default=None),
    max_docs: int = Query(10, ge=1, le=50),
    chunks_per_doc: int = Query(3, ge=1, le=10),
):
    effective_corpus = corpus_id or ACTIVE_CORPUS_ID
    logger.info(
        "Getting samples for blob_container='%s' corpus_id='%s' lang=%s max_docs=%d chunks_per_doc=%d",
        blob_container,
        effective_corpus,
        lang,
        max_docs,
        chunks_per_doc,
    )

    return svc.get_indexed_doc_samples(
        blob_container=blob_container,
        corpus_id=effective_corpus,
        lang=(lang.strip().lower() if lang else None),
        max_docs=max_docs,
        chunks_per_doc=chunks_per_doc,
    )

@router.get("/delta")
def get_ifu_delta(
    svc: IFUStatsService = Depends(get_stats_service),
    blob_container: str = Query("ifu-docs-test"),
    corpus_id: Optional[str] = Query(default=None),
):
    effective_corpus = corpus_id or ACTIVE_CORPUS_ID
    return svc.get_storage_index_delta(
        blob_container=blob_container,
        corpus_id=effective_corpus,
    )

