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

from settings import ACTIVE_CORPUS_ID, VECTOR_COLLECTION_DEFAULT, BLOB_CONTAINER_DEFAULT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("", response_model=IFUStatsResponse)
def get_ifu_stats(
    svc: IFUStatsService = Depends(get_stats_service),
    blob_container: str = Query(default=BLOB_CONTAINER_DEFAULT),
    corpus_id: str = Query(default=ACTIVE_CORPUS_ID),
    vector_collection: str = Query(default=VECTOR_COLLECTION_DEFAULT),
):
    effective_corpus = (corpus_id or ACTIVE_CORPUS_ID).strip()

    return svc.get_stats(
        vector_collection=vector_collection,
        corpus_id=effective_corpus,
        blob_container=blob_container,  # optional enrichment
    )


@router.get("/samples")
def get_ifu_samples(
    svc: IFUStatsService = Depends(get_stats_service),
    vector_collection: str = Query(default=VECTOR_COLLECTION_DEFAULT),
    corpus_id: Optional[str] = Query(default=None),
    lang: Optional[str] = Query(default=None),
    max_docs: int = Query(10, ge=1, le=50),
    chunks_per_doc: int = Query(3, ge=1, le=10),
):
    effective_corpus = (corpus_id or ACTIVE_CORPUS_ID).strip()
    lang_norm = (lang.strip().lower() if lang else None)

    logger.info(
        "Getting samples for vector_collection='%s' corpus_id='%s' lang=%s max_docs=%d chunks_per_doc=%d",
        vector_collection,
        effective_corpus,
        lang_norm,
        max_docs,
        chunks_per_doc,
    )

    return svc.get_indexed_doc_samples(
        vector_collection=vector_collection,
        corpus_id=effective_corpus,
        lang=lang_norm,
        max_docs=max_docs,
        chunks_per_doc=chunks_per_doc,
    )


@router.get("/delta")
def get_ifu_delta(
    svc: IFUStatsService = Depends(get_stats_service),
    vector_collection: str = Query(default=VECTOR_COLLECTION_DEFAULT),
    blob_container: str = Query(default=BLOB_CONTAINER_DEFAULT),
    corpus_id: Optional[str] = Query(default=None),
):
    effective_corpus = (corpus_id or ACTIVE_CORPUS_ID).strip()

    return svc.get_storage_index_delta(
        vector_collection=vector_collection,
        blob_container=blob_container,
        corpus_id=effective_corpus,
    )

