# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-28
# Description: chat.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_chat_service, get_stats_service
from api.schemas.chat import ChatResponse, ChatRequest, ChatSource
from services.IFUChatService import IFUChatService
from services.IFUStatsService import IFUStatsService

DEFAULT_BLOB_CONTAINER = "default-container"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

def _extract_container_from_where(where: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Extract a blob container from the `where` filter if present.

    Supports patterns like:
      where = {"container": "ifu-docs"}
      where = {"blob_container": "ifu-docs"}
      where = {"metadata": {"container": "ifu-docs"}}

    Returns None if not found.
    """
    if not where or not isinstance(where, dict):
        return None

    # Direct keys
    container = where.get("container")
    if isinstance(container, str) and container.strip():
        return container.strip()

    blob_container = where.get("blob_container")
    if isinstance(blob_container, str) and blob_container.strip():
        return blob_container.strip()

    # Nested metadata
    metadata = where.get("metadata")
    if isinstance(metadata, dict):
        md_container = metadata.get("container")
        if isinstance(md_container, str) and md_container.strip():
            return md_container.strip()

    return None


def _format_stats_context(stats: Dict[str, Any]) -> str:
    """
    Convert stats dict into a short, prompt-friendly context string.
    Adjust to match your stats model fields.
    """
    # Keep it short, stable, and deterministic (important for eval + caching).
    parts: List[str] = ["STATS SUMMARY:"]
    for k in sorted(stats.keys()):
        v = stats.get(k)
        parts.append(f"- {k}: {v}")
    return "\n".join(parts).strip()


# ---------------------------
# Routes
# ---------------------------

@router.post("", response_model=ChatResponse)
def post_chat(
    req: ChatRequest,
    svc: IFUChatService = Depends(get_chat_service),
    stats_svc: IFUStatsService = Depends(get_stats_service),
) -> ChatResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    logger.info(
        "POST /chat (start) question_len=%d n_results=%d",
        len(question),
        req.n_results,
    )

    # Determine which container to ask stats for:
    # priority: where override -> request.container -> default
    container = _extract_container_from_where(req.where) or getattr(req, "container", None) or DEFAULT_BLOB_CONTAINER

    # Best-effort stats context: never fail chat if stats fails
    stats_context: Optional[str] = None
    stats_status = "disabled"

    # Optional: allow request to provide its own stats_context (manual override)
    if getattr(req, "stats_context", None):
        stats_context = req.stats_context
        stats_status = "provided"
    else:
        try:
            stats = stats_svc.get_stats(blob_container=container)  # ensure kwarg name matches your service
            stats_dict = stats.model_dump() if hasattr(stats, "model_dump") else dict(stats)
            stats_context = _format_stats_context(stats_dict)
            stats_status = "ok" if stats_context else "empty"
        except Exception as e:
            logger.warning(
                "POST /chat stats fetch failed (container='%s'): %s",
                container,
                e,
                exc_info=True,
            )
            stats_context = None
            stats_status = "error"

    logger.info(
        "POST /chat stats: container='%s' stats_status=%s stats_context_chars=%d",
        container,
        stats_status,
        len(stats_context or ""),
    )

    try:
        out: Dict[str, Any] = svc.ask(
            question=question,
            n_results=req.n_results,
            where=req.where,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            history=req.history,
            tone=req.tone,
            language=req.language,
            stats_context=stats_context,
        )
        logger.info("POST /chat req: tone=%s language=%s", req.tone, req.language)
    except Exception as e:
        logger.exception("post_chat failed: %s", e)
        raise HTTPException(status_code=500, detail=f"chat failed: {e}")

    # Map sources
    raw_sources: List[Dict[str, Any]] = out.get("sources", []) or []
    sources = [
        ChatSource(
            doc_id=s.get("doc_id"),
            chunk_id=s.get("chunk_id"),
            page=s.get("page"),
            score=s.get("score"),
            metadata=s.get("metadata"),
            text=s.get("text"),
        )
        for s in raw_sources
    ]

    logger.info(
        "POST /chat (done) answer_len=%d sources=%d",
        len(out.get("answer", "") or ""),
        len(sources),
    )

    return ChatResponse(
        question=out["question"],
        answer=out["answer"],
        n_results=req.n_results,
        sources=sources,
        model=out.get("model"),
        usage=out.get("usage"),
    )