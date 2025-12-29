# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-28
# Description: chat.py
# -----------------------------------------------------------------------------
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_chat_service
from api.schemas.chat import ChatRequest, ChatResponse, ChatSource
from services.IFUChatService import IFUChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
def post_chat(
        req: ChatRequest,
        svc: IFUChatService = Depends(get_chat_service),
) -> ChatResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    logger.info("POST /chat (start) question_len=%d n_results=%d", len(question), req.n_results)

    try:
        out: Dict[str, Any] = svc.ask(
            question=question,
            n_results=req.n_results,
            where=req.where,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            history=req.history,
        )
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

    logger.info("POST /chat (done) answer_len=%d sources=%d", len(out.get("answer", "") or ""), len(sources))

    return ChatResponse(
        question=out["question"],
        answer=out["answer"],
        n_results=req.n_results,
        sources=sources,
        model=out.get("model"),
        usage=out.get("usage"),
    )