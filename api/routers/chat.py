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
from settings import ACTIVE_CORPUS_ID

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
    """
    if not where or not isinstance(where, dict):
        return None

    container = where.get("container")
    if isinstance(container, str) and container.strip():
        return container.strip()

    blob_container = where.get("blob_container")
    if isinstance(blob_container, str) and blob_container.strip():
        return blob_container.strip()

    metadata = where.get("metadata")
    if isinstance(metadata, dict):
        md_container = metadata.get("container")
        if isinstance(md_container, str) and md_container.strip():
            return md_container.strip()

    return None


def _format_stats_context(stats: Dict[str, Any]) -> str:
    """
    Convert stats dict into a short, prompt-friendly context string.
    Keep it short + deterministic.
    """
    parts: List[str] = ["STATS SUMMARY:"]
    for k in sorted(stats.keys()):
        v = stats.get(k)
        parts.append(f"- {k}: {v}")
    return "\n".join(parts).strip()


def _format_ops_context(
        *,
        stats_dict: Dict[str, Any] | None,
        delta_dict: Dict[str, Any] | None,
        max_list: int = 50,
) -> str:
    """
    Returns a compact, model-friendly operational context block.
    """
    lines: list[str] = []
    lines.append("OPS CONTEXT (storage + index status)")
    lines.append("")

    # 1) Stats
    if stats_dict:
        lines.append("STATS:")
        lines.append(f"- collection_name: {stats_dict.get('collection_name')}")
        lines.append(f"- corpus_id: {stats_dict.get('corpus_id')}")
        lines.append(f"- blob_container: {stats_dict.get('blob_container')}")
        lines.append(f"- total_blobs_in_storage: {stats_dict.get('total_blobs')}")
        lines.append(f"- total_documents_indexed: {stats_dict.get('total_documents')}")
        lines.append(f"- total_chunks_indexed: {stats_dict.get('total_chunks')}")
        lines.append("")
    else:
        lines.append("STATS: [not available]")
        lines.append("")

    # 2) Delta
    if delta_dict:
        lines.append("DELTA (storage vs index):")

        storage_not_indexed = delta_dict.get("storage_not_indexed") or []
        indexed_not_in_storage = delta_dict.get("indexed_not_in_storage") or []

        lines.append(f"- storage_not_indexed_count: {len(storage_not_indexed)}")
        lines.append(f"- indexed_not_in_storage_count: {len(indexed_not_in_storage)}")

        def _clip(xs: Any) -> list[Any]:
            if not isinstance(xs, list):
                return []
            return xs[:max_list]

        if storage_not_indexed:
            lines.append("storage_not_indexed:")
            for x in _clip(storage_not_indexed):
                if isinstance(x, dict):
                    name = x.get("blob_name") or x.get("name") or str(x)
                    lm = x.get("last_modified") or x.get("lastModified")
                    lines.append(f"  - {name}" + (f" (last_modified={lm})" if lm else ""))
                else:
                    lines.append(f"  - {x}")
        else:
            lines.append("storage_not_indexed: []")

        if indexed_not_in_storage:
            lines.append("indexed_not_in_storage:")
            for x in _clip(indexed_not_in_storage):
                if isinstance(x, dict):
                    name = x.get("doc_id") or x.get("name") or str(x)
                    lines.append(f"  - {name}")
                else:
                    lines.append(f"  - {x}")
        else:
            lines.append("indexed_not_in_storage: []")

        extras = {k: v for k, v in delta_dict.items() if k not in {
            "storage_not_indexed", "indexed_not_in_storage"
        }}
        if extras:
            lines.append("")
            lines.append("delta_extra_keys: " + ", ".join(sorted(extras.keys())))

        lines.append("")
    else:
        lines.append("DELTA: [not available]")
        lines.append("")

    return "\n".join(lines).strip()


def _as_dict(obj: Any) -> Dict[str, Any]:
    """
    Best-effort convert models / dict-like objects into a plain dict.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # pydantic v2
    if hasattr(obj, "dict"):
        return obj.dict()  # pydantic v1
    return dict(obj)


@router.post("", response_model=ChatResponse)
def post_chat(
        req: ChatRequest,
        svc: IFUChatService = Depends(get_chat_service),
        stats_svc: IFUStatsService = Depends(get_stats_service),
) -> ChatResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    # priority: where override -> request.container -> default
    container = _extract_container_from_where(req.where) or getattr(req, "container", None) or DEFAULT_BLOB_CONTAINER
    effective_corpus = getattr(req, "corpus_id", None) or ACTIVE_CORPUS_ID
    mode_key = (getattr(req, "mode", None) or "").strip().lower() or None

    want_ops = (mode_key == "ops")

    logger.info(
        "POST /chat (start) q_len=%d n_results=%d container=%s corpus_id=%s mode=%s want_ops=%s",
        len(question),
        req.n_results,
        container,
        effective_corpus,
        mode_key,
        want_ops,
    )

    # -------------------------
    # Best-effort contexts
    # -------------------------
    stats_context: Optional[str] = None
    ops_context: Optional[str] = None

    stats_dict: Optional[Dict[str, Any]] = None
    delta_dict: Optional[Dict[str, Any]] = None

    # 1) Stats context (allow request override)
    if getattr(req, "stats_context", None):
        stats_context = req.stats_context
        logger.info("POST /chat stats_context provided by request (chars=%d)", len(stats_context or ""))
    else:
        try:
            stats = stats_svc.get_stats(blob_container=container, corpus_id=effective_corpus)
            stats_dict = _as_dict(stats)
            stats_context = _format_stats_context(stats_dict)
        except Exception as e:
            logger.warning("POST /chat stats fetch failed: %s", e, exc_info=True)
            stats_context = None
            stats_dict = None

        # -------------------------
        # 2) Delta + ops_context
        # -------------------------
        if want_ops:
            try:
                delta = stats_svc.get_storage_index_delta(
                    blob_container=container,
                    corpus_id=effective_corpus,
                )
                delta_dict = _as_dict(delta)

            except Exception as e:
                logger.warning(
                    "POST /chat delta fetch failed: %s",
                    e,
                    exc_info=True,
                )
                delta_dict = None

            ops_context = _format_ops_context(
                stats_dict=stats_dict,
                delta_dict=delta_dict,
                max_list=50,
            )

        else:
            # Lightweight ops summary even when not in ops mode
            ops_context = _format_ops_context(
                stats_dict=stats_dict,
                delta_dict=None,
                max_list=0,
            )

        # -------------------------
        # DEBUG: confirm contexts built
        # -------------------------
        logger.debug(
            "Built contexts: stats_context=%s ops_context=%s",
            "yes" if stats_context else "no",
            "yes" if ops_context else "no",
        )

        # -------------------------
        # Ask the chat service
        # -------------------------
        try:
            out: Dict[str, Any] = svc.ask(
                container=container,
                corpus_id=effective_corpus,
                mode=mode_key,
                question=question,
                n_results=req.n_results,
                where=req.where,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                history=req.history,
                tone=req.tone,
                language=req.language,
                stats_context=stats_context,
                ops_context=ops_context,
            )

        except Exception as e:
            logger.exception("post_chat failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"chat failed: {e}",
            )

        # -------------------------
        # Build response sources
        # -------------------------
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

        # -------------------------
        # Final logging
        # -------------------------
        logger.info(
            "POST /chat (done) mode=%s answer_len=%d sources=%d",
            out.get("mode"),
            len(out.get("answer", "") or ""),
            len(sources),
        )

        # -------------------------
        # Return response
        # -------------------------
        return ChatResponse(
            question=out["question"],
            answer=out["answer"],
            n_results=out.get("n_results", req.n_results),
            sources=sources,
            mode=out.get("mode"),
            corpus_id=out.get("corpus_id"),
            samples=out.get("samples"),
            model=out.get("model"),
            usage=out.get("usage"),
        )

