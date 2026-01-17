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


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # pydantic v2
    if hasattr(obj, "dict"):
        return obj.dict()  # pydantic v1
    return dict(obj)


def _format_stats_context(stats: Dict[str, Any]) -> str:
    parts: List[str] = ["STATS SUMMARY:"]
    for k in sorted(stats.keys()):
        parts.append(f"- {k}: {stats.get(k)}")
    return "\n".join(parts).strip()


def _format_ops_context(
    *,
    stats_dict: Dict[str, Any] | None,
    delta_dict: Dict[str, Any] | None,
    max_list: int = 50,
    include_blob_list: bool = True,
    include_delta_lists: bool = True,
) -> str:
    """
    Compact, model-friendly operational context block.
    Uses ONLY system facts (storage + index status).
    """
    lines: list[str] = ["OPS CONTEXT (storage + index status)", ""]

    # 1) Stats
    if stats_dict:
        lines += [
            "STATS:",
            f"- collection_name: {stats_dict.get('collection_name')}",
            f"- corpus_id: {stats_dict.get('corpus_id')}",
            f"- blob_container: {stats_dict.get('blob_container')}",
            f"- total_blobs_in_storage: {stats_dict.get('total_blobs')}",
            f"- total_documents_indexed: {stats_dict.get('total_documents')}",
            f"- total_chunks_indexed: {stats_dict.get('total_chunks')}",
            "",
        ]
    else:
        lines += ["STATS: [not available]", ""]

    # 2) Blob list (optional)
    if include_blob_list:
        blobs = stats_dict.get("blobs") if stats_dict else None
        if isinstance(blobs, list) and blobs:
            lines.append("STORAGE BLOBS:")
            shown = 0
            for b in blobs[:max_list]:
                if not isinstance(b, dict):
                    continue
                name = b.get("blob_name") or b.get("name") or "UNKNOWN_BLOB"
                lm = b.get("last_modified")
                size = b.get("size")

                suffix = []
                if size is not None:
                    suffix.append(f"size={size}")
                if lm:
                    suffix.append(f"last_modified={lm}")
                meta = f" ({', '.join(suffix)})" if suffix else ""
                lines.append(f"- {name}{meta}")
                shown += 1

            omitted = max(0, len(blobs) - min(len(blobs), max_list))
            if omitted:
                lines.append(f"... {omitted} more blobs omitted")
            if shown == 0:
                lines.append("- [no blob dicts available]")
            lines.append("")
        else:
            lines += ["STORAGE BLOBS: [not available]", ""]
    else:
        lines += ["STORAGE BLOBS: [omitted]", ""]

    # 3) Delta
    if not delta_dict:
        lines += ["DELTA: [not available]", ""]
        return "\n".join(lines).strip()

    # Support BOTH naming schemes
    blobs_not_indexed = (
        delta_dict.get("blobs_not_indexed")
        if isinstance(delta_dict.get("blobs_not_indexed"), list)
        else delta_dict.get("storage_not_indexed")
        if isinstance(delta_dict.get("storage_not_indexed"), list)
        else []
    )

    indexed_not_in_storage = (
        delta_dict.get("indexed_not_in_storage")
        if isinstance(delta_dict.get("indexed_not_in_storage"), list)
        else []
    )

    indexed_in_storage = (
        delta_dict.get("indexed_in_storage")
        if isinstance(delta_dict.get("indexed_in_storage"), list)
        else []
    )

    # Prefer explicit counts if present
    blobs_not_indexed_count = (
        delta_dict.get("blobs_not_indexed_count")
        if delta_dict.get("blobs_not_indexed_count") is not None
        else delta_dict.get("storage_not_indexed_count")
        if delta_dict.get("storage_not_indexed_count") is not None
        else len(blobs_not_indexed)
    )

    indexed_not_in_storage_count = (
        delta_dict.get("indexed_not_in_storage_count")
        if delta_dict.get("indexed_not_in_storage_count") is not None
        else len(indexed_not_in_storage)
    )

    indexed_in_storage_count = (
        delta_dict.get("indexed_in_storage_count")
        if delta_dict.get("indexed_in_storage_count") is not None
        else len(indexed_in_storage)
    )

    lines += [
        "DELTA (storage vs index):",
        f"- blobs_not_indexed_count: {blobs_not_indexed_count}",
        f"- indexed_not_in_storage_count: {indexed_not_in_storage_count}",
        f"- indexed_in_storage_count: {indexed_in_storage_count}",
        "",
    ]

    if not include_delta_lists:
        lines += ["DELTA LISTS: [omitted]", ""]
        return "\n".join(lines).strip()

    def _emit_list(title: str, items: list[Any]) -> None:
        lines.append(f"{title}:")
        if not items:
            lines.append("- [none]")
            lines.append("")
            return
        for x in items[:max_list]:
            lines.append(f"- {x}")
        omitted = max(0, len(items) - min(len(items), max_list))
        if omitted:
            lines.append(f"... {omitted} more omitted")
        lines.append("")

    _emit_list("BLOBS NOT INDEXED", blobs_not_indexed)
    _emit_list("INDEXED NOT IN STORAGE", indexed_not_in_storage)
    _emit_list("INDEXED IN STORAGE", indexed_in_storage)

    return "\n".join(lines).strip()


def _resolve_mode_for_router(
    *,
    svc: IFUChatService,
    question: str,
    mode_key: Optional[str],
) -> str:
    """
    Router-side mode resolution so the router fetches the right context
    BEFORE calling svc.ask().

    Priority:
      - explicit mode in request
      - auto ops
      - auto inventory
      - default qa
    """
    if mode_key in ("ops", "inventory", "qa"):
        return mode_key

    # Use service heuristics (best), but guard if private or missing.
    q = (question or "").strip()

    auto_ops = False
    auto_inventory = False

    is_ops_fn = getattr(svc, "_is_ops_question", None)
    if callable(is_ops_fn):
        try:
            auto_ops = bool(is_ops_fn(q))
        except Exception:
            auto_ops = False

    is_inv_fn = getattr(svc, "_is_inventory_question", None)
    if callable(is_inv_fn):
        try:
            auto_inventory = bool(is_inv_fn(q))
        except Exception:
            auto_inventory = False

    if auto_ops:
        return "ops"
    if auto_inventory:
        return "inventory"
    return "qa"


@router.post("", response_model=ChatResponse)
def post_chat(
    req: ChatRequest,
    svc: IFUChatService = Depends(get_chat_service),
    stats_svc: IFUStatsService = Depends(get_stats_service),
) -> ChatResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    container = _extract_container_from_where(req.where) or getattr(req, "container", None) or DEFAULT_BLOB_CONTAINER
    effective_corpus = getattr(req, "corpus_id", None) or ACTIVE_CORPUS_ID
    mode_key = (getattr(req, "mode", None) or "").strip().lower() or None

    # IMPORTANT: resolve mode here so we fetch correct contexts for auto-routed ops/inventory
    resolved_mode = _resolve_mode_for_router(svc=svc, question=question, mode_key=mode_key)
    want_ops = (resolved_mode == "ops")

    logger.info(
        "POST /chat (start) q_len=%d n_results=%d container=%s corpus_id=%s mode_key=%s resolved_mode=%s want_ops=%s",
        len(question),
        req.n_results,
        container,
        effective_corpus,
        mode_key,
        resolved_mode,
        want_ops,
    )

    stats_context: Optional[str] = None
    ops_context: Optional[str] = None
    stats_dict: Optional[Dict[str, Any]] = None
    delta_dict: Optional[Dict[str, Any]] = None

    # 1) Stats context (best-effort, never fail chat)
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

    # 2) Delta + ops_context
    if want_ops:
        try:
            delta = stats_svc.get_storage_index_delta(blob_container=container, corpus_id=effective_corpus)
            delta_dict = _as_dict(delta)
        except Exception as e:
            logger.warning("POST /chat delta fetch failed: %s", e, exc_info=True)
            delta_dict = None

        ops_context = _format_ops_context(
            stats_dict=stats_dict,
            delta_dict=delta_dict,
            max_list=50,
            include_blob_list=True,
            include_delta_lists=True,
        )
    else:
        # Lightweight (counts only) summary can still help QA/inventory explain indexed vs stored
        ops_context = _format_ops_context(
            stats_dict=stats_dict,
            delta_dict=None,
            max_list=0,
            include_blob_list=False,
            include_delta_lists=False,
        )

    logger.debug(
        "Built contexts: stats_context=%s ops_context=%s stats_blobs=%s delta=%s",
        "yes" if stats_context else "no",
        "yes" if ops_context else "no",
        "yes" if (stats_dict and stats_dict.get("blobs")) else "no",
        "yes" if delta_dict else "no",
    )
    logger.debug("ops_context preview (first 1200 chars):\n%s", (ops_context or "")[:1200])

    # 3) Ask chat service (ALWAYS)
    try:
        out: Dict[str, Any] = svc.ask(
            container=container,
            corpus_id=effective_corpus,
            mode=mode_key,  # keep original; svc still does its own resolution
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
        raise HTTPException(status_code=500, detail=f"chat failed: {e}")

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
        "POST /chat (done) out_mode=%s answer_len=%d sources=%d",
        out.get("mode"),
        len(out.get("answer", "") or ""),
        len(sources),
    )

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


