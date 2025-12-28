# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-27
# Description: IFUChatService.py
# -----------------------------------------------------------------------------
import logging
from typing import Any, Optional, Dict, List, Sequence

from OpenAIChat import OpenAIChat, Message
from services.IFUQueryService import IFUQueryService
from utility.logging_utils import get_class_logger


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


class IFUChatService:
    """
    Chat Service:
        - retrieves relevant chunks using IFUQueryService
        - builds an instruction + context prompt
        - calls OpenAIChat to generate answer
        - returns answer + sources
    """
    query_service: IFUQueryService
    chat_client: OpenAIChat
    logger: logging.Logger | None = None

    system_prompt: str = (
        "You are an assistant helping with medical device IFU documents.\n"
        "Use ONLY the provided context to answer.\n"
        "If the context is insufficient, say so and ask a precise follow-up.\n"
        "Cite sources as [doc_id pX] where possible.\n"
    )

    max_context_chars: int = 18_000  # guardrail (fits most models comfortably)
    default_temperature: float = 0.0
    default_max_tokens: int = 700

    def __post_init__(self) -> None:
        self.logger = self.logger or get_class_logger(self.__class__)
        self.logger.info(
            "IFUChatService initialised (query_service=%s chat_client=%s)",
            type(self.query_service).__name__,
            type(self.chat_client).__name__,
        )

    def chat(self,
             *,
             user_query: str,
             n_results: int = 5,
             where: Optional[Dict[str: Any]] = None,
             conversation: Optional[List[Message]] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             ) -> Dict[str, Any]:
        """
            Returns:
            {
                "answer": str,
                "sources": [ {doc_id, page, chunk_id, score, text, metadata}, ... ],
                "retrieval": {"n_results":..., "where":...},
                "model": str|None,
                "usage": Any|None,
            }
        """
        q = (user_query or "").strip()
        if not q:
            raise ValueError("user_query must not be empty")

        conversation = conversation or []

        self.logger.info(
            "chat: query='%s' n_results=%d where=%s conv_turns=%d (start)",
            q[:120],
            n_results,
            where,
            len(conversation),
        )

        # Retrieve raw results from IFUQueryService
        raw = self.query_service.query(query_text=q, n_results=n_results, where=where)
        hits = self.query_service.to_hits(raw, include_text=True, include_scores=True, include_metadata=True)

        self.logger.info("chat: retrieved hits=%d", len(hits))

        # Build context block
        context = self._build_context_block(hits)
        self.logger.debug("chat: context_chars=%d", len(context))

        # Build messages
        messages: List[Message] = [{"role": "system", "content": self.system_prompt}]
        # Optional: include prior turns (if you want multi-turn chat in UI)
        # Keep them BEFORE the new user question, and DO NOT include retrieved context as assistant text.
        if conversation:
            messages.extend(conversation)

        user_payload = (
            f"USER QUESTION:\n{q}\n\n"
            f"CONTEXT (retrieved passages):\n{context}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Answer the user question.\n"
            f"- If you use a passage, cite it like [doc_id pX].\n"
            f"- If you cannot answer from context, say so.\n"
        )
        messages.append({"role": "user", "content": user_payload})

        # Call OpenAIChat LLM to generate answer
        temp = self.default_temperature if temperature is None else temperature
        mtok = self.default_max_tokens if max_tokens is None else max_tokens

        resp = self.chat_client.chat(messages, temperature=temp, max_tokens=mtok)

        try:
            answer = resp.choices[0].message.content or ""
        except Exception as e:
            self.logger.error("chat: unexpected OpenAI response: %s", e, exc_info=True)
            raise RuntimeError(f"Unexpected OpenAI response format: {e}")

        self.logger.info("chat: answer_chars=%d (done)", len(answer))

        return {
            "answer": answer,
            "sources": hits,  # already dicts from to_hits()
            "retrieval": {"n_results": n_results, "where": where},
            "model": getattr(resp, "model", None),
            "usage": getattr(resp, "usage", None),
        }

    def _build_context_block(self, hits: Sequence[Dict[str, Any]]) -> str:
        """
        Turn hits into a prompt-friendly context block.
        """
        parts: List[str] = []
        total = 0

        for i, h in enumerate(hits, start=1):
            doc_id = _safe_str(h.get("doc_id") or h.get("metadata", {}).get("doc_id"))
            page = h.get("page") or h.get("metadata", {}).get("page_start")
            text = _safe_str(h.get("text"))
            score = h.get("score")

            # Keep it compact but traceable
            header = f"[{i}] {doc_id}"
            if page is not None:
                header += f" p{page}"
            if score is not None:
                header += f" score={score:.4f}" if isinstance(score, (int, float)) else f" score={score}"

            chunk = f"{header}\n{text}".strip() + "\n"

            if total + len(chunk) > self.max_context_chars:
                self.logger.warning(
                    "_build_context_block: truncating context at %d chars (limit=%d)",
                    total,
                    self.max_context_chars,
                )
                break

            parts.append(chunk)
            total += len(chunk)

        if not parts:
            return "(no retrieved context)"

        return "\n---\n".join(parts)


