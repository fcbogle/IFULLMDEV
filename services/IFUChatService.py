# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-28
# Description: services/IFUChatService.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utility.logging_utils import get_class_logger
from services.IFUQueryService import IFUQueryService
from chat.OpenAIChat import OpenAIChat, Message


@dataclass
class IFUChatService:
    """
    RAG chat service:
      - retrieve context via IFUQueryService
      - construct a grounded prompt
      - call OpenAIChat
      - return answer + sources
    """

    query_service: IFUQueryService
    chat_client: OpenAIChat
    logger: Any = None

    # prompt knobs
    max_context_chars: int = 12000  # keeps prompts sane without tokenizers
    include_scores_in_prompt: bool = True

    def __post_init__(self) -> None:
        self.logger = self.logger or get_class_logger(self.__class__)
        self.logger.info(
            "IFUChatService initialised (query_service=%s, chat_client=%s)",
            type(self.query_service).__name__,
            type(self.chat_client).__name__,
        )

    def ask(
        self,
        *,
        question: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        q = (question or "").strip()
        if not q:
            raise ValueError("question must not be empty")

        self.logger.info(
            "ask: question_len=%d n_results=%d where=%s temp=%.2f max_tokens=%d (start)",
            len(q),
            n_results,
            "yes" if where else "no",
            temperature,
            max_tokens,
        )

        # 1) Retrieve
        try:
            raw = self.query_service.query(query_text=q, n_results=n_results, where=where)
            hits = self.query_service.to_hits(
                raw,
                include_text=True,
                include_scores=True,
                include_metadata=True,
            )
            self.logger.info("ask: retrieved_hits=%d (done)", len(hits))
        except Exception as e:
            self.logger.error("ask: retrieval failed: %s", e, exc_info=True)
            raise

        # 2) Build context text
        context_text = self._build_context(hits)

        # 3) Compose messages
        messages = self._build_messages(
            question=q,
            context=context_text,
            history=history,
        )

        # 4) Call OpenAI
        try:
            resp = self.chat_client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            model = getattr(resp, "model", None)

            self.logger.info("ask: answer_len=%d model=%s usage=%r (done)", len(answer), model, usage)
        except Exception as e:
            self.logger.error("ask: LLM call failed: %s", e, exc_info=True)
            raise

        return {
            "question": q,
            "answer": answer,
            "sources": hits,  # return raw hits; router can map to schema
            "model": model,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else (usage if isinstance(usage, dict) else None),
        }

    def _build_context(self, hits: List[Dict[str, Any]]) -> str:
        """
        Build a compact context pack. Keep it deterministic and easy to cite:
        [1] doc_id=... page=... chunk_id=... score=...
            <text>
        """
        parts: List[str] = []
        used = 0

        for idx, h in enumerate(hits, start=1):
            doc_id = h.get("doc_id")
            page = h.get("page")
            chunk_id = h.get("chunk_id")
            score = h.get("score")
            text = (h.get("text") or "").strip()

            header = f"[{idx}] doc_id={doc_id} page={page} chunk_id={chunk_id}"
            if self.include_scores_in_prompt:
                header += f" score={score}"

            block = header + "\n" + text
            if not text:
                continue

            # crude char limit (works well enough without token counting)
            if used + len(block) > self.max_context_chars:
                self.logger.info("build_context: hit_limit reached at %d hits (chars=%d)", idx - 1, used)
                break

            parts.append(block)
            used += len(block) + 2

        context = "\n\n".join(parts).strip()
        self.logger.info("build_context: context_chars=%d hits_used=%d", len(context), len(parts))
        return context

    def _build_messages(
        self,
        *,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Message]:
        system = (
            "You are an assistant helping with medical device IFU content. "
            "Answer ONLY using the provided context. "
            "If the context does not contain the answer, say you don't have enough information and ask what to check. "
            "When you use facts from context, cite them using [#] indices. "
            "Formatting: default to 1–3 sentences. Avoid block formatting unless the user asks. "
            "If an address appears in context, present it inline in a sentence."
        )

        messages: List[Message] = [{"role": "system", "content": system}]

        # Optional chat history (v2)
        if history:
            for m in history:
                role = (m.get("role") or "").strip()
                content = (m.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        user = (
            "Please answer the question below using only the information provided in the context. "
            "Write in clear, natural prose (not bullet points or copied blocks). "
            "If the answer involves structured information (e.g. steps, warnings, contact details), "
            "integrate it smoothly into sentences. "
            "If the context does not contain enough information, say so briefly and suggest what to check.\n\n"
            f"Question: {question}\n\n"
            f"Context (for reference only, do not quote verbatim):\n{context if context else '[no context retrieved]'}\n\n"
            "Answer:"
        )
        messages.append({"role": "user", "content": user})

        self.logger.debug("build_messages: messages=%d context_present=%s", len(messages), bool(context))
        return messages


