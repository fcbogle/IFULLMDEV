# services/IFUChatService.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

from utility.logging_utils import get_class_logger
from services.IFUQueryService import IFUQueryService
from chat.OpenAIChat import OpenAIChat


@dataclass
class IFUChatService:
    TONE_PRESETS: ClassVar[Dict[str, str]] = {
        "neutral": "Answer clearly and accurately.",
        "plain": (
            "Answer in plain English. Avoid jargon. Explain acronyms. "
            "Assume the reader is not technical."
        ),
        "technical": "Answer as a technical expert. Use precise terminology and structure.",
        "regulatory": (
            "Answer in a regulatory and clinical tone. Be precise, cautious, and factual. "
            "Do not speculate. Reference standards where relevant."
        ),
        "training": "Answer as a training instructor. Use step-by-step guidance and clear instructions.",
    }

    query_service: IFUQueryService
    chat_client: OpenAIChat
    logger: Any = None

    max_context_chars: int = 12000
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
        tone: str = "neutral",
        language: str = "en",
    ) -> Dict[str, Any]:
        q = (question or "").strip()
        if not q:
            raise ValueError("question must not be empty")

        tone_key = (tone or "neutral").strip().lower()
        if tone_key not in self.TONE_PRESETS:
            self.logger.warning("ask: unknown tone='%s' -> defaulting to neutral", tone_key)
            tone_key = "neutral"

        lang = (language or "en").strip().lower()

        self.logger.info(
            "ask: question_len=%d n_results=%d where=%s temp=%.2f max_tokens=%d tone=%s lang=%s (start)",
            len(q),
            n_results,
            "yes" if where else "no",
            float(temperature),
            int(max_tokens),
            tone_key,
            lang,
        )

        # 1) Retrieve
        raw = self.query_service.query(query_text=q, n_results=n_results, where=where)
        hits = self.query_service.to_hits(raw, include_text=True, include_scores=True, include_metadata=True)
        self.logger.info("ask: retrieved_hits=%d", len(hits))

        # 2) Build context
        context_text = self._build_context(hits)

        # 3) Compose messages (tone in system prompt)
        messages = self._build_messages(
            question=q,
            context=context_text,
            history=history,
            tone=tone_key,
            language=lang,
        )

        # 4) Call LLM
        resp = self.chat_client.chat(messages=messages, temperature=float(temperature), max_tokens=int(max_tokens))
        answer = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        model = getattr(resp, "model", None)

        return {
            "question": q,
            "answer": answer,
            "sources": hits,
            "tone": tone_key,
            "language": lang,
            "model": model,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else (usage if isinstance(usage, dict) else None),
        }

    def _build_context(self, hits: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        used = 0

        for idx, h in enumerate(hits, start=1):
            text = (h.get("text") or "").strip()
            if not text:
                continue

            header = f"[{idx}] doc_id={h.get('doc_id')} page={h.get('page')} chunk_id={h.get('chunk_id')}"
            if self.include_scores_in_prompt:
                header += f" score={h.get('score')}"

            block = header + "\n" + text
            if used + len(block) > self.max_context_chars:
                break

            parts.append(block)
            used += len(block) + 2

        return "\n\n".join(parts).strip()

    def _build_messages(
        self,
        *,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
        tone: str = "neutral",
        language: str = "en",
    ) -> List[Dict[str, str]]:
        tone_instruction = self.TONE_PRESETS.get(tone, self.TONE_PRESETS["neutral"])

        lang = (language or "en").strip().lower()
        language_instruction = (
            f"Respond in {lang}. If the user writes in {lang}, keep the same language. "
            f"Do not switch languages unless the user asks."
        )

        system = (
            "You are an assistant helping with medical device IFU content.\n"
            "Use the provided CONTEXT as the authoritative source.\n"
            "Do not invent facts. If the answer is not in the context, say so.\n"
            "When you use facts from context, cite them using [#] indices.\n"
            "Be helpful and conversational.\n"
            "If needed, ask at most ONE clarifying question.\n"
            "Formatting: prefer short paragraphs; use steps only if the user asks or it is clearly procedural.\n\n"
            f"TONE:\n{tone_instruction}"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

        for m in (history or []):
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        user = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context if context else '[no context retrieved]'}\n\n"
            "Answer:"
        )
        messages.append({"role": "user", "content": user})
        return messages


