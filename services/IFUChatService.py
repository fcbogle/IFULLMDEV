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
        stats_context: Optional[str] = None,
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
            "ask: q_len=%d n_results=%d tone=%s lang=%s where=%s stats_ctx=%s stats_len=%d",
            len(q),
            n_results,
            tone_key,
            language,
            "yes" if where else "no",
            "yes" if stats_context else "no",
            len(stats_context or ""),
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
            stats_context=stats_context,
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
            stats_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build OpenAI-style chat messages using:
          - system prompt (rules + language + tone)
          - optional prior history
          - user prompt (question + operational stats block + retrieved IFU context)

        Notes:
          - stats_context is treated as operational/system information (NOT IFU evidence).
          - retrieved IFU context remains the only source of clinical/IFU facts.
        """
        tone_instruction = self.TONE_PRESETS.get(tone, self.TONE_PRESETS["neutral"])

        lang = (language or "en").strip().lower()
        language_instruction = (
            f"Respond in {lang}. If the user writes in {lang}, keep the same language. "
            f"Do not switch languages unless the user asks."
        )

        # --- System prompt (updated: includes language_instruction + tighter regulatory rules + more conversational style) ---
        system = (
            f"{language_instruction}\n\n"
            "You are an assistant supporting questions about regulated medical device Instructions for Use (IFUs).\n\n"
            "AUTHORITATIVE SOURCES:\n"
            "- Use the provided CONTEXT as the authoritative source of factual IFU information.\n"
            "- Do NOT invent facts or make assumptions.\n"
            "- If the answer is not present in the context, clearly say what information is missing.\n\n"
            "CITATION RULES:\n"
            "- When you use facts from the retrieved IFU context, cite them using [#] indices.\n"
            "- Do NOT cite operational or system statistics as IFU evidence.\n\n"
            "STYLE & CONVERSATION:\n"
            "- Be professional, clear, and approachable.\n"
            "- Use natural language, not legal or robotic phrasing.\n"
            "- Prefer short paragraphs.\n"
            "- Use step-by-step formatting ONLY when the question is procedural.\n"
            "- If the context partially answers the question, answer what you can and ask AT MOST ONE clarifying question.\n"
            "- When appropriate, conclude with ONE short suggested next step.\n\n"
            "REGULATORY SAFETY:\n"
            "- Do not provide medical advice beyond the IFU.\n"
            "- Do not speculate beyond the provided information.\n\n"
            f"TONE GUIDANCE:\n{tone_instruction}"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

        # --- Optional: include prior chat history ---
        for m in (history or []):
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # --- Stats block (updated: clearly marked as OPERATIONAL, not IFU) ---
        stats_block = ""
        if stats_context:
            stats_block = (
                "OPERATIONAL CONTEXT (system statistics — NOT IFU content; do not cite as evidence):\n"
                f"{stats_context}\n\n"
            )

        # --- User prompt (updated: simpler, more natural) ---
        user = (
            f"Question:\n{question}\n\n"
            f"{stats_block}"
            "Retrieved context:\n"
            f"{context if context else '[No relevant IFU content was retrieved]'}\n\n"
            "Please respond according to the system instructions."
        )

        self.logger.debug(
            "build_messages: lang=%s tone=%s history=%d ctx_chars=%d stats_present=%s stats_chars=%d",
            lang,
            tone,
            len(history or []),
            len(context or ""),
            bool(stats_context),
            len(stats_context or ""),
        )

        messages.append({"role": "user", "content": user})
        return messages



