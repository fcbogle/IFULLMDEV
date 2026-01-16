# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-21
# Description: IFUStatsService.py
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

from services.IFUStatsService import IFUStatsService
from utility.logging_utils import get_class_logger
from services.IFUQueryService import IFUQueryService
from chat.OpenAIChat import OpenAIChat

from settings import ACTIVE_CORPUS_ID


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
    stats_service: IFUStatsService

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

    def _is_inventory_question(self, q: str) -> bool:
        ql = (q or "").strip().lower()
        triggers = [
            # listing / awareness
            "what documents", "which documents", "list documents", "list the documents",
            "indexed documents", "ingested documents", "what is indexed", "what's indexed",
            "what is in the corpus", "what's in the corpus", "what does the document cover",

            # per-document overview / focus
            "each document", "every document",
            "each indexed document", "each ingested document",
            "focuses on", "focus of each", "what each document covers",
            "overview of each", "summary of each", "summarise each", "summarize each",
            "document overview", "document summary",

            # size / scale of corpus
            "corpus size", "how big is the corpus", "total chunks", "total documents",
            "how many chunks", "how many documents", "indexed corpus", "indexed size",

            # sampling wording
            "sample chunks", "samples", "excerpts",
        ]

        return any(t in ql for t in triggers)

    def _add_d_labels_to_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Adds stable D# labels (D1, D2, ...) to the samples list.
        Only used for inventory mode responses.
        """
        out: List[Dict[str, Any]] = []
        for i, s in enumerate(samples or [], start=1):
            d_label = f"D{i}"
            s2 = dict(s)
            s2["d_label"] = d_label

            # Optional: also label the chunks so UI can group easily
            chunks = s2.get("sample_chunks") or []
            if isinstance(chunks, list):
                labeled_chunks = []
                for c in chunks:
                    c2 = dict(c) if isinstance(c, dict) else {"text": str(c)}
                    c2["d_label"] = d_label
                    labeled_chunks.append(c2)
                s2["sample_chunks"] = labeled_chunks

            out.append(s2)
        return out

    def _is_ops_question(self, q: str) -> bool:
        ql = (q or "").strip().lower()
        triggers = [
            # operational terms
            "blob", "blobs", "storage",
            "indexed", "ingested",
            "vector", "vectors", "chroma", "collection",
            "corpus", "delta",
            "not indexed", "missing", "difference",
            "how many blobs", "how many documents", "size of the corpus",
            "last indexed", "when was it indexed", "recent indexing",
        ]
        return any(t in ql for t in triggers)

    def ask(
            self,
            *,
            container: str = "ifu-docs-test",
            corpus_id: Optional[str] = None,
            mode: Optional[str] = None,
            question: str,
            n_results: int = 5,
            where: Optional[Dict[str, Any]] = None,
            temperature: float = 0.0,
            max_tokens: int = 512,
            history: Optional[List[Dict[str, str]]] = None,
            tone: str = "neutral",
            language: str = "en",
            stats_context: Optional[str] = None,
            ops_context: Optional[str] = None
    ) -> Dict[str, Any]:
        q = (question or "").strip()
        if not q:
            raise ValueError("question must not be empty")

        tone_key = (tone or "neutral").strip().lower()
        if tone_key not in self.TONE_PRESETS:
            self.logger.warning("ask: unknown tone='%s' -> defaulting to neutral", tone_key)
            tone_key = "neutral"

        lang = (language or "en").strip().lower()
        mode_key = (mode or "").strip().lower() or None

        blob_container = (container or "ifu-docs-test").strip()
        effective_corpus = (corpus_id or ACTIVE_CORPUS_ID).strip()

        self.logger.info(
            "ask: mode=%s resolved? corpus=%s container=%s q_len=%d n_results=%d tone=%s lang=%s where=%s stats_ctx=%s ops_ctx=%s",
            mode_key,
            effective_corpus,
            blob_container,
            len(q),
            n_results,
            tone_key,
            lang,
            "yes" if where else "no",
            "yes" if stats_context else "no",
            "yes" if ops_context else "no",
        )

        # ----------------------------
        # 0) Decide mode: ops vs inventory vs qa
        # ----------------------------
        mode_key = (mode or "").strip().lower() or None
        force_inventory = mode_key == "inventory"
        force_qa = mode_key == "qa"
        force_ops = mode_key == "ops"

        # auto-detect only if not forced
        auto_inventory = False if (force_inventory or force_qa or force_ops) else self._is_inventory_question(q)
        auto_ops = False if (force_inventory or force_qa or force_ops) else self._is_ops_question(q)

        if force_ops:
            resolved_mode = "ops"
        elif force_inventory:
            resolved_mode = "inventory"
        elif force_qa:
            resolved_mode = "qa"
        else:
            # priority: ops > inventory > qa
            resolved_mode = "ops" if auto_ops else ("inventory" if auto_inventory else "qa")

        self.logger.info(
            "ask: mode_resolved=%s (mode_key=%r force_ops=%s force_inventory=%s force_qa=%s auto_ops=%s auto_inventory=%s)",
            resolved_mode,
            mode_key,
            force_ops,
            force_inventory,
            force_qa,
            auto_ops,
            auto_inventory,
        )

        # ----------------------------
        # INVENTORY PATH
        # ----------------------------
        if resolved_mode == "inventory":
            # inventory language filter should be optional
            lang_filter = (lang or "").strip().lower() or None

            samples = self.stats_service.get_indexed_doc_samples(
                blob_container=blob_container,
                corpus_id=effective_corpus,
                lang=lang_filter,
                max_docs=25,
                chunks_per_doc=2,
            )

            samples = self._add_d_labels_to_samples(samples)

            messages = self._build_inventory_messages(
                question=q,
                samples=samples,
                history=history,
                tone=tone_key,
                language=lang,
            )

            if not messages:
                raise ValueError("messages must be non-empty")

            resp = self.chat_client.chat(
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            answer = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            model = getattr(resp, "model", None)

            return {
                "mode": "inventory",
                "corpus_id": effective_corpus,
                "question": q,
                "answer": answer,
                "n_results": n_results,
                "sources": [],
                "samples": samples,  # remove if your response schema disallows extra fields
                "tone": tone_key,
                "language": lang,
                "model": model,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else (
                    usage if isinstance(usage, dict) else None
                ),
            }

        # ----------------------------
        # OPS PATH (operational/system awareness)
        # ----------------------------
        if resolved_mode == "ops":
            messages = self._build_ops_messages(
                question=q,
                ops_context=ops_context or "[No operational context provided]",
                history=history,
                tone=tone_key,
                language=lang,
            )

            resp = self.chat_client.chat(
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            answer = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            model = getattr(resp, "model", None)

            return {
                "mode": "ops",
                "corpus_id": effective_corpus,
                "question": q,
                "answer": answer,
                "n_results": n_results,
                "sources": [],
                "tone": tone_key,
                "language": lang,
                "model": model,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else (
                    usage if isinstance(usage, dict) else None
                ),
            }

        # ----------------------------
        # QA PATH
        # ----------------------------

        # Merge caller where with enforced scoping
        merged_filters: Dict[str, Any] = dict(where or {})
        merged_filters.setdefault("corpus_id", effective_corpus)
        merged_filters.setdefault("container", blob_container)

        # Optional: enforce retrieval language only if caller didn't specify one
        # (prevents surprising behaviour if they pass {"lang": "de"} explicitly)
        if "lang" not in merged_filters and lang:
            merged_filters["lang"] = lang

        if len(merged_filters) == 1:
            retrieval_where = merged_filters
        elif "$and" in merged_filters or "$or" in merged_filters:
            retrieval_where = merged_filters
        else:
            retrieval_where = {"$and": [{k: v} for k, v in merged_filters.items()]}

        raw = self.query_service.query(query_text=q, n_results=n_results, where=retrieval_where)
        hits = self.query_service.to_hits(raw, include_text=True, include_scores=True, include_metadata=True)
        self.logger.info("ask: retrieved_hits=%d", len(hits))

        context_text = self._build_context(hits)

        messages = self._build_messages(
            question=q,
            context=context_text,
            history=history,
            tone=tone_key,
            language=lang,
            stats_context=stats_context,
            # (optional) pass retrieved_sources/allowed_docs if you wire them
        )

        resp = self.chat_client.chat(
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        answer = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        model = getattr(resp, "model", None)

        return {
            "mode": "qa",
            "corpus_id": effective_corpus,
            "question": q,
            "answer": answer,
            "n_results": n_results,
            "sources": hits,
            "tone": tone_key,
            "language": lang,
            "model": model,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else (
                usage if isinstance(usage, dict) else None
            ),
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
            retrieved_sources: Optional[List[Dict[str, Any]]] = None,
            allowed_docs: Optional[List[str]] = None,
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
        # --- Language instruction ---
        language_instruction = (
            f"Respond in {lang}. If the user writes in {lang}, keep the same language. "
            f"Do not switch languages unless the user asks."
        )

        # --- OPTIONAL (recommended): build an explicit "RETRIEVED SOURCES" list + allowed docs ---
        # Pass these in from your retrieval layer if you have them.
        # retrieved_sources example item:
        #   {"i": 1, "doc": "938327PK1 Iss2.pdf", "chunk_id": "c12", "page": 7, "section": "Warnings", "score": 0.18}
        retrieved_sources = retrieved_sources or []  # ensure defined
        allowed_docs = allowed_docs or []  # ensure defined

        # If you have retrieved_sources, derive allowed_docs from them
        if retrieved_sources and not allowed_docs:
            allowed_docs = sorted({s.get("doc") for s in retrieved_sources if s.get("doc")})

        allowed_docs_block = ""
        if allowed_docs:
            allowed_docs_block = (
                    "ALLOWED DOCUMENTS (you may ONLY mention/cite these documents):\n"
                    + "\n".join(f"- {d}" for d in allowed_docs)
                    + "\n\n"
            )

        sources_text = ""
        if retrieved_sources:
            lines = ["RETRIEVED SOURCES (use these indices for citations like [#]):"]
            for s in retrieved_sources:
                i = s.get("i")
                doc = s.get("doc", "Unknown document")
                page = s.get("page")
                section = s.get("section")
                chunk_id = s.get("chunk_id")
                score = s.get("score")

                extras = []
                if page is not None:
                    extras.append(f"p.{page}")
                if section:
                    extras.append(f"{section}")
                if chunk_id:
                    extras.append(f"chunk:{chunk_id}")
                if score is not None:
                    extras.append(f"score:{score}")

                suffix = f" ({', '.join(extras)})" if extras else ""
                lines.append(f"[{i}] {doc}{suffix}")

            sources_text = "\n".join(lines) + "\n\n"
        else:
            sources_text = "RETRIEVED SOURCES:\n[No sources available]\n\n"

        # --- System prompt (UPDATED: strict grounding + doc control + citation gating) ---
        system = (
            f"{language_instruction}\n\n"
            "You are an assistant supporting questions about regulated medical device Instructions for Use (IFUs).\n\n"
            "GROUNDING (MOST IMPORTANT):\n"
            "- Use ONLY the provided RETRIEVED IFU CONTEXT for factual claims about the IFU.\n"
            "- If RETRIEVED IFU CONTEXT is empty or does not contain the answer, say: "
            "\"I can't find that in the currently retrieved IFU content.\" Then state what is missing.\n"
            "- Do NOT use prior chat history as a factual source.\n"
            "- Do NOT infer or guess.\n\n"
            "DOCUMENT CONTROL:\n"
            "- You may ONLY mention or cite documents that appear in the RETRIEVED SOURCES list / ALLOWED DOCUMENTS.\n"
            "- If a document is not in that list, do not reference it.\n\n"
            "CITATIONS:\n"
            "- Every factual statement must be supported by the retrieved context and MUST include a citation like [#].\n"
            "- If you cannot provide a citation [#] for a factual claim, do not include that claim.\n"
            "- Never cite OPERATIONAL CONTEXT as IFU evidence.\n\n"
            "RETRIEVAL QUALITY CHECK:\n"
            "- If the retrieved context is only loosely related, say so and ask AT MOST ONE clarifying question.\n"
            "- If the user asks for a value (limits, warnings, contraindications), quote exact wording when available.\n\n"
            "REGULATORY SAFETY:\n"
            "- Do not provide medical advice beyond the IFU.\n"
            "- Do not speculate beyond the provided IFU context.\n\n"
            "STYLE & CONVERSATION:\n"
            "- Be professional, clear, and approachable.\n"
            "- Prefer short paragraphs.\n"
            "- Use step-by-step formatting ONLY when the question is procedural.\n"
            "- If the context partially answers the question, answer what you can and ask AT MOST ONE clarifying question.\n"
            "- When appropriate, conclude with ONE short suggested next step.\n\n"
            f"TONE GUIDANCE:\n{tone_instruction}"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

        # --- Optional: include prior chat history ---
        # NOTE: history can still help conversational continuity, but system prompt forbids using it as factual evidence.
        for m in (history or []):
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # --- Stats block (UPDATED: even stronger separation; placed AFTER retrieved context) ---
        stats_block = ""
        if stats_context:
            stats_block = (
                "\n\nOPERATIONAL CONTEXT (system statistics — NOT IFU content):\n"
                "- Use ONLY to explain system status (e.g., what is indexed).\n"
                "- NEVER use as evidence for IFU facts.\n"
                f"{stats_context}\n"
            )

        # --- User prompt (UPDATED: sources + allowed docs + context-first ordering) ---
        user = (
            f"Question:\n{question}\n\n"
            f"{allowed_docs_block}"
            "RETRIEVED IFU CONTEXT:\n"
            f"{context if context else '[No relevant IFU content was retrieved]'}\n\n"
            f"{sources_text}"
            f"{stats_block}\n\n"
            "Answer using ONLY the RETRIEVED IFU CONTEXT. "
            "Cite sources as [#] for every factual claim. "
            "If the answer is not present, say you cannot find it in the retrieved IFU content."
        )

        messages.append({"role": "user", "content": user})
        return messages

    def _build_inventory_messages(
            self,
            *,
            question: str,
            samples: List[Dict[str, Any]],
            history: Optional[List[Dict[str, str]]] = None,
            tone: str = "neutral",
            language: str = "en",
    ) -> List[Dict[str, str]]:
        tone_instruction = self.TONE_PRESETS.get(tone, self.TONE_PRESETS["neutral"])
        lang = (language or "en").strip().lower()

        system = (
            f"Respond in {lang}. Do not switch languages unless the user asks.\n\n"
            "You are helping the user understand what indexed IFU documents cover using ONLY the excerpts provided.\n\n"
            "HARD RULES (MOST IMPORTANT):\n"
            "- You MUST ground every statement in the provided excerpts.\n"
            "- Do NOT guess, do NOT generalise from typical IFU structure.\n"
            "- If an excerpt does not show something, say: 'Not shown in the excerpts.'\n"
            "- If a document has no excerpts, say: 'No excerpts returned for this document (filter too strict or no data).'\n\n"
            "D# DOCUMENT LABELS (IMPORTANT):\n"
            "- Each document has a label D1, D2, D3... (DOCUMENT MAP).\n"
            "- ALWAYS refer to documents using their D# label in your answer.\n"
            "- You may include the filename, but always include D#.\n"
            "- Do NOT mention any document not in the DOCUMENT MAP.\n\n"
            "OUTPUT FORMAT (STRICT):\n"
            "1) DOCUMENT MAP: one line per doc.\n"
            "2) FOR EACH DOCUMENT:\n"
            "   - Title line: D# — filename\n"
            "   - 'What the excerpts show' (1–2 sentences)\n"
            "   - 'Key topics visible' (3–6 bullets)\n"
            "   - 'Evidence snippets' (quote 1–3 short verbatim snippets from the excerpts)\n\n"
            "REGULATORY SAFETY:\n"
            "- Do not provide medical advice beyond what is shown.\n\n"
            f"TONE GUIDANCE:\n{tone_instruction}"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

        # history for conversational flow (not factual)
        for m in (history or []):
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # Build a compact but explicit document map + excerpt payload
        doc_map_lines: List[str] = []
        excerpt_blocks: List[str] = []

        for idx, d in enumerate(samples or [], start=1):
            d_label = (d.get("d_label") or "").strip() or f"D{idx}"
            doc_name = d.get("doc_name") or d.get("doc_id") or "Unknown"

            doc_map_lines.append(f"{d_label} = {doc_name}")

            excerpt_blocks.append(f"{d_label} DOCUMENT: {doc_name}")

            chunks = d.get("sample_chunks") or []
            if not chunks:
                excerpt_blocks.append("EXCERPTS: [none returned for requested filter]")
                excerpt_blocks.append("")
                continue

            # include a numbered excerpt list (easy for the model to quote)
            excerpt_blocks.append("EXCERPTS:")
            for j, c in enumerate(chunks, start=1):
                page = c.get("page_start")
                cl = c.get("lang")
                txt = (c.get("text") or "").strip()

                # keep excerpts short to reduce token bloat + force precision
                if len(txt) > 450:
                    txt = txt[:450].rstrip() + "…"

                meta_bits = []
                if cl:
                    meta_bits.append(f"lang={cl}")
                if page:
                    meta_bits.append(f"p.{page}")
                meta = f" ({', '.join(meta_bits)})" if meta_bits else ""

                excerpt_blocks.append(f"[{d_label}.{j}]{meta} {txt}")

            excerpt_blocks.append("")

        doc_map_text = "\n".join(doc_map_lines).strip() if doc_map_lines else "[no documents returned]"
        excerpts_text = "\n".join(excerpt_blocks).strip() if excerpt_blocks else "[no excerpts returned]"

        user = (
            f"Question:\n{question}\n\n"
            "DOCUMENT MAP:\n"
            f"{doc_map_text}\n\n"
            "EXCERPTS BY DOCUMENT (you MUST use these as the ONLY evidence):\n"
            f"{excerpts_text}\n\n"
            "Now answer using the STRICT OUTPUT FORMAT. "
            "When you quote evidence snippets, quote from the excerpt IDs like [D1.1], [D2.2] etc."
        )

        messages.append({"role": "user", "content": user})
        return messages

    def _build_ops_messages(
            self,
            *,
            question: str,
            ops_context: str,
            history: Optional[List[Dict[str, str]]] = None,
            tone: str = "neutral",
            language: str = "en",
    ) -> List[Dict[str, str]]:
        tone_instruction = self.TONE_PRESETS.get(tone, self.TONE_PRESETS["neutral"])
        lang = (language or "en").strip().lower()

        system = (
            f"Respond in {lang}. Do not switch languages unless the user asks.\n\n"
            "You answer ONLY operational/system questions about storage, indexing, corpora, and ingestion.\n\n"
            "GROUNDING RULES (MOST IMPORTANT):\n"
            "- Use ONLY the provided OPERATIONAL CONTEXT.\n"
            "- If the answer is not present, say exactly what is missing.\n"
            "- Do NOT use IFU excerpts or clinical content.\n"
            "- Do NOT guess.\n\n"
            "OUTPUT:\n"
            "- Be explicit and structured.\n"
            "- Prefer short bullet lists.\n\n"
            f"TONE GUIDANCE:\n{tone_instruction}"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

        for m in (history or []):
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        user = (
            f"Question:\n{question}\n\n"
            "OPERATIONAL CONTEXT:\n"
            f"{ops_context}\n\n"
            "Answer using ONLY OPERATIONAL CONTEXT."
        )
        messages.append({"role": "user", "content": user})
        return messages







