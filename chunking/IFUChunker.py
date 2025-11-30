# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-11
# Description: IFUChunker
# -----------------------------------------------------------------------------
import logging
import uuid
from typing import List, Callable, Dict, Any, Optional, Tuple
from collections import Counter

from chunking.IFUChunk import IFUChunk
from utility.logging_utils import get_class_logger


class IFUChunker:
    """
    Splits IFU documents into language-aware IFUChunk objects.
    Expects `pages` to be List[str] (page 1 = pages[0]).
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        lang_detector,
        *,
        chunk_size_tokens: int = 300,
        overlap_tokens: int = 100,
        page_fallback_threshold: float = 0.65,
        logger: logging.Logger | None = None,
    ):
        self.tokenizer = tokenizer
        self.lang_detector = lang_detector
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        self.page_fallback_threshold = page_fallback_threshold
        self.logger = logger or get_class_logger(self.__class__)

        # guard against bad config that can cause infinite loops
        if self.overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be < chunk_size_tokens ({self.chunk_size_tokens})"
            )

    def _detect_with_optional_fallback(
        self, text: str, page_text: Optional[str] = None
    ) -> Tuple[str, float, Optional[str]]:
        """
        Works with detectors that either support detect(text, fallback=...)
        or only detect(text).
        """
        try:
            return self.lang_detector.detect(text, fallback=page_text)  # type: ignore[arg-type]
        except TypeError:
            return self.lang_detector.detect(text)

    def chunk_document(
        self,
        doc_id: str,
        doc_name: str,
        pages: List[str],
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> List["IFUChunk"]:

        # Logger safety
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

        doc_metadata = doc_metadata or {}

        version = doc_metadata.get("version")
        region = doc_metadata.get("region")
        is_primary_language = doc_metadata.get("is_primary_language", False)

        if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
            raise TypeError("`pages` must be List[str] (one string per page).")

        chunks: List["IFUChunk"] = []

        # Stats
        lang_counts = Counter()
        lang_conf_vals: List[float] = []
        chunk_lengths: List[int] = []

        char_offset = 0

        self.logger.info(
            "Chunking doc_id=%s doc_name=%r pages=%d chunk_size=%d overlap=%d",
            doc_id, doc_name, len(pages), self.chunk_size_tokens, self.overlap_tokens
        )

        for page_index, page_text in enumerate(pages, start=1):

            # page-level language detection doesn't need fallback
            page_lang, page_confidence, page_script = self._detect_with_optional_fallback(
                page_text, None
            )

            tokens = self.tokenizer(page_text)
            num_tokens = len(tokens)

            start_idx = 0
            while start_idx < num_tokens:
                end_idx = min(start_idx + self.chunk_size_tokens, num_tokens)
                window_tokens = tokens[start_idx:end_idx]
                chunk_text = " ".join(window_tokens)

                # chunk-level detection with page fallback
                lang, lang_confidence, script = self._detect_with_optional_fallback(
                    chunk_text, page_text
                )

                # enforce page fallback if chunk is weak
                if (lang == "und" or lang_confidence < self.page_fallback_threshold) and page_lang != "und":
                    lang, lang_confidence, script = page_lang, page_confidence, page_script

                # compute once for efficiency
                chars_before_len = len(" ".join(tokens[:start_idx]))
                chunk_char_start = char_offset + chars_before_len
                chunk_char_end = chunk_char_start + len(chunk_text)

                chunk = IFUChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    text=chunk_text,
                    page_start=page_index,
                    page_end=page_index,
                    char_start=chunk_char_start,
                    char_end=chunk_char_end,
                    lang=lang,
                    lang_confidence=lang_confidence,
                    script=script,
                    version=version,
                    region=region,
                    section_type=None,
                    is_primary_language=is_primary_language,
                    translation_group_id=None,
                )

                # assumes IFUChunk.metadata is Dict[str, Any]
                if doc_metadata and isinstance(chunk.metadata, dict):
                    chunk.metadata.update(doc_metadata)

                chunks.append(chunk)

                # essential per-chunk logging
                chunk_len = len(chunk_text)
                chunk_lengths.append(chunk_len)
                lang_counts[lang] += 1
                lang_conf_vals.append(lang_confidence)

                self.logger.info(
                    "Chunk created: idx=%d page=%d lang=%s conf=%.2f length=%d chars",
                    len(chunks), page_index, lang, lang_confidence, chunk_len
                )

                if end_idx == num_tokens:
                    break

                start_idx = end_idx - self.overlap_tokens

            char_offset += len(page_text) + 1

        # summary
        total_chunks = len(chunks)
        if total_chunks:
            avg_len = sum(chunk_lengths) / total_chunks
            avg_conf = sum(lang_conf_vals) / total_chunks

            self.logger.info(
                "Chunking Summary: chunks=%d | avg_len=%.1f chars | lang_counts=%s | avg_lang_conf=%.2f",
                total_chunks, avg_len, dict(lang_counts), avg_conf
            )
        else:
            self.logger.warning("No chunks produced for doc_id=%s doc_name=%r", doc_id, doc_name)

        return chunks




