# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-11
# Description: IFUChunker
# -----------------------------------------------------------------------------
import logging
import uuid
from typing import List, Callable, Dict, Any, Optional, Tuple, Counter

from chunking.IFUChunk import IFUChunk
from utility.logging_utils import get_class_logger


# from .ifu_chunk import IFUChunk  # adjust to your path


class IFUChunker:
    """
    Splits IFU documents into language-aware IFUChunk objects.
    Expects `pages` to be List[str] (page 1 = pages[0]).
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        lang_detector,
        chunk_size_tokens: int = 300,
        overlap_tokens: int = 100,
        page_fallback_threshold: float = 0.65,   # if chunk conf < threshold, use page lang
        logger = None,
    ):
        """
        :param tokenizer: function(text) -> List[str]
        :param lang_detector: has detect(text[, fallback]) -> (lang:str, confidence:float, script:Optional[str])
        :param chunk_size_tokens: tokens per chunk
        :param overlap_tokens: tokens overlapping between chunks
        :param page_fallback_threshold: min chunk confidence before falling back to page language
        """
        self.tokenizer = tokenizer
        self.lang_detector = lang_detector
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        self.page_fallback_threshold = page_fallback_threshold
        self.logger = logger or get_class_logger(self.__class__)

    # --- internal helper so we work with detectors w/ or w/o fallback kwarg ---
    def _detect_with_optional_fallback(
        self, text: str, page_text: Optional[str] = None
    ) -> Tuple[str, float, Optional[str]]:
        try:
            # Try detector that supports fallback keyword
            return self.lang_detector.detect(text, fallback=page_text)  # type: ignore[arg-type]
        except TypeError:
            # Fallback to simple signature
            return self.lang_detector.detect(text)

    def chunk_document(
            self,
            doc_id: str,
            doc_name: str,
            pages: List[str],
            doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> List["IFUChunk"]:

        # Logger setup
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

        if doc_metadata is None:
            doc_metadata = {}

        version = doc_metadata.get("version")
        region = doc_metadata.get("region")
        is_primary_language = doc_metadata.get("is_primary_language", False)

        if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
            raise TypeError("`pages` must be List[str] (one string per page).")

        chunks: List["IFUChunk"] = []

        # Stats collection
        lang_counts = Counter()
        lang_conf_vals = []
        chunk_lengths = []

        char_offset = 0

        # Begin Processing Pages
        for page_index, page_text in enumerate(pages, start=1):

            page_lang, page_confidence, page_script = self._detect_with_optional_fallback(
                page_text, page_text
            )

            tokens = self.tokenizer(page_text)
            num_tokens = len(tokens)

            start_idx = 0
            while start_idx < num_tokens:
                end_idx = min(start_idx + self.chunk_size_tokens, num_tokens)
                window_tokens = tokens[start_idx:end_idx]
                chunk_text = " ".join(window_tokens)

                # Detect language for chunk (with fallback)
                lang, lang_confidence, script = self._detect_with_optional_fallback(
                    chunk_text, page_text
                )
                if (lang == "und" or lang_confidence < self.page_fallback_threshold) and page_lang != "und":
                    lang, lang_confidence, script = page_lang, page_confidence, page_script

                # Build chunk
                chunk = IFUChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    text=chunk_text,
                    page_start=page_index,
                    page_end=page_index,
                    char_start=char_offset + len(" ".join(tokens[:start_idx])),
                    char_end=char_offset + len(" ".join(tokens[:start_idx])) + len(chunk_text),
                    lang=lang,
                    lang_confidence=lang_confidence,
                    script=script,
                    version=version,
                    region=region,
                    section_type=None,
                    is_primary_language=is_primary_language,
                    translation_group_id=None,
                )

                if doc_metadata:
                    chunk.metadata.update(doc_metadata)

                chunks.append(chunk)

                # Essential chunk logging per chunk
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

        # Summary Logging
        total_chunks = len(chunks)
        if total_chunks > 0:
            avg_len = sum(chunk_lengths) / total_chunks
            avg_conf = sum(lang_conf_vals) / total_chunks

            self.logger.info(
                "Chunking Summary: chunks=%d | avg_len=%.1f chars | lang_counts=%s | avg_lang_conf=%.2f",
                total_chunks,
                avg_len,
                dict(lang_counts),
                avg_conf
            )
        else:
            self.logger.warning("No chunks produced for doc_id=%s doc_name=%r", doc_id, doc_name)

        return chunks



