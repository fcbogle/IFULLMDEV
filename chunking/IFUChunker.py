# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-11
# Description: IFUChunker
# -----------------------------------------------------------------------------
import uuid
from typing import List, Callable, Dict, Any, Optional, Tuple

from chunking.IFUChunk import IFUChunk


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
        if doc_metadata is None:
            doc_metadata = {}

        version = doc_metadata.get("version")
        region = doc_metadata.get("region")
        is_primary_language = doc_metadata.get("is_primary_language", False)

        # Basic sanity checks (helpful in integration tests)
        if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
            raise TypeError("`pages` must be List[str] with one entry per PDF page.")

        chunks: List["IFUChunk"] = []
        char_offset = 0  # approximate character offset in full document

        for page_index, page_text in enumerate(pages, start=1):
            # 1) Detect language at the PAGE level (for fallback)
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

                # 2) Detect language at the CHUNK level, with page fallback support
                lang, lang_confidence, script = self._detect_with_optional_fallback(
                    chunk_text, page_text
                )

                # 3) Enforce page-level fallback if chunk detection is weak/undetermined
                if (lang == "und" or lang_confidence < self.page_fallback_threshold) and page_lang != "und":
                    lang, lang_confidence, script = page_lang, page_confidence, page_script

                # 4) Approximate char positions (good enough for traceability)
                chars_before = " ".join(tokens[:start_idx])
                chunk_char_start = char_offset + len(chars_before)
                chunk_char_end = chunk_char_start + len(chunk_text)

                # 5) Build IFUChunk
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
                    section_type=None,           # set later if you have section tagging
                    is_primary_language=is_primary_language,
                    translation_group_id=None,   # set later if you align translations
                )

                # Optionally propagate all doc-level metadata into chunk.metadata
                if doc_metadata:
                    chunk.metadata.update(doc_metadata)

                chunks.append(chunk)

                if end_idx == num_tokens:
                    break
                # Slide window with overlap
                start_idx = end_idx - self.overlap_tokens

            # Move global char offset by this page's length (+1 as separator)
            char_offset += len(page_text) + 1

        return chunks



