# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-11
# Description: IFUChunker
# -----------------------------------------------------------------------------
import uuid
from typing import List, Callable, Dict, Any, Optional

from IFUChunk import IFUChunk


class IFUChunker:
    """
        Splits IFU documents into language-aware IFUChunk objects.
        Requires a utility class to extract page texts from PDF bytes.
    """

    def __init__(self,
                 tokenizer: Callable[[str], List[str]],
                 lang_detector,
                 chunk_size_tokens: int = 800,
                 overlap_tokens: int = 100,
                 ):
        """
        :param tokenizer: function(text) -> List[str]
                          e.g. lambda s: s.split() or a tiktoken-based tokenizer.
        :param lang_detector: object with:
                              detect(text: str) -> (lang: str, confidence: float, script: Optional[str])
        :param chunk_size_tokens: target number of tokens per chunk
        :param overlap_tokens: number of tokens to overlap between consecutive chunks
        """
        self.tokenizer = tokenizer
        self.lang_detector = lang_detector
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(
            self,
            doc_id: str,
            doc_name: str,
            pages: List[str],
            doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[IFUChunk]:
        if doc_metadata is None:
            doc_metadata = {}

        version = doc_metadata.get("version")
        region = doc_metadata.get("region")
        is_primary_language = doc_metadata.get("is_primary_language", False)

        chunks: List[IFUChunk] = []
        char_offset = 0  # approximate character offset in full document

        for page_index, page_text in enumerate(pages, start=1):
            tokens = self.tokenizer(page_text)
            num_tokens = len(tokens)

            start_idx = 0
            while start_idx < num_tokens:
                end_idx = min(start_idx + self.chunk_size_tokens, num_tokens)
                window_tokens = tokens[start_idx:end_idx]
                chunk_text = " ".join(window_tokens)

                # Language detection for this chunk
                lang, lang_confidence, script = self.lang_detector.detect(chunk_text)

                # Approximate char positions (good enough for traceability)
                chars_before = " ".join(tokens[:start_idx])
                chunk_char_start = char_offset + len(chars_before)
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
                    section_type=None,  # you can set this later
                    is_primary_language=is_primary_language,
                    translation_group_id=None,  # you can set this later too
                )

                # Optionally propagate all doc-level metadata into chunk.metadata
                if doc_metadata:
                    chunk.metadata.update(doc_metadata)

                chunks.append(chunk)

                if end_idx == num_tokens:
                    break

                # Slide window forward with overlap
                start_idx = end_idx - self.overlap_tokens

            # Move global char offset by this page's length (+1 as separator)
            char_offset += len(page_text) + 1

        return chunks
