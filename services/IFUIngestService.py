# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-21
# Description: IFUIngestService.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from chunking.IFUChunker import IFUChunker
from chunking.LangDetectDetector import LangDetectDetector
from extractor.IFUTextExtractor import IFUTextExtractor
from embedding.IFUEmbedder import IFUEmbedder
from utility.logging_utils import get_class_logger
from vectorstore.IFUVectorStore import IFUVectorStore
from loader.IFUDocumentLoader import IFUDocumentLoader


class IFUIngestService:
    """
    Owns the ingest/index pipeline:
      - read blob bytes (via IFUDocumentLoader)
      - extract text/pages
      - chunk (with language detection)
      - embed
      - upsert into vector store
    """

    def __init__(
        self,
        *,
        document_loader: IFUDocumentLoader,
        store: IFUVectorStore,
        embedder: IFUEmbedder,
        chunker: IFUChunker,
        extractor: IFUTextExtractor,
        collection_name: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.document_loader = document_loader
        self.store = store
        self.embedder = embedder
        self.chunker = chunker
        self.extractor = extractor
        self.collection_name = collection_name
        self.logger = logger or get_class_logger(self.__class__)

    @staticmethod
    def build_default_chunker(
        *,
        lang_detector: Any | None = None,
        chunk_size_tokens: int = 300,
        overlap_tokens: int = 100,
    ) -> IFUChunker:
        tokenizer = lambda text: re.findall(r"\w+|\S", text)
        detector = lang_detector or LangDetectDetector()
        return IFUChunker(
            tokenizer=tokenizer,
            lang_detector=detector,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )

    def ingest_blob_pdfs(
        self,
        blob_names: Iterable[str],
        *,
        container: str,
        document_type: str = "IFU",
    ) -> int:
        ingested_count = 0
        blob_list = list(blob_names)

        for blob_name in blob_list:
            try:
                self._process_single_blob_pdf(
                    container=container,
                    blob_name=blob_name,
                    document_type=document_type,
                )
                ingested_count += 1
            except Exception as e:
                self.logger.error("Failed ingest for blob '%s': %s", blob_name, e, exc_info=True)

        self.logger.info(
            "Blob ingest complete: %d/%d blobs successfully indexed",
            ingested_count,
            len(blob_list),
        )
        return ingested_count

    def _process_single_blob_pdf(
        self,
        *,
        container: str,
        blob_name: str,
        document_type: str,
        source_path: Optional[Path] = None,
    ) -> None:
        self.logger.info(
            "Processing blob '%s' from container '%s' into collection '%s'",
            blob_name,
            container,
            self.collection_name,
        )

        pdf_bytes = self.document_loader.load_blob_bytes(container=container, blob_name=blob_name)

        raw = self.extractor.extract_text_from_pdf(pdf_bytes)

        # Normalise -> pages: List[str]
        if isinstance(raw, list) and all(isinstance(p, str) for p in raw):
            pages: List[str] = raw
        elif isinstance(raw, list) and all(isinstance(p, list) for p in raw):
            pages = [" ".join(part for part in page if isinstance(part, str)) for page in raw]
        elif isinstance(raw, str):
            if not raw.strip():
                raise ValueError(f"Empty text extracted for blob '{blob_name}'")
            pages = [raw]
        else:
            raise TypeError(f"Unexpected type from extractor: {type(raw)}")

        page_count = len(pages)
        last_modified = self.document_loader.try_get_last_modified_iso(
            container=container, blob_name=blob_name
        )

        doc_id = blob_name
        doc_name = source_path.name if source_path is not None else blob_name

        doc_metadata: Dict[str, Any] = {
            "blob_name": blob_name,
            "container": container,
            "filename": doc_name,
            "page_count": page_count,
            "last_modified": last_modified,
            "document_type": document_type,
        }
        if source_path is not None:
            doc_metadata["source_path"] = str(source_path)

        self.logger.info("Doc metadata before chunking: %r", doc_metadata)

        chunks = self.chunker.chunk_document(
            doc_id=doc_id,
            doc_name=doc_name,
            pages=pages,
            doc_metadata=doc_metadata,
        )
        if not chunks:
            self.logger.warning("No chunks produced for blob '%s'", blob_name)
            return

        records = self.embedder.embed_chunks(chunks)
        if not records:
            self.logger.warning("No embeddings produced for blob '%s'", blob_name)
            return

        if len(records) != len(chunks):
            raise ValueError(f"Embedding count mismatch: {len(records)} != {len(chunks)}")

        self.store.upsert_chunk_embeddings(doc_id, chunks, records=records)

        self.logger.info(
            "Successfully ingested blob '%s' into collection '%s'",
            blob_name,
            self.collection_name,
        )
