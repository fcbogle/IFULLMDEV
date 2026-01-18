# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-21
# Description: IFUIngestService.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional

from config.Config import Config
from chunking.IFUChunker import IFUChunker
from chunking.LangDetectDetector import LangDetectDetector
from embedding.IFUEmbedder import IFUEmbedder
from extractor.IFUTextExtractor import IFUTextExtractor
from loader.IFUDocumentLoader import IFUDocumentLoader
from utility.logging_utils import get_class_logger
from vectorstore.IFUVectorStore import IFUVectorStore

from settings import BLOB_CONTAINER_DEFAULT, VECTOR_COLLECTION_DEFAULT, ACTIVE_CORPUS_ID

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
            collection_name: str | None = None,
            logger: logging.Logger | None = None,
    ) -> None:
        self.document_loader = document_loader
        self.store = store
        self.embedder = embedder
        self.chunker = chunker
        self.extractor = extractor
        self.collection_name = (collection_name or VECTOR_COLLECTION_DEFAULT).strip()
        self.logger = logger or get_class_logger(self.__class__)



    @staticmethod
    def build_default_chunker(
            *,
            lang_detector: Any | None = None,
            cfg: Config | None = None,
    ) -> IFUChunker:
        tokenizer = lambda text: re.findall(r"\w+|\S", text)
        detector = lang_detector or LangDetectDetector()

        cfg = cfg or Config.from_env()

        return IFUChunker(
            tokenizer=tokenizer,
            lang_detector=detector,
            cfg=cfg,
        )

    def ingest_blob_pdfs(
            self,
            blob_names: Iterable[str],
            *,
            container: str | None = None,
            document_type: str = "IFU",
    ) -> Dict[str, Any]:
        container = (container or BLOB_CONTAINER_DEFAULT).strip()
        blob_list = list(blob_names)

        ingested = 0
        rejected = 0
        errors = 0

        results: List[Dict[str, Any]] = []

        for blob_name in blob_list:
            ext = os.path.splitext(blob_name.lower())[1]

            # --- policy gate: only PDFs for now ---
            if ext != ".pdf":
                rejected += 1
                results.append({
                    "blob_name": blob_name,
                    "ok": False,
                    "status": "rejected",
                    "message": (
                        f"Uploading/ingesting '{ext or '(none)'}' is not supported yet. "
                        f"Please upload PDF files only."
                    ),
                })
                continue

            try:
                # (Optional but recommended) download bytes once so we can sanity-check header
                pdf_bytes = self.document_loader.load_blob_bytes(container=container, blob_name=blob_name)

                if not pdf_bytes or not pdf_bytes.lstrip().startswith(b"%PDF"):
                    errors += 1
                    msg = "File does not look like a valid PDF (missing %PDF header)."
                    self.logger.warning("Skipping blob '%s': %s", blob_name, msg)
                    results.append({
                        "blob_name": blob_name,
                        "ok": False,
                        "status": "error",
                        "message": msg,
                    })
                    continue

                # Process using existing pipeline; tweak _process_single_blob_pdf to accept bytes
                self._process_single_blob_pdf(
                    container=container,
                    blob_name=blob_name,
                    document_type=document_type,
                    pdf_bytes=pdf_bytes,
                )

                ingested += 1
                results.append({
                    "blob_name": blob_name,
                    "ok": True,
                    "status": "ingested",
                })

            except Exception as e:
                errors += 1
                self.logger.error("Failed ingest for blob '%s': %s", blob_name, e, exc_info=True)
                results.append({
                    "blob_name": blob_name,
                    "ok": False,
                    "status": "error",
                    "message": str(e),
                })

        out = {
            "container": container,
            "document_type": document_type,
            "attempted": len(blob_list),
            "ingested": ingested,
            "rejected": rejected,
            "errors": errors,
            "results": results,
        }

        self.logger.info(
            "Blob ingest complete: ingested=%d rejected=%d errors=%d attempted=%d",
            ingested, rejected, errors, len(blob_list),
        )
        return out

    def _process_single_blob_pdf(
            self,
            *,
            container: str | None = None,
            blob_name: str,
            document_type: str,
            pdf_bytes: Optional[bytes] = None,
            source_path: Optional[Path] = None,) -> None:

        container = (container or BLOB_CONTAINER_DEFAULT).strip()

        self.logger.info(
            "Processing blob '%s' from container '%s' into collection '%s'",
            blob_name,
            container,
            self.collection_name,
        )

        # Only download if caller didn’t supply bytes
        if pdf_bytes is None:
            pdf_bytes = self.document_loader.load_blob_bytes(container=container, blob_name=blob_name)

        # Defensive sanity check to avoid deep PyMuPDF errors
        if not pdf_bytes or not pdf_bytes.lstrip().startswith(b"%PDF"):
            raise ValueError(f"Blob '{blob_name}' does not appear to be a valid PDF (missing %PDF header).")

        # Continue with your existing extraction/chunk/embed/index pipeline...
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

        self.store.upsert_chunk_embeddings(
            doc_id=doc_id,
            chunks=chunks,
            records=records,
            corpus_id=ACTIVE_CORPUS_ID,
        )

        self.logger.info(
            "Successfully ingested blob '%s' into collection '%s'",
            blob_name,
            self.collection_name,
        )
