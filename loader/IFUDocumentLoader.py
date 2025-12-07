# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-29
# Description: IFUDocumentLoader.py
# -----------------------------------------------------------------------------
import logging
import re
from dataclasses import field
from pathlib import Path
from typing import Any, Iterable, Dict, List

from chunking.IFUChunker import IFUChunker
from chunking.LangDetectDetector import LangDetectDetector
from config.Config import Config
from embedding.IFUEmbedder import IFUEmbedder
from extractor.IFUTextExtractor import IFUTextExtractor
from ingestion.IFUFileLoader import IFUFileLoader
from utility.logging_utils import get_class_logger
from vectorstore.ChromaIFUVectorStore import ChromaIFUVectorStore
from vectorstore.IFUVectorStore import IFUVectorStore


class IFUDocumentLoader:
    cfg: Config
    collection_name: str = "ifu_chunks_all"
    loader: IFUFileLoader = field(init=False)
    chunker: IFUChunker = field(init=False)
    embedder: IFUEmbedder = field(init=False)
    extractor: Any = None
    store: IFUVectorStore = field(init=False)
    lang_detector: Any = None
    logger: Any = None

    def __init__(self,
                 cfg: Config,
                 *,
                 collection_name: str = "ifu_chunks",
                 lang_detector: Any = None,
                 logger: logging.Logger | None = None
                 ):
        self.cfg = cfg
        self.collection_name = collection_name
        self.lang_detector = lang_detector or LangDetectDetector()
        self.logger = logger or get_class_logger(self.__class__)

        self.logger.info(
            "Initialising IFUMultiDocumentLoader (collection=%s)",
            self.collection_name
        )

        # File Loader
        self.loader = IFUFileLoader(cfg=self.cfg)

        # Tokenizer
        tokenizer = lambda text: re.findall(r"\w+|\S", text)

        # Chunker
        self.chunker = IFUChunker(
            tokenizer=tokenizer,
            lang_detector=self.lang_detector,
            chunk_size_tokens=300,
            overlap_tokens=100,
        )

        # Embedder
        self.embedder = IFUEmbedder(
            cfg=self.cfg,
            batch_size=16,
            normalize=True,
            out_dtype="float32",
            filter_lang=None,
        )

        # Extractor
        self.extractor = IFUTextExtractor()

        # Chroma vector store
        self.store = ChromaIFUVectorStore(
            cfg=self.cfg,
            embedder=self.embedder,
            collection_name=self.collection_name
        )

        self.logger.info("IFUMultiDocumentLoader initialised successfully.")

    def upload_multiple_pdfs(self,
                             pdf_paths: Iterable[str | Path],
                             *,
                             container: str = "ifu_docs",
                             blob_prefix: str = "",
                             ) -> Dict[Path, str]:
        """
        Upload multiple PDFs to Azure Blob Storage using IFUFileLoader.
        Explicit for-loop version (no implicit pass-through).

        Args:
            pdf_paths: Iterable of local file paths.
            container: Target blob container.
            blob_prefix: Optional prefix for uploaded blob names.

        Returns:
            A dict mapping local Path -> uploaded blob name.
        """

        results: Dict[Path, str] = {}
        for p in pdf_paths:
            path_obj = Path(p)

            # Ensure the file exists
            if not path_obj.is_file():
                self.logger.error("Skipping this file: %s", path_obj)
                continue

            # Define blob name
            blob_name = f"{blob_prefix}{path_obj.name}" if blob_prefix else path_obj.name

            try:
                # Perform pdf upload
                uploaded = self.loader.upload_document_from_path(
                    local_path=str(path_obj),
                    container=container,
                    blob_name=blob_name,
                )
                results[path_obj] = uploaded

                self.logger.info("Uploaded '%s' as blob '%s'", path_obj, uploaded)

            except Exception as e:
                self.logger.error("Failed to upload '%s': %s", path_obj, e)

        return results

    def ingest_blob_pdfs(self,
                         blob_names: Iterable[str],
                         *,
                         container: str = "ifu_docs",
                         ) -> int:
        """
        Ingest existing PDFs from Azure Blob into the vector store.

        For each blob:
          - download PDF bytes
          - extract text
          - chunk text
          - embed & store chunks

        Args:
            blob_names: Iterable of blob names (within the given container).
            container: Blob container where PDFs live.

        Returns:
            Number of blobs successfully ingested.
        """

        ingested_count = 0
        for blob_name in blob_names:
            try:
                self._process_single_pdf(
                    container=container,
                    blob_name=blob_name,
                    source_path=None
                )
                ingested_count += 1
            except Exception as e:
                self.logger.error("Failed to ingest blob '%s': %s",
                                  blob_name,
                                  container,
                                  e,
                                  )
                self.logger.info(
                    "Blob ingest complete: %d/%d blobs successfully indexed",
                    ingested_count,
                    len(list(blob_names)),
                )
        return ingested_count

    def _process_single_pdf(self,
                            *,
                            container: str,
                            blob_name: str,
                            source_path: Path | None = None,
                            ) -> None:
        """
        Process a single PDF already present in Blob Storage.

        Steps:
          - Download PDF bytes
          - Extract text
          - Chunk (with language detection)
          - Add chunks to vector store

        """
        self.logger.info(
        "processing blob '%s' from container '%s' into collection '%s", blob_name, container, self.collection_name
        )

        # Download PDF from blob storage
        pdf_bytes = self.loader.load_document(blob_name=blob_name, container=container)
        if not pdf_bytes:
            raise ValueError(f"No bytes returned for blob '{blob_name}' from container '{container}'")

        # Extract text from PDF bytes
        text = self.extractor.extract_text_from_pdf(pdf_bytes)

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty text extracted for blob '{blob_name}'")

        # Get per page text
        pages: List[str]
        if hasattr(self.extractor, "extract_pages"):
            pages = self.extractor.extract_pages(pdf_bytes)
            if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
                self.logger.warning(
                    "Page extraction returned invalid format for blob '%s'; "
                    "falling back to single page.",
                    blob_name,
                )
                pages = [text]
        else:
            pages = [text]


        # Build IDs and metadata for chunker
        doc_id = blob_name  # simple, stable ID; can be changed later if needed
        doc_name = source_path.name if source_path is not None else blob_name

        doc_metadata: Dict[str, Any] = {
            "blob_name": blob_name,
            "container": container,
        }
        if source_path is not None:
            doc_metadata["source_path"] = str(source_path)
            doc_metadata["filename"] = source_path.name
        else:
            doc_metadata["filename"] = blob_name

        # Chunk the document
        chunks = self.chunker.chunk_document(doc_id=doc_id, doc_name=doc_name, pages=pages, doc_metadata=doc_metadata)

        if not chunks:
            self.logger.warning(
                "No chunks produced for doc_id=%s doc_name=%r (blob=%s)",
                doc_id,
                doc_name,
                blob_name,
            )
            return

        self.logger.info(
            "Generated %d chunks for doc_id=%s doc_name=%r",
            len(chunks),
            doc_id,
            doc_name,
        )

        # Embed chunks using IFUEmbedder
        embedding_records = self.embedder.embed_chunks(chunks)
        if not embedding_records:
            self.logger.warning(
                "No embeddings produced for doc_id=%s doc_name=%r (blob=%s)",
                doc_id,
                doc_name,
                blob_name,
            )
            return

        # Check length of embedding records prior to upsert
        if len(embedding_records) != len(chunks):
            self.logger.error("Embedding records length mismatch: %d != %d", len(embedding_records), len(chunks))
            raise ValueError(f"Expected {len(chunks)} embedding records, got {len(embedding_records)}")

        # Upsert chunks and records into Chroma vector store
        self.store.upsert_chunk_embeddings(doc_id, chunks, records=embedding_records)

        self.logger.info(
            "Successfully ingested blob '%s' into collection '%s'", blob_name, self.collection_name
        )

