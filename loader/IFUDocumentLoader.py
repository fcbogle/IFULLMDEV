# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-29
# Description: IFUDocumentLoader.py
# -----------------------------------------------------------------------------
import re
from dataclasses import field
from pathlib import Path
from typing import Any, Iterable, Dict

from chunking.LangDetectDetector import LangDetectDetector
from config.Config import Config
from chunking.IFUChunker import IFUChunker
from embedding.IFUEmbedder import IFUEmbedder
from ingestion.IFUFileLoader import IFUFileLoader
from vectorstore.ChromaIFUVectorStore import ChromaIFUVectorStore
from vectorstore.IFUVectorStore import IFUVectorStore
from utility.logging_utils import get_class_logger

import logging


class IFUDocumentLoader:
    cfg: Config
    collection_name: str = "ifu_chunks_all"
    loader: IFUFileLoader = field(init=False)
    chunker: IFUChunker = field(init=False)
    embedder: IFUEmbedder = field(init=False)
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













