# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-21
# Description: IFUStatsService.py
# -----------------------------------------------------------------------------

import logging
from typing import Any, Dict, List

from loader.IFUDocumentLoader import IFUDocumentLoader
from loader.types import IFUStatsDict, DocumentEntry, BlobEntry

from utility.logging_utils import get_class_logger
from vectorstore.IFUVectorStore import IFUVectorStore


class IFUStatsService:
    """
    Stats service for the /stats endpoint.

    Responsibilities:
      - query Chroma for doc + chunk stats
      - query blob storage for blob list
      - return response matching IFUStatsResponse schema (incl total_documents)
    """

    def __init__(
        self,
        *,
        document_loader: IFUDocumentLoader,
        store: IFUVectorStore,
        collection_name: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.document_loader = document_loader
        self.store = store
        self.collection_name = collection_name
        self.logger = logger or get_class_logger(self.__class__)

    def get_stats(self, *, blob_container: str) -> IFUStatsDict:
        self.logger.info(
            "Stats for collection='%s', blob_container='%s'",
            self.collection_name,
            blob_container,
        )

        # --- Chroma / embeddings ---
        try:
            total_chunks = self.store.collection.count()
        except Exception as e:
            self.logger.error("Failed to count Chroma collection '%s': %s", self.collection_name, e)
            total_chunks = 0

        docs_raw = self.store.list_documents()

        documents: List[DocumentEntry] = []
        for d in docs_raw:
            if not isinstance(d, dict):
                continue

            doc_id = d.get("doc_id")
            chunk_count = d.get("chunk_count")
            page_count = d.get("page_count")
            last_modified = d.get("last_modified")
            document_type = d.get("document_type")

            if isinstance(doc_id, str) and isinstance(chunk_count, int):
                documents.append(
                    DocumentEntry(
                        doc_id=doc_id,
                        chunk_count=chunk_count,
                        page_count=page_count if page_count is not None else None,
                        last_modified=last_modified if isinstance(last_modified, str) else None,
                        document_type=document_type if isinstance(document_type, str) else None,
                    )
                )

        # --- Blob side ---
        blob_details = self.document_loader.get_blob_details(container=blob_container)

        blobs: List[BlobEntry] = []
        for b in blob_details:
            blobs.append(
                BlobEntry(
                    blob_name=b.get("blob_name"),
                    size=b.get("size"),
                    content_type=b.get("content_type"),
                    last_modified=b.get("last_modified"),
                )
            )

        total_blobs = len(blobs)

        # FIX: include total_documents to satisfy IFUStatsResponse
        total_documents = len(documents) if documents else len(blobs)

        return IFUStatsDict(
            collection_name=self.collection_name,
            total_chunks=total_chunks,
            total_documents=total_documents,
            total_blobs=total_blobs,
            documents=documents,
            blob_container=blob_container,
            blobs=blobs,
        )


