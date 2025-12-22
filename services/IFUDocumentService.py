# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-22
# Description: IFUDocumentService.py
# -----------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

from loader.IFUDocumentLoader import IFUDocumentLoader
from services.IFUIngestService import IFUIngestService
from vectorstore.IFUVectorStore import IFUVectorStore


class IFUDocumentService:
    """
    Document facade used by FastAPI
    - list/retrieve documents from blob storage
    - trigger ingest/reingest via IFUIngestService
    - delete vectors by doc_id via IFUVectorStore (prototype)
    """

    def __init__(self,
                 *,
                 document_loader: IFUDocumentLoader,
                 ingest_service: IFUIngestService,
                 store: IFUVectorStore,
                 logger: logging.Logger | None = None, ) -> None:
        self.document_loader = document_loader
        self.ingest_service = ingest_service
        self.store = store

        self.logger = logger or logging.getLogger(__name__)

        self.logger.info("IFUDocumentService initialised successfully (loader=%s, ingest=%s, store=%s)",
                         type(document_loader).__name__, type(ingest_service).__name__, type(store).__name__)

    def list_documents(self, *, container: str) -> list[Dict[str, Any]]:
        # Get details (size/content_type/last_modified) for API usefulness
        try:
            details = self.document_loader.get_blob_details(container=container)
            # Enforce convention doc_id == blob_name
            for d in details:
                d["doc_id"] = d["blob_name"]
                d["container"] = container
                self.logger.info("list_documents: container='%s' -> %d documents (done)", container, len(details),)
            return details
        except Exception as e:
            self.logger.error("list_documents: container='%s' -> failed: %s", container, e, exc_info=True)
            raise

    def list_document_ids(self, *, container: str) -> list[str]:
        # Not as computationally expensive if only ids are needed
        self.logger.info("list_document_ids: container='%s' (start)", container,)
        try:
            names = self.document_loader.list_blob_names(container=container)
            self.logger.info("list_document_ids: container='%s' -> %d ids (done)", container, len(names),)
            return names
        except Exception as e:
            self.logger.error("list_document_ids: container='%s' -> failed: %s", container, e, exc_info=True)
            raise

    def get_document(self, *, container: str, doc_id: str) -> Dict[str, Any]:
        self.logger.info("get_document: container='%s' doc_id='%s' (start)", container, doc_id)
        try:
            docs = self.document_loader.get_blob_details(container=container)
            match = next((d for d in docs if d.get("blob_name") == doc_id), None)

            if not match:
                self.logger.warning("get_document: container='%s' doc_id='%s' -> not found", container, doc_id)
                raise KeyError(f"No document found with doc_id={doc_id} in container={container}")

            match["doc_id"] = doc_id
            match["container"] = container

            indexing: Optional[Dict[str, Any]] = None

            self.logger.info("get_document: container='%s' doc_id='%s' (done)", container, doc_id)
            return {"document": match, "indexing": indexing}

        except Exception as e:
            self.logger.error("get_document: container='%s' doc_id='%s' -> failed: %s", container, doc_id, e, exc_info=True)
            raise

    def document_exists(self, container: str, doc_id: str) -> bool:
        self.logger.info("document_exists: container='%s' doc_id='%s' (start)", container, doc_id,)
        try:
            names = self.document_loader.list_blob_names(container=container)
            exists = doc_id in names
            self.logger.info("document_exists: container='%s' doc_id='%s' exists=%s (done)", container, doc_id, exists,)
            return exists
        except Exception as e:
            self.logger.error("document_exists: container='%s' doc_id='%s' -> failed: %s", container, doc_id, e, exc_info=True)
            raise

    def get_document_bytes(self, container: str, doc_id: str) -> bytes:
        # Only needed to download bytes for text extraction
        self.logger.info("get_document_bytes: container='%s' doc_id='%s' (start)", container, doc_id,)
        try:
            data = self.document_loader.load_blob_bytes(container=container, blob_name=doc_id,)
            self.logger.info("get_document_bytes: container='%s' doc_id='%s' bytes=%d (done)", container, doc_id, len(data),)
            return data
        except Exception as e:
            self.logger.error("get_document_bytes: container='%s' doc_id='%s' -> failed: %s", container, doc_id, e, exc_info=True)
            raise

    def upload_documents(self, pdf_paths: Iterable[Path], *, container: str, blob_prefix: str) -> Dict[Path, str]:
        self.logger.info("upload_documents: container='%s' blob_prefix='%s' (start)", container, blob_prefix,)
        try:
            results = self.document_loader.upload_multiple_pdfs(pdf_paths=pdf_paths, container=container, blob_prefix=blob_prefix,)
            self.logger.info("upload_documents: container='%s' blob_prefix='%s' uploaded=%d (done)", container, blob_prefix, len(results),)
            return results
        except Exception as e:
            self.logger.error("upload_documents: container='%s' blob_prefix='%s' -> failed: %s", container, blob_prefix, e, exc_info=True)

    def ingest_documents(self, *, container: str, doc_ids: list[str], document_type: str = "IFU", ) -> int:
        self.logger.info("ingest_documents: container='%s' document_type='%s' doc_ids=%d (start)", container, doc_ids, document_type,)

        if not doc_ids:
            self.logger.warning("ingest_documents: container='%s' document_type='%s' requested=%d ingested=%d (done)", container, doc_ids)
            return 0

        try:
            ingested = self.ingest_service.ingest_blob_pdfs(container=container, blob_names=doc_ids, document_type=document_type,)
            self.logger.info("ingest_documents: container='%s' document_type='%s' ingested=%d (done)", container, doc_ids, ingested,)
            return ingested
        except Exception as e:
            self.logger.error("ingest_documents: container='%s' document_type='%s' doc_ids=%s -> failed: %s", container, doc_ids, document_type, e, exc_info=True)
            raise

    def reindex_document(self,
                         *,
                         container: str,
                         doc_id: str,
                         document_type: str = "IFU") -> int:
        self.logger.info("reindex_document: container='%s' doc_id='%s' document_type='%s' (start)", container, doc_id, document_type,)
        try:
            ingested = self.ingest_documents(container=container, doc_ids=[doc_id], document_type=document_type,)
            self.logger.info("reindex_document: container='%s' doc_id='%s' document_type='%s' ingested=%d (done)", container, doc_id, document_type, ingested,)
            return ingested
        except Exception as e:
            self.logger.error("reindex_document: container='%s' doc_id='%s' document_type='%s' -> failed: %s", container, doc_id, document_type, e, exc_info=True)
            raise

    def delete_document_vectors(self,
                                *,
                                doc_id: str) -> int:
        self.logger.info("delete_document_vectors: doc_id='%s' (start)", doc_id, )
        try:
            deleted = self.store.delete_by_doc_id(doc_id=doc_id)
            self.logger.info("delete_document_vectors: doc_id='%s' deleted=%d (done)", doc_id, deleted, )
            return deleted
        except Exception as e:
            self.logger.error("delete_document_vectors: doc_id='%s' -> failed: %s", doc_id, e, exc_info=True)
            raise


