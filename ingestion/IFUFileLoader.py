# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-08
# Description: IFUFileLoader
# -----------------------------------------------------------------------------
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.credentials import AzureNamedKeyCredential
from azure.core.exceptions import ResourceNotFoundError, AzureError, ResourceExistsError

from utility.logging_utils import get_class_logger
from config.Config import Config  # or Config if renamed


class IFUFileLoader:
    """
    Handles loading IFU documents from Azure Blob Storage.

    Provides:
      - list_documents(): lists available IFU blobs
      - load_document(): downloads a blob's bytes

    Logs detailed timing and error information.
    """

    def __init__(self, cfg: Config,*, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.blob_service: BlobServiceClient | None = None
        self.logger = logger or get_class_logger(self.__class__)

        start_time = time.time()

        try:
            self.blob_service = BlobServiceClient(
                account_url=f"https://{cfg.storage_account}.blob.core.windows.net",
                credential=AzureNamedKeyCredential(cfg.storage_account, cfg.storage_key),
            )
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.info(
                "Initialised BlobServiceClient for account '%s' (%.1f ms)",
                cfg.storage_account,
                elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.exception(
                "Failed to initialise BlobServiceClient after %.1f ms: %s", elapsed, e
            )
            raise

    def list_documents(self, container: str = "ifu_docs") -> List[str]:
        """
        Lists all documents in the given Azure Blob Storage container.
        """
        start_time = time.time()
        try:
            self.logger.info("Listing documents from container '%s'...", container)
            container_client = self.blob_service.get_container_client(container)
            blobs = [b.name for b in container_client.list_blobs()]
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.info(
                "Found %d document(s) in container '%s' (%.1f ms)",
                len(blobs),
                container,
                elapsed,
            )
            return blobs
        except ResourceNotFoundError:
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.error(
                "Container '%s' not found (%.1f ms)", container, elapsed
            )
            raise
        except AzureError as e:
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.exception(
                "Azure error while listing blobs in '%s' after %.1f ms: %s",
                container,
                elapsed,
                e,
            )
            raise
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.exception(
                "Unexpected error in list_documents after %.1f ms: %s", elapsed, e
            )
            raise

    def load_document(self, blob_name: str, container: str = "ifu_docs") -> bytes:
        """
        Downloads the specified blob from Azure Blob Storage and returns its bytes.
        """
        start_time = time.time()
        try:
            self.logger.info(
                "Loading document '%s' from container '%s'...", blob_name, container
            )
            blob_client = self.blob_service.get_blob_client(container, blob_name)
            data = blob_client.download_blob().readall()
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.info(
                "Loaded document '%s' (%d bytes) from '%s' (%.1f ms)",
                blob_name,
                len(data),
                container,
                elapsed,
            )
            return data
        except ResourceNotFoundError:
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.error(
                "Document '%s' not found in container '%s' (%.1f ms)",
                blob_name,
                container,
                elapsed,
            )
            raise
        except AzureError as e:
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.exception(
                "Azure error while downloading '%s' after %.1f ms: %s",
                blob_name,
                elapsed,
                e,
            )
            raise
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.exception(
                "Unexpected error in load_document after %.1f ms: %s", elapsed, e
            )
            raise

    def upload_document_from_path(
            self,
            local_path: str | Path,
            container: str = "ifudocs",
            blob_name: str | None = None,
    ) -> str:
        """
        Upload a local file to Azure Blob Storage.

        Args:
            local_path: Path to the local file on disk.
            container: Target container name in Blob Storage.
            blob_name: Optional name for the blob; if not provided,
                       the local file name is used.

        Returns:
            The blob name used for the uploaded file.
        """
        start_time = time.time()

        path_obj = Path(local_path)
        if not path_obj.is_file():
            self.logger.error("Local file not found: %s", path_obj)
            raise FileNotFoundError(f"Local file not found: {path_obj}")

        if blob_name is None:
            blob_name = path_obj.name

        file_size = path_obj.stat().st_size

        try:
            self.logger.info(
                "Uploading local file '%s' (%d bytes) to container '%s' as blob '%s'...",
                path_obj,
                file_size,
                container,
                blob_name,
            )

            container_client = self.blob_service.get_container_client(container)
            try:
                container_client.create_container()
                self.logger.info("Created container '%s' for upload.", container)
            except ResourceExistsError:
                self.logger.debug("Container '%s' already exists.", container)

            blob_client = container_client.get_blob_client(blob_name)

            with path_obj.open("rb") as f:
                metadata = {
                    "document_type": "IFU",
                    "source": "ingest",
                    "owner": "Blatchford QARA",
                }

                blob_client.upload_blob(
                    f,
                    overwrite=True,
                    metadata=metadata,
                    content_settings=ContentSettings(content_type="application/pdf"),
                )

            elapsed = (time.time() - start_time) * 1000.0
            self.logger.info(
                "Uploaded '%s' to '%s/%s' (%d bytes) in %.1f ms",
                path_obj,
                container,
                blob_name,
                file_size,
                elapsed,
            )

            return blob_name

        except AzureError as e:
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.exception(
                "Azure error while uploading '%s' to '%s/%s' after %.1f ms: %s",
                path_obj,
                container,
                blob_name,
                elapsed,
                e,
            )
            raise
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000.0
            self.logger.exception(
                "Unexpected error in upload_document_from_path after %.1f ms: %s",
                elapsed,
                e,
            )
            raise
    def upload_document_bytes(
            self,
            *,
            data: bytes,
            container: str,
            blob_name: str,
            content_type: str = "application/octet-stream",
            metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        if data is None:
            raise ValueError("data must not be None")

        container_client = self.blob_service.get_container_client(container)
        try:
            container_client.create_container()
            self.logger.info("Created container '%s' for upload.", container)
        except ResourceExistsError:
            pass

        blob_client = container_client.get_blob_client(blob_name)

        # IMPORTANT: metadata values must be str
        md = {str(k): str(v) for k, v in (metadata or {}).items()}

        blob_client.upload_blob(
            data,
            overwrite=True,
            metadata=md,
            content_settings=ContentSettings(content_type=content_type),
        )

        self.logger.info("Uploaded bytes -> '%s/%s' (%d bytes)", container, blob_name, len(data))
        return blob_name
