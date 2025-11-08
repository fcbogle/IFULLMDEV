# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-08
# Description: IFUFileLoader
# -----------------------------------------------------------------------------

import logging
import time
from typing import List
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential
from azure.core.exceptions import ResourceNotFoundError, AzureError

from utility.logging_utils import get_class_logger
from config import Config  # or Config if renamed


class IFUFileLoader:
    """
    Handles loading IFU documents from Azure Blob Storage.

    Provides:
      - list_documents(): lists available IFU blobs
      - load_document(): downloads a blob's bytes

    Logs detailed timing and error information.
    """

    def __init__(self, cfg: Config, logger: logging.Logger | None = None):
        self.cfg = cfg
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
