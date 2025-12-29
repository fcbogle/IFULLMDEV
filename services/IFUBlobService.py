# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-29
# Description: IFUBlobService.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, BinaryIO

from azure.storage.blob import ContentSettings
from azure.core.exceptions import AzureError

from utility.logging_utils import get_class_logger
from ingestion.IFUFileLoader import IFUFileLoader


def _norm_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Azure blob metadata must be str->str, keys are case-insensitive but can be
    returned normalized depending on SDK/path. Metadata keys are normalized to
    lowercase and stringify values to keep the API consistent.
    """

    if not meta:
        return {}
    return {str(k).lower(): str(v) for k, v in meta.items() if v is not None}

@dataclass
class IFUBlobService:
    """
    Blob façade used by FastAPI (and by UI indirectly):
      - list/get blob details incl metadata
      - upload files (bytes/streams) with metadata
      - download bytes
      - exists check
    """
    file_loader: IFUFileLoader
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        self.logger = self.logger or get_class_logger(self.__class__)
        self.logger.info("IFUBlobService initialised (loader=%s)", type(self.file_loader).__name__)

    def list_blobs(self, *, container: str) -> List[Dict[str, Any]]:
        self.logger.info("list_blobs: container='%s' (start)", container)
        try:
            container_client = self.file_loader.blob_service.get_container_client(container)
            out: List[Dict[str, Any]] = []

            # include metadata in list call
            for blob in container_client.list_blobs(include=["metadata"]):
                md = _norm_meta(getattr(blob, "metadata", None))

                out.append(
                    {
                        "blob_name": blob.name,
                        "size": getattr(blob, "size", None),
                        "content_type": None,  # best effort via properties below
                        "last_modified": getattr(blob, "last_modified", None).isoformat()
                        if getattr(blob, "last_modified", None)
                        else None,
                        "blob_metadata": md or {},
                    }
                )

            # best effort: enrich content_type (and metadata/last_modified) via HEAD
            for item in out:
                name = item["blob_name"]
                try:
                    bc = container_client.get_blob_client(name)
                    props = bc.get_blob_properties()
                    if props and getattr(props, "content_settings", None):
                        item["content_type"] = props.content_settings.content_type
                    if props and getattr(props, "last_modified", None):
                        item["last_modified"] = props.last_modified.isoformat()
                    if props and getattr(props, "metadata", None):
                        item["blob_metadata"] = _norm_meta(props.metadata)
                except Exception as e:
                    self.logger.debug("list_blobs: properties failed for '%s': %s", name, e)

            self.logger.info("list_blobs: container='%s' -> %d blobs (done)", container, len(out))
            return out

        except Exception as e:
            self.logger.error("list_blobs: container='%s' -> failed: %s", container, e, exc_info=True)
            raise

    def get_blob(self, *, container: str, blob_name: str) -> Dict[str, Any]:
        self.logger.info("get_blob: container='%s' blob='%s' (start)", container, blob_name)
        try:
            bc = self.file_loader.blob_service.get_blob_client(container, blob_name)
            props = bc.get_blob_properties()

            md = _norm_meta(getattr(props, "metadata", None))
            content_type = None
            if props and getattr(props, "content_settings", None):
                content_type = props.content_settings.content_type

            out = {
                "blob_name": blob_name,
                "size": getattr(props, "size", None),
                "content_type": content_type,
                "last_modified": getattr(props, "last_modified", None).isoformat()
                if getattr(props, "last_modified", None)
                else None,
                "blob_metadata": md or {},
            }

            self.logger.info("get_blob: container='%s' blob='%s' (done)", container, blob_name)
            return out

        except Exception as e:
            self.logger.error(
                "get_blob: container='%s' blob='%s' -> failed: %s",
                container, blob_name, e, exc_info=True
            )
            raise

    def blob_exists(self, *, container: str, blob_name: str) -> bool:
        self.logger.info("blob_exists: container='%s' blob='%s' (start)", container, blob_name)
        try:
            bc = self.file_loader.blob_service.get_blob_client(container, blob_name)
            exists = bc.exists()
            self.logger.info("blob_exists: container='%s' blob='%s' exists=%s (done)", container, blob_name, exists)
            return bool(exists)
        except Exception as e:
            self.logger.error(
                "blob_exists: container='%s' blob='%s' -> failed: %s",
                container, blob_name, e, exc_info=True
            )
            raise

    def download_blob_bytes(self, *, container: str, blob_name: str) -> bytes:
        # Reuse your existing loader method
        self.logger.info("download_blob_bytes: container='%s' blob='%s' (start)", container, blob_name)
        data = self.file_loader.load_document(blob_name=blob_name, container=container)
        self.logger.info("download_blob_bytes: container='%s' blob='%s' bytes=%d (done)", container, blob_name, len(data))
        return data

    def upload_blob_stream(
        self,
        *,
        container: str,
        blob_name: str,
        stream: BinaryIO,
        content_type: str = "application/pdf",
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """
        Upload from an open stream (UploadFile.file works here).
        """
        self.logger.info(
            "upload_blob_stream: container='%s' blob='%s' content_type='%s' (start)",
            container, blob_name, content_type
        )

        md = _norm_meta(metadata)

        try:
            container_client = self.file_loader.blob_service.get_container_client(container)
            # create container if missing
            try:
                container_client.create_container()
                self.logger.info("upload_blob_stream: created container '%s'", container)
            except Exception:
                pass

            bc = container_client.get_blob_client(blob_name)
            bc.upload_blob(
                stream,
                overwrite=overwrite,
                metadata=md or None,
                content_settings=ContentSettings(content_type=content_type),
            )

            # return fresh properties
            out = self.get_blob(container=container, blob_name=blob_name)
            self.logger.info("upload_blob_stream: container='%s' blob='%s' (done)", container, blob_name)
            return out

        except AzureError as e:
            self.logger.error(
                "upload_blob_stream: container='%s' blob='%s' -> AzureError: %s",
                container, blob_name, e, exc_info=True
            )
            raise
        except Exception as e:
            self.logger.error(
                "upload_blob_stream: container='%s' blob='%s' -> failed: %s",
                container, blob_name, e, exc_info=True
            )
            raise