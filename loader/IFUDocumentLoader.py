# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-29
# Description: IFUDocumentLoader.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config.Config import Config
from ingestion.IFUFileLoader import IFUFileLoader
from utility.logging_utils import get_class_logger


class IFUDocumentLoader:
    """
    Thin document access layer (Blob Storage).
    """

    def __init__(
        self,
        cfg: Config,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or get_class_logger(self.__class__)
        self.file_loader = IFUFileLoader(cfg=self.cfg)
        self.logger.info("IFUDocumentLoader initialised successfully.")

    def upload_multiple_pdfs(
        self,
        pdf_paths: Iterable[str | Path],
        *,
        container: str,
        blob_prefix: str = "",
    ) -> Dict[Path, str]:
        results: Dict[Path, str] = {}

        for p in pdf_paths:
            path_obj = Path(p)

            if not path_obj.is_file():
                self.logger.error("Skipping missing file: %s", path_obj)
                continue

            blob_name = f"{blob_prefix}{path_obj.name}" if blob_prefix else path_obj.name

            try:
                uploaded = self.file_loader.upload_document_from_path(
                    local_path=str(path_obj),
                    container=container,
                    blob_name=blob_name,
                )
                results[path_obj] = uploaded
                self.logger.info("Uploaded '%s' as blob '%s'", path_obj, uploaded)
            except Exception as e:
                self.logger.error("Failed to upload '%s': %s", path_obj, e)

        return results

    def list_blob_names(self, *, container: str) -> List[str]:
        raw = self.file_loader.list_documents(container=container)
        if isinstance(raw, list) and raw:
            if all(isinstance(x, str) for x in raw):
                return raw
            if all(isinstance(x, dict) for x in raw):
                names: List[str] = []
                for item in raw:
                    name = item.get("blob_name") or item.get("name")
                    if isinstance(name, str):
                        names.append(name)
                return names
        return []

    def get_blob_details(self, *, container: str) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        container_client = self.file_loader.blob_service.get_container_client(container)

        self.logger.info("get_blob_details: container='%s' (start)", container)

        for blob in container_client.list_blobs(include=["metadata"]):
            blob_name = blob.name
            size = getattr(blob, "size", None)
            last_modified = getattr(blob, "last_modified", None)

            # --- metadata from list_blobs (may be None or {})
            metadata = getattr(blob, "metadata", None)
            self.logger.info(
                "list_blobs: blob='%s' metadata=%r",
                blob_name,
                metadata,
            )

            content_type = None

            try:
                blob_client = container_client.get_blob_client(blob_name)
                props = blob_client.get_blob_properties()

                # --- content type
                if getattr(props, "content_settings", None):
                    content_type = props.content_settings.content_type

                # --- authoritative last_modified / size
                if getattr(props, "last_modified", None):
                    last_modified = props.last_modified
                if getattr(props, "size", None) is not None:
                    size = props.size

                # --- metadata from blob properties (most reliable)
                props_metadata = getattr(props, "metadata", None)
                self.logger.info(
                    "get_blob_properties: blob='%s' metadata=%r",
                    blob_name,
                    props_metadata,
                )

                if isinstance(props_metadata, dict) and props_metadata:
                    # normalise to str -> str
                    metadata = {str(k): str(v) for k, v in props_metadata.items()}
                elif not metadata:
                    metadata = None

            except Exception as e:
                self.logger.warning(
                    "Failed to read blob properties for '%s': %s",
                    blob_name,
                    e,
                )

            self.logger.info(
                "final blob detail: blob='%s' blob_metadata=%r",
                blob_name,
                metadata,
            )

            details.append(
                {
                    "blob_name": blob_name,
                    "size": size,
                    "content_type": content_type,
                    "last_modified": last_modified.isoformat() if last_modified else None,
                    "blob_metadata": metadata or {},
                }
            )

        self.logger.info(
            "get_blob_details: container='%s' -> %d blobs (done)",
            container,
            len(details),
        )
        return details

    def load_blob_bytes(self, *, container: str, blob_name: str) -> bytes:
        pdf_bytes = self.file_loader.load_document(blob_name=blob_name, container=container)
        if not pdf_bytes:
            raise ValueError(f"No bytes returned for blob '{blob_name}' from '{container}'")
        return pdf_bytes

    def try_get_last_modified_iso(self, *, container: str, blob_name: str) -> Optional[str]:
        try:
            container_client = self.file_loader.blob_service.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)
            props = blob_client.get_blob_properties()
            if props and getattr(props, "last_modified", None):
                return props.last_modified.isoformat()
        except Exception:
            return None
        return None
