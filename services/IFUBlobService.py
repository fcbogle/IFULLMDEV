# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-29
# Description: services/IFUBlobService.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from azure.core.exceptions import AzureError, ResourceNotFoundError
from fastapi import UploadFile

from ingestion.IFUFileLoader import IFUFileLoader
from utility.logging_utils import get_class_logger

REQUIRED_BLOB_META_KEYS = ["document_type"]

from settings import BLOB_CONTAINER_DEFAULT


def _norm_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Azure Blob metadata must be a dict[str, str].
    Normalising here avoids surprises (None values, non-strings, inconsistent casing).
    """
    if not meta:
        return {}
    return {str(k).lower(): str(v) for k, v in meta.items() if v is not None}


@dataclass
class IFUBlobService:
    """
    Blob-management façade (list/get/upload/delete/metadata).
    Re-uses IFUFileLoader for authenticated BlobServiceClient.
    """
    file_loader: IFUFileLoader
    logger: Any = None

    def __post_init__(self) -> None:
        self.logger = self.logger or get_class_logger(self.__class__)

    def _resolve_container(self, container: Optional[str]) -> str:
        return (container or BLOB_CONTAINER_DEFAULT).strip()

    def list_blobs(self, *, container: str, prefix: str = "") -> List[Dict[str, Any]]:
        container = self._resolve_container(container)
        self.logger.info("list_blobs: container='%s' prefix='%s' (start)", container, prefix)
        try:
            cc = self.file_loader.blob_service.get_container_client(container)

            out: List[Dict[str, Any]] = []
            for b in cc.list_blobs(name_starts_with=prefix, include=["metadata"]):
                name = b.name
                size = getattr(b, "size", None)
                last_modified = getattr(b, "last_modified", None)
                metadata = getattr(b, "metadata", None) or {}

                out.append(
                    {
                        "blob_name": name,
                        "size": size,
                        "content_type": None,  # fill via props below
                        "last_modified": last_modified.isoformat() if last_modified else None,
                        "blob_metadata": metadata,
                    }
                )

            # Optional: enrich content_type via HEAD (slower but accurate)
            for row in out:
                blob_name = row["blob_name"]
                try:
                    props = cc.get_blob_client(blob_name).get_blob_properties()
                    if props and getattr(props, "content_settings", None):
                        row["content_type"] = props.content_settings.content_type
                    if props and getattr(props, "metadata", None):
                        row["blob_metadata"] = props.metadata or row["blob_metadata"]
                except Exception:
                    pass

            self.logger.info("list_blobs: container='%s' -> %d blobs (done)", container, len(out))
            return out

        except AzureError as e:
            self.logger.exception("list_blobs failed: container='%s': %s", container, e)
            raise

    def get_blob_details(self, *, container: str, prefix: str = "") -> List[Dict[str, Any]]:
        """
        Returns list of {blob_name, size, content_type, last_modified, blob_metadata}.
        """
        container = self._resolve_container(container)
        self.logger.info("get_blob_details: container='%s' prefix='%s' (start)", container, prefix)
        out: List[Dict[str, Any]] = []
        cc = self.file_loader.blob_service.get_container_client(container)

        try:
            for blob in cc.list_blobs(name_starts_with=prefix, include=["metadata"]):
                blob_name = blob.name
                size = getattr(blob, "size", None)
                last_modified = getattr(blob, "last_modified", None)
                meta = getattr(blob, "metadata", None)

                content_type = None
                try:
                    bc = cc.get_blob_client(blob_name)
                    props = bc.get_blob_properties()
                    if props and getattr(props, "content_settings", None):
                        content_type = props.content_settings.content_type
                    if props and getattr(props, "last_modified", None):
                        last_modified = props.last_modified
                    if props and getattr(props, "size", None) is not None:
                        size = props.size
                    if props and getattr(props, "metadata", None) is not None:
                        meta = props.metadata
                except Exception:
                    # best-effort only
                    pass

                norm = _norm_meta(meta)
                self.logger.info("get_blob_details: blob='%s' blob_metadata=%s", blob_name, norm or "{}")

                out.append(
                    {
                        "blob_name": blob_name,
                        "size": size,
                        "content_type": content_type,
                        "last_modified": last_modified.isoformat() if last_modified else None,
                        "blob_metadata": norm or {},
                    }
                )

            self.logger.info("get_blob_details: container='%s' -> %d blobs (done)", container, len(out))
            return out
        except AzureError as e:
            self.logger.exception("get_blob_details failed: container='%s': %s", container, e)
            raise

    def get_blob_metadata(self, *, container: str, blob_name: str) -> Dict[str, str]:
        container = self._resolve_container(container)
        self.logger.info("get_blob_metadata: container='%s' blob='%s' (start)", container, blob_name)
        try:
            bc = self.file_loader.blob_service.get_blob_client(container, blob_name)
            props = bc.get_blob_properties()
            meta = _norm_meta(getattr(props, "metadata", None))
            self.logger.info("get_blob_metadata: container='%s' blob='%s' keys=%d (done)", container, blob_name, len(meta))
            return meta
        except ResourceNotFoundError:
            self.logger.warning("get_blob_metadata: not found container='%s' blob='%s'", container, blob_name)
            raise
        except AzureError as e:
            self.logger.exception("get_blob_metadata failed: container='%s' blob='%s': %s", container, blob_name, e)
            raise

    def set_blob_metadata(self, *, container: str, blob_name: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Overwrites metadata (Azure semantics). If you want merge semantics, do it in the router/service.
        """
        container = self._resolve_container(container)
        norm = _norm_meta(metadata)
        self.logger.info(
            "set_blob_metadata: container='%s' blob='%s' keys=%d (start)",
            container, blob_name, len(norm)
        )
        try:
            bc = self.file_loader.blob_service.get_blob_client(container, blob_name)
            bc.set_blob_metadata(metadata=norm)
            self.logger.info("set_blob_metadata: container='%s' blob='%s' (done)", container, blob_name)
            return norm
        except ResourceNotFoundError:
            self.logger.warning("set_blob_metadata: not found container='%s' blob='%s'", container, blob_name)
            raise
        except AzureError as e:
            self.logger.exception("set_blob_metadata failed: container='%s' blob='%s': %s", container, blob_name, e)
            raise

    def delete_blob(self, *, container: str, blob_name: str) -> bool:
        container = self._resolve_container(container)
        self.logger.info("delete_blob: container='%s' blob='%s' (start)", container, blob_name)
        try:
            bc = self.file_loader.blob_service.get_blob_client(container, blob_name)
            bc.delete_blob()
            self.logger.info("delete_blob: container='%s' blob='%s' deleted (done)", container, blob_name)
            return True
        except ResourceNotFoundError:
            self.logger.info("delete_blob: container='%s' blob='%s' not found -> False", container, blob_name)
            return False
        except AzureError as e:
            self.logger.exception("delete_blob failed: container='%s' blob='%s': %s", container, blob_name, e)
            raise

    async def upload_files(
            self,
            *,
            container: str,
            blob_prefix: str = "",
            files: List[UploadFile],
            default_metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        container = self._resolve_container(container)
        self.logger.info(
            "upload_files: container='%s' prefix='%s' files=%d (start)",
            container, blob_prefix, len(files)
        )

        results: List[Dict[str, Any]] = []
        uploaded = 0
        errors = 0
        rejected = 0

        for f in files:
            filename = (f.filename or "").strip() or "upload.bin"
            blob_name = f"{blob_prefix}{filename}" if blob_prefix else filename

            try:
                ext = ""
                if "." in filename:
                    allowed_extensions = {".pdf"}  # later: {".pdf", ".txt"}

                    ext = ""
                    if "." in filename:
                        ext = "." + filename.lower().rsplit(".", 1)[-1]

                    if ext not in allowed_extensions:
                        rejected += 1
                        msg = (
                            f"Rejected '{filename}'. "
                            f"Unsupported file type '{ext or '(none)'}'. "
                            f"Allowed types: {', '.join(sorted(allowed_extensions))}"
                        )
                        self.logger.warning("upload_files: %s", msg)
                        results.append({
                            "filename": filename,
                            "blob_name": blob_name,
                            "ok": False,
                            "error": msg,
                        })
                        continue

                data = await f.read()
                size = len(data) if data else 0

                raw_ct = (f.content_type or "").strip().lower()  # from client/multipart
                guessed_ct, _ = mimetypes.guess_type(filename)
                guessed_ct = (guessed_ct or "").strip().lower()

                # --- choose final content type ---
                is_pdf_name = filename.lower().endswith(".pdf")

                # raw_ct is often wrong; override when it contradicts filename/guess
                if is_pdf_name:
                    # force pdf unless client gave something better than empty/generic
                    final_ct = raw_ct or "application/pdf"
                    if final_ct in ("application/octet-stream", "binary/octet-stream", ""):
                        final_ct = "application/pdf"
                else:
                    # not a pdf filename -> do NOT trust raw_ct if it's pdf or generic
                    if raw_ct in ("application/pdf", "application/octet-stream", "binary/octet-stream", ""):
                        final_ct = guessed_ct or "application/octet-stream"
                    else:
                        # client provided something non-generic; keep it
                        final_ct = raw_ct

                metadata = dict(default_metadata or {})
                metadata.setdefault("source", "ui_upload")
                metadata.setdefault("filename", filename)
                metadata.setdefault("document_type", "IFU")

                # optional: keep a trace of what the client claimed
                if raw_ct:
                    metadata.setdefault("client_content_type", raw_ct)

                self.logger.info(
                    "upload_files: uploading filename='%s' -> blob='%s' bytes=%d raw_ct='%s' guessed_ct='%s' final_ct='%s'",
                    filename, blob_name, size, raw_ct, guessed_ct, final_ct
                )

                self.file_loader.upload_document_bytes(
                    data=data,
                    container=container,
                    blob_name=blob_name,
                    content_type=final_ct,
                    metadata=metadata,
                )

                uploaded += 1
                results.append({
                    "filename": filename,
                    "blob_name": blob_name,
                    "bytes": size,
                    "ok": True,
                    "content_type": final_ct,
                })

            except Exception as e:
                errors += 1
                self.logger.exception("upload_files: failed filename='%s' blob='%s': %s", filename, blob_name, e)
                results.append({"filename": filename, "blob_name": blob_name, "ok": False, "error": str(e)})

            finally:
                try:
                    await f.close()
                except Exception:
                    pass

        out = {
            "container": container,
            "blob_prefix": blob_prefix,
            "uploaded": uploaded,
            "rejected": rejected,
            "errors": errors,
            "results": results,
        }
        self.logger.info("upload_files: container='%s' uploaded=%d errors=%d (done)", container, uploaded, errors)
        return out

