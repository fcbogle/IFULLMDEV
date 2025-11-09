# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-08
# Description: test_ifu_file_loader_integration.py
# -----------------------------------------------------------------------------
import os
import uuid
from pathlib import Path

import pytest

from ingestion.IFUFileLoader import IFUFileLoader
from config.Config import Config

@pytest.mark.integration
def test_list_documents_in_ifu_container():
    """
        Basic integration test:
        - Uses real AzureConfig.from_env()
        - Calls list_documents() on the IFU container
        - Asserts that we get back a list (can be empty).
    """
    cfg = Config.from_env()
    loader = IFUFileLoader(cfg)

    container = os.getenv("IFU_BLOB_CONTAINER", "ifudocs")

    docs = loader.list_documents(container=container)

    assert isinstance(docs, list)


@pytest.mark.integration
def test_upload_and_download_round_trip(tmp_path):
    """
    Integration test for upload_document_from_path + load_document:

    - Create a small temp file locally
    - Upload it to Blob Storage
    - List documents and ensure the blob name is present
    - Download it back and compare bytes
    """
    cfg = Config.from_env()
    loader = IFUFileLoader(cfg)

    container = "ifudocs"

    # Create a small local test file
    local_file = tmp_path / f"ifu_test_{uuid.uuid4().hex[:8]}.txt"
    content = b"IFU integration test content"
    local_file.write_bytes(content)

    # Upload to Blob
    blob_name = loader.upload_document_from_path(local_file, container=container)

    # Verify it's visible in list_documents()
    docs = loader.list_documents(container=container)
    assert blob_name in docs

    # Download back and compare
    downloaded = loader.load_document(blob_name, container=container)
    assert downloaded == content

    # Optional: clean up the test blob (best-effort)
    try:
        container_client = loader.blob_service.get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.delete_blob()
    except Exception:
        # Don't fail the test just because cleanup failed
        pass

@pytest.mark.integration
def test_upload_named_pdf_round_trip():
    """
    Integration test for uploading and downloading a named PDF IFU document.

    - Loads config from environment
    - Uploads a specified local PDF file to Azure Blob Storage
    - Downloads it back
    - Verifies byte-for-byte equality
    """

    cfg = Config.from_env()
    loader = IFUFileLoader(cfg)

    # Access file on local machine
    local_pdf_path = Path("/Users/frankbogle/Documents/ifu/BMK2IFU.pdf")

    assert local_pdf_path.is_file(), f"Local PDF not found: {local_pdf_path}"

    container = "ifudocs"
    blob_name = local_pdf_path.name
    file_size = local_pdf_path.stat().st_size

    # Upload to Blob Storage
    uploaded_blob_name = loader.upload_document_from_path(
        local_path=local_pdf_path,
        container=container,
        blob_name=blob_name,
    )
    assert uploaded_blob_name == blob_name

    # Download it back using load_pdf() if available, else load_document()
    if hasattr(loader, "load_pdf"):
        downloaded_bytes = loader.load_pdf(pdf_name=blob_name, container=container)
    else:
        downloaded_bytes = loader.load_document(blob_name=blob_name, container=container)

    # Verify content matches
    original_bytes = local_pdf_path.read_bytes()
    assert downloaded_bytes == original_bytes

    # Optional cleanup
    try:
        container_client = loader.blob_service.get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.delete_blob()
        loader.logger.info(
            "Deleted test blob '%s' from container '%s'", blob_name, container
        )
    except Exception as e:
        loader.logger.warning(
            "Cleanup failed for blob '%s' from container '%s': %s",
            blob_name,
            container,
            e,
        )


