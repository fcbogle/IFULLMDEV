# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-09
# Description: test_ifu_file_extractor_integration.py
# -----------------------------------------------------------------------------
import hashlib
import os
import uuid
import fitz
from pathlib import Path

import pytest

from Config import Config
from IFUFileLoader import IFUFileLoader
from IFUTextExtractor import IFUTextExtractor
from utility.logging_utils import get_class_logger


@pytest.mark.integration
def test_text_reading_named_pdf():
    """
        End-to-end integration test:

        - Upload a local IFU PDF from path to Azure Blob Storage
        - Load the PDF back from Blob
        - Extract text using IFUTextExtractor
        - Compare MD5 hash of source PDF and blob contents
        - Delete the test blob from Blob Storage
    """
    cfg = Config.from_env()
    loader = IFUFileLoader(cfg)
    extractor = IFUTextExtractor()

    # Upload pdf from local path
    pdf_path = "/Users/frankbogle/Documents/ifu/BMK2IFU.pdf"
    local_pdf_path = Path(os.getenv("IFU_LOCAL_TEST_PDF", pdf_path))
    assert local_pdf_path.is_file(), f"Local test PDF not found: {local_pdf_path}"

    container = os.getenv("IFU_BLOB_CONTAINER", "ifudocs")

    # Create a unique blob name for the test
    unique_suffix = uuid.uuid4().hex[:8]
    blob_name = f"{local_pdf_path.stem}_{unique_suffix}.pdf"
    file_size = local_pdf_path.stat().st_size

    print(f"\nStarting IFUTextExtractor test for {local_pdf_path.name} ({file_size} bytes)")
    print(f"Target container: {container}, Blob name: {blob_name}")

    # Upload the local pdf to Blob Storage
    uploaded_blob_name = loader.upload_document_from_path(
        local_path=local_pdf_path,
        container=container,
        blob_name=blob_name,
    )
    assert uploaded_blob_name == blob_name
    print(f"Uploaded blob: {uploaded_blob_name}")

    # Load pdf bytes back from the blob
    downloaded_bytes = loader.load_document(
        blob_name=uploaded_blob_name,
        container=container,
    )

    assert isinstance(downloaded_bytes, (bytes, bytearray))
    assert len(downloaded_bytes) > 0
    print(f"Downloaded blob ({len(downloaded_bytes)} bytes)")

    # Compute MD5 hashes of source PDF and downloaded blob and compare
    local_hash = hashlib.md5(local_pdf_path.read_bytes()).hexdigest()
    blob_hash = hashlib.md5(downloaded_bytes).hexdigest()

    print(f"Local MD5: {local_hash}")
    print(f"Blob  MD5: {blob_hash}")

    assert local_hash == blob_hash, "Uploaded and downloaded PDFs differ in content!"
    print("PDF byte-for-byte match verified via MD5 hash comparison")

    # Extract text using IFUTextExtractor
    text = extractor.extract_text_from_pdf(downloaded_bytes)
    assert isinstance(text, str)
    assert len(text.strip()) > 0

    # Open the PDF again (from bytes) to count pages and first 200 chars of each page
    with fitz.open(stream=downloaded_bytes, filetype="pdf") as doc:
        print(f"\nPDF has {doc.page_count} pages. Showing first 200 characters of each page:\n")
        for i, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            preview = page_text[:200].replace("\n", " ")
            print(f"--- Page {i} ---")
            print(preview)
            print("\n")

    # Delete the downloaded blob from blob storage
    try:
        container_client = loader.blob_service.get_container_client(container)
        blob_client = container_client.get_blob_client(uploaded_blob_name)
        blob_client.delete_blob()
        print(f"Deleted test blob '{uploaded_blob_name}' from container '{container}'")
    except Exception as e:
        print(f"Cleanup failed for blob '{uploaded_blob_name}': {e}")

