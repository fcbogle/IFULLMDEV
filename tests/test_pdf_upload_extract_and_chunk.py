# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-11
# Description: test_pdf_upload_download_and_chunk
# -----------------------------------------------------------------------------
import hashlib
import os
import uuid
from collections import Counter
from pathlib import Path

import pytest

from config.Config import Config
from chunking.IFUChunker import IFUChunker
from ingestion.IFUFileLoader import IFUFileLoader
from chunking.LangDetectDetector import LangDetectDetector
from extractor.IFUTextExtractor import IFUTextExtractor


@pytest.mark.integration
def test_pdf_upload_extract_and_chunk():
    """
        Basic integration test:
        - Uses real AzureConfig.from_env()
        - Calls list_documents() on the IFU container
        - Asserts that we get back a list (can be empty).
    """
    cfg = Config.from_env()
    loader = IFUFileLoader(cfg)
    # pdf_extractor = PDFPageExtractor()
    pdf_extractor = IFUTextExtractor()

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

    # Extract text pages from PDF using utility class
    pages = pdf_extractor.extract_text_from_pdf(downloaded_bytes)
    assert len(pages) > 1, "Problem with number of pages extracted i.e., < 1"
    print(f"Extracted {len(pages)} pages of text from PDF")

    total_chars = sum(len(p) for p in pages)
    print(f"Total characters across all pages: {total_chars}")

    tokenizer = lambda s: s.split()

    # Token counts per page
    token_counts = [len(tokenizer(p)) for p in pages]
    print(f"Pages: {len(pages)}")
    print(f"Tokens per page → min={min(token_counts)}, max={max(token_counts)}, "
          f"avg={sum(token_counts) / len(token_counts):.1f}")
    print("First 10 token counts:", token_counts[:10])

    # Chunking pages into language-aware IFUChunks
    lang_detector = LangDetectDetector()
    chunker = IFUChunker(
        tokenizer=tokenizer,
        lang_detector=lang_detector,
        chunk_size_tokens=300,
        overlap_tokens=100,
    )

    hash_short = hashlib.md5(downloaded_bytes).hexdigest()[:8]
    doc_id = f"{local_pdf_path.stem}_{hash_short}"  # e.g. BMK2IFU_9f84c7a2
    doc_name = local_pdf_path.name  # or use blob_name if you prefer
    doc_metadata = {
        "version": "Unknown",
        "region": "Unknown",
        "is_primary_language": True,
    }

    chunks = chunker.chunk_document(
        doc_id=doc_id,
        doc_name=doc_name,
        pages=pages,
        doc_metadata=doc_metadata,
    )

    num_chunks = len(chunks)
    print(f"Chunking complete: {num_chunks} chunks created")
    assert num_chunks > 0, "No chunks created from pages"

    # --- 5. Language distribution ---
    print("Calculating language distribution for chunks...")
    lang_counts = Counter(c.lang for c in chunks)
    for lang, count in lang_counts.items():
        print(f"   • {lang}: {count} chunks")
    print("Language analysis complete")



