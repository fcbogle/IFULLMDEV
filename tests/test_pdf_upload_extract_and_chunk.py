# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-11
# Description: test_pdf_upload_download_and_chunk
# -----------------------------------------------------------------------------
import hashlib
import os
import re
import uuid
from collections import Counter
from pathlib import Path

import pytest
import pandas as pd

from config.Config import Config
from chunking.IFUChunker import IFUChunker
from ingestion.IFUFileLoader import IFUFileLoader
from chunking.LangDetectDetector import LangDetectDetector
from extractor.IFUTextExtractor import IFUTextExtractor


# ---------- Test prerequisites helpers ----------

# Default to your known path; allow override with IFU_LOCAL_TEST_PDF
DEFAULT_PDF = "/Users/frankbogle/Documents/ifu/BMK2IFU.pdf"
TEST_PDF_PATH = Path(os.getenv("IFU_LOCAL_TEST_PDF", DEFAULT_PDF))


def _missing_storage_env_vars() -> list[str]:
    """
    Check only the Azure Storage env vars needed for this test.

    We use Config.ENV_VARS to avoid duplicating env var names.
    """
    storage_env_names = [
        Config.ENV_VARS["storage_account"],  # e.g. "AZURE_STORAGE_ACCOUNT"
        Config.ENV_VARS["storage_key"],      # e.g. "AZURE_STORAGE_KEY"
    ]
    return [name for name in storage_env_names if not os.getenv(name)]


def _skip_if_missing_storage_or_pdf():
    """
    Skip this integration test if:
      - required Azure Storage env vars are not set
      - the local IFU PDF file is not present
    """
    missing_storage = _missing_storage_env_vars()
    if missing_storage:
        pytest.skip(
            f"Missing env vars for Azure Storage: {', '.join(missing_storage)}"
        )

    if not TEST_PDF_PATH.is_file():
        pytest.skip(f"Local IFU PDF not found: {TEST_PDF_PATH}")


# ---------- Integration test ----------


@pytest.mark.integration
def test_pdf_upload_extract_and_chunk():
    """
    End-to-end integration test:

    - Upload a local IFU PDF to Azure Blob Storage
    - Download it back and verify bytes (MD5)
    - Extract pages with IFUTextExtractor
    - Chunk with IFUChunker + LangDetectDetector
    - Save a CSV of chunk metadata and print some stats
    """
    _skip_if_missing_storage_or_pdf()

    # NOTE: with the current Config.__post_init__, *all* env vars used by Config
    # must be present (storage + OpenAI + Chroma). If that’s too strict, consider
    # relaxing __post_init__ or adding a cfg.validate_storage() method.
    cfg = Config.from_env()
    loader = IFUFileLoader(cfg)
    pdf_extractor = IFUTextExtractor()

    local_pdf_path = TEST_PDF_PATH
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

    tokenizer = lambda text: re.findall(r"\w+|\S", text)

    # Token counts per page
    token_counts = [len(tokenizer(p)) for p in pages]
    print(f"Pages: {len(pages)}")
    print(
        "Tokens per page → "
        f"min={min(token_counts)}, max={max(token_counts)}, "
        f"avg={sum(token_counts) / len(token_counts):.1f}"
    )
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

    # Use pandas to print data about chunk metadata
    df = pd.DataFrame(
        [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "lang": c.lang,
                "lang_conf": float(getattr(c, "lang_confidence", 0.0)),
                "char_start": c.char_start,
                "char_end": c.char_end,
                "version": getattr(c, "version", "Unknown"),
                "region": getattr(c, "region", "Unknown"),
                "text_preview": (
                    c.short_preview(100)
                    if hasattr(c, "short_preview")
                    else c.text[:100].replace("\n", " ") + "…"
                ),
            }
            for c in chunks
        ]
    )

    print("\nChunk DataFrame (first 10 rows):")
    print(df.head(10).to_string(index=False))

    # Useful roll-ups
    print("\nCounts by language:")
    print(df["lang"].value_counts().to_string())

    print("\nChunks per page (first 20):")
    print(df.groupby("page_start").size().head(20).to_string())

    # Optional: persist for offline inspection (ignored by git if added to .gitignore)
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"chunks_{doc_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved chunk summary → {csv_path}")

    # Basic stats / assertions
    num_chunks = len(chunks)
    print(f"Chunking complete: {num_chunks} chunks created")
    assert num_chunks > 0, "No chunks created from pages"

    print("Calculating language distribution for chunks...")
    lang_counts = Counter(c.lang for c in chunks)
    for lang, count in lang_counts.items():
        print(f"   • {lang}: {count} chunks")
    print("Language analysis complete")

