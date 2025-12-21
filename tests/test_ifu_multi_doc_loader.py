# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Updated: 2025-12-21
# Description: test_ifu_ingest_pipeline.py
# -----------------------------------------------------------------------------

import os
from pathlib import Path
from typing import List

import pytest
import logging

from config.Config import Config

# Thin blob layer
from loader.IFUDocumentLoader import IFUDocumentLoader

# Ingestion pipeline components
from extractor.IFUTextExtractor import IFUTextExtractor
from chunking.IFUChunker import IFUChunker
from chunking.LangDetectDetector import LangDetectDetector
from embedding.IFUEmbedder import IFUEmbedder
from vectorstore.ChromaIFUVectorStore import ChromaIFUVectorStore

# Your new service (adjust import to your actual module path)
from services.IFUIngestService import IFUIngestService

pytestmark = pytest.mark.integration
test_logger = logging.getLogger("test_logger")


def _build_cfg_or_skip() -> Config:
    """
    Build Config from env vars. Skip if required env vars are missing.
    Prefer using your Config.from_env() if you have it.
    """
    try:
        # If you already have Config.from_env(), use that:
        if hasattr(Config, "from_env"):
            cfg = Config.from_env()
        else:
            cfg = Config(
                ifu_sample_pdf=os.getenv("IFU_SAMPLE_PDF"),
                ifu_sample_folder=os.getenv("IFU_SAMPLE_FOLDER"),
                storage_account=os.getenv("AZURE_STORAGE_ACCOUNT"),
                storage_key=os.getenv("AZURE_STORAGE_KEY"),
                openai_base_url=os.getenv("OPENAI_BASE_URL"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_chat_model=os.getenv("OPENAI_CHAT_MODEL"),
                openai_azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                openai_azure_embed_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
                chroma_endpoint=os.getenv("CHROMA_ENDPOINT"),
                chroma_api_key=os.getenv("CHROMA_API_KEY"),
                chroma_tenant=os.getenv("CHROMA_TENANT"),
                chroma_database=os.getenv("CHROMA_DATABASE"),
            )
    except (TypeError, ValueError) as e:
        pytest.skip(f"Config could not be initialised for integration tests: {e}")

    # Minimal sanity: storage + chroma + embedding must exist for end-to-end ingest
    required = [
        ("AZURE_STORAGE_ACCOUNT", getattr(cfg, "storage_account", None)),
        ("AZURE_STORAGE_KEY", getattr(cfg, "storage_key", None)),
        ("CHROMA_TENANT", getattr(cfg, "chroma_tenant", None)),
        ("CHROMA_DATABASE", getattr(cfg, "chroma_database", None)),
        ("CHROMA_API_KEY", getattr(cfg, "chroma_api_key", None)),
        ("AZURE_OPENAI_EMBED_DEPLOYMENT", getattr(cfg, "openai_azure_embed_deployment", None)),
        ("AZURE_OPENAI_ENDPOINT", getattr(cfg, "openai_azure_endpoint", None)),
        ("AZURE_OPENAI_API_KEY", getattr(cfg, "openai_azure_api_key", None)),
    ]
    missing = [name for name, val in required if not val]
    if missing:
        pytest.skip(f"Missing required env/config values: {missing}")

    return cfg


def _get_sample_pdf_files(cfg: Config) -> list[Path]:
    folder = getattr(cfg, "ifu_sample_folder", None)
    sample_pdf = getattr(cfg, "ifu_sample_pdf", None)

    # Prefer folder, fallback to single file
    if folder:
        path = Path(folder)
        if path.is_file():
            return [path]
        if path.is_dir():
            pdfs = sorted(path.glob("*.pdf"))
            if not pdfs:
                pytest.skip("No PDFs found in IFU_SAMPLE_FOLDER")
            return pdfs

    if sample_pdf:
        path = Path(sample_pdf)
        if path.is_file():
            return [path]

    pytest.skip("Neither IFU_SAMPLE_FOLDER nor IFU_SAMPLE_PDF is configured to a valid path.")


def _build_chunker() -> IFUChunker:
    # Keep this aligned with your production tokenizer/chunker logic
    import re
    tokenizer = lambda text: re.findall(r"\w+|\S", text)
    lang_detector = LangDetectDetector()

    return IFUChunker(
        tokenizer=tokenizer,
        lang_detector=lang_detector,
        chunk_size_tokens=300,
        overlap_tokens=100,
    )


@pytest.mark.integration
def test_ifu_document_loader_initialises_and_lists_blob_details():
    cfg = _build_cfg_or_skip()

    doc_loader = IFUDocumentLoader(cfg=cfg)

    assert doc_loader.cfg is cfg
    assert hasattr(doc_loader, "file_loader")
    assert doc_loader.file_loader is not None

    # If the test container exists, metadata should be obtainable.
    # (If not, we still allow the test to pass without hard failing.)
    container = "ifu-docs-integration-test"
    try:
        details = doc_loader.get_blob_details(container=container)
    except Exception as e:
        pytest.skip(f"Blob container '{container}' not accessible: {e!r}")

    assert isinstance(details, list)

    # If there are blobs, last_modified should not be None.
    # content_type may be None unless you fetch per-blob properties (that’s OK).
    if details:
        first = details[0]
        assert "blob_name" in first
        assert "last_modified" in first
        # size/last_modified are the key metadata your /stats cares about
        assert first["last_modified"] is not None


@pytest.mark.integration
def test_upload_then_ingest_pipeline_via_services():
    """
    End-to-end integration:

      - Upload PDFs to Azure Blob via IFUDocumentLoader (thin layer)
      - Ingest those blobs via IFUIngestService (extract -> chunk -> embed -> upsert)
      - Sanity-check Chroma now contains data
      - Verify blob metadata can be read (size + last_modified) for /stats
    """
    cfg = _build_cfg_or_skip()
    sample_pdfs = _get_sample_pdf_files(cfg)

    container = "ifu-docs-integration-test"
    collection_name = "ifu-docs-integration-test"

    # --- Thin blob loader ---
    doc_loader = IFUDocumentLoader(cfg=cfg)

    # --- Core pipeline components ---
    chunker = _build_chunker()
    extractor = IFUTextExtractor()
    embedder = IFUEmbedder(
        cfg=cfg,
        batch_size=16,
        normalize=True,
        out_dtype="float32",
        filter_lang=None,
    )
    store = ChromaIFUVectorStore(
        cfg=cfg,
        embedder=embedder,
        collection_name=collection_name,
    )

    # --- Ingest service (constructor should NOT take cfg if you refactored that out) ---
    ingest_svc = IFUIngestService(
        document_loader=doc_loader,
        extractor=extractor,
        chunker=chunker,
        embedder=embedder,
        store=store,
        collection_name=collection_name,
    )

    # ---- Act: upload PDFs ----
    results = doc_loader.upload_multiple_pdfs(
        pdf_paths=sample_pdfs,
        container=container,
        blob_prefix="",
    )

    assert set(results.keys()) == set(sample_pdfs)
    blob_names = list(results.values())
    assert blob_names, "Expected at least one uploaded blob name"

    # ---- Assert: blob details include size + last_modified (fixes your /stats concern) ----
    details = doc_loader.get_blob_details(container=container)
    details_by_name = {d["blob_name"]: d for d in details if isinstance(d, dict) and d.get("blob_name")}

    # Ensure each uploaded blob is present and has last_modified
    for bn in blob_names:
        assert bn in details_by_name, f"Uploaded blob '{bn}' not found in get_blob_details()"
        assert details_by_name[bn].get("last_modified") is not None, f"Blob '{bn}' missing last_modified"
        # size often present from list_blobs; keep as soft assertion if your SDK doesn’t include it
        assert details_by_name[bn].get("size") is not None, f"Blob '{bn}' missing size"

    # ---- Act: ingest uploaded blobs ----
    ingested_count = ingest_svc.ingest_blob_pdfs(blob_names=blob_names, container=container)

    # ---- Assert: all blobs ingested ----
    assert ingested_count == len(blob_names)

    # ---- Assert: Chroma has chunks ----
    try:
        total_chunks = store.collection.count()
    except Exception as e:
        pytest.fail(f"Could not count Chroma collection after ingestion: {e!r}")

    assert total_chunks > 0, "Expected >0 chunks in Chroma after ingestion"

    # ---- Optional: semantic query sanity check ----
    try:
        res = store.query_text("Blatchford", n_results=5)
    except Exception as e:
        pytest.skip(f"Chroma query_text failed: {e!r}")

    assert isinstance(res, dict)
    assert "ids" in res
    assert isinstance(res["ids"], list)

