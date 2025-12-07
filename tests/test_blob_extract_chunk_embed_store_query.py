# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-16
# Description: test_upload_download_extract_chunk_embed_store_query
# -----------------------------------------------------------------------------
import os
import uuid
from typing import List

import pytest
from pathlib import Path

from ingestion.IFUFileLoader import IFUFileLoader
from extractor.IFUTextExtractor import IFUTextExtractor
from chunking.IFUChunker import IFUChunker
from embedding.IFUEmbedder import IFUEmbedder
from vectorstore.IFUVectorStore import IFUVectorStore
from vectorstore.ChromaIFUVectorStore import ChromaIFUVectorStore
from chunking.LangDetectDetector import LangDetectDetector
from config.Config import Config

# Default PDF; can be overridden with IFU_LOCAL_TEST_PDF
DEFAULT_PDF = "/Users/frankbogle/Documents/ifu/BMK2IFU.pdf"
TEST_PDF_PATH = Path(os.getenv("IFU_LOCAL_TEST_PDF", DEFAULT_PDF))

def _missing_azure_openai_env_vars() -> list[str]:
    """
    Check only the Azure OpenAI env vars needed for embeddings.
    We derive the env var names from Config.ENV_VARS to avoid duplication.
    """
    azure_env_names = [
        # Azure OpenAI
        Config.ENV_VARS["openai_azure_api_key"],
        Config.ENV_VARS["openai_azure_endpoint"],
        Config.ENV_VARS["openai_azure_embed_deployment"],

        # Azure Blob Storage
        Config.ENV_VARS["storage_account"],
        Config.ENV_VARS["storage_key"],

        # Chroma
        Config.ENV_VARS["chroma_endpoint"],
        Config.ENV_VARS["chroma_api_key"],
        Config.ENV_VARS["chroma_tenant"],
        Config.ENV_VARS["chroma_database"],
    ]

    return [name for name in azure_env_names if not os.getenv(name)]

def _skip_if_missing_prereqs():
    """
    Skip integration test if:
      - the local IFU PDF is not present
      - required Azure OpenAI / storage env vars are not set
      - Chroma endpoint is not configured
    """
    if not TEST_PDF_PATH.is_file():
        pytest.skip(f"Local IFU PDF not found: {TEST_PDF_PATH}")

    missing = _missing_azure_openai_env_vars()
    if missing:
        pytest.skip(
            f"Missing env vars for Azure OpenAI / Storage: {', '.join(missing)}"
        )

    # Chroma-specific env var check (optional but helpful)
    chroma_endpoint_var = Config.ENV_VARS.get("chroma_endpoint", "CHROMA_ENDPOINT")
    if not os.getenv(chroma_endpoint_var):
        pytest.skip(f"Missing env var for Chroma endpoint: {chroma_endpoint_var}")

@pytest.mark.integration
def test_chroma_end_to_end_via_blob():
    """
        End-to-end integration:

          1. Upload local IFU PDF to Azure Blob (IFUFileLoader)
          2. Download PDF bytes back from Blob
          3. Extract pages (IFUTextExtractor)
          4. Chunk (IFUChunker)
          5. Embed (IFUEmbedder)
          6. Upsert into Chroma (ChromaIFUVectorStore via IFUVectorStore Protocol)
          7. Semantic query
          8. Cleanup: delete doc_id from Chroma + delete blob
    """

    _skip_if_missing_prereqs()

    cfg = Config.from_env()

    # Blob upload and download via IFULoader
    loader = IFUFileLoader(cfg)

    container = os.getenv("IFU_CONTAINER", "ifudocs")
    assert TEST_PDF_PATH.is_file(), f"Local PDF not found: {TEST_PDF_PATH}"

    # Give the blob a unique name so repeated test runs don't collide
    unique_suffix = uuid.uuid4().hex[:8]
    blob_name = f"{TEST_PDF_PATH.stem}_{unique_suffix}.pdf"

    uploaded_blob_name = loader.upload_document_from_path(
        local_path=TEST_PDF_PATH,
        container=container,
        blob_name=blob_name,
    )
    assert uploaded_blob_name == blob_name

    # Download bytes back from Blob
    downloaded_bytes = loader.load_document(
        blob_name=uploaded_blob_name,
        container=container,
    )
    assert isinstance(downloaded_bytes, (bytes, bytearray))
    assert len(downloaded_bytes) > 0

    # Extract pages from PDF bytes
    extractor = IFUTextExtractor()
    pages: List[str] = extractor.extract_text_from_pdf(downloaded_bytes)

    assert isinstance(pages, list)
    assert len(pages) > 0, "No pages extracted from IFU PDF"
    assert any(p.strip() for p in pages), "All extracted pages are empty!"

    # Chunk extracted pages into IFUChunks
    tokenizer = lambda s: s.split()
    lang_detector = LangDetectDetector()
    chunker = IFUChunker(
        tokenizer=tokenizer,
        lang_detector=lang_detector,
        chunk_size_tokens=300,
        overlap_tokens=100,
    )

    doc_id = TEST_PDF_PATH.stem
    doc_name = TEST_PDF_PATH.name
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
    assert len(chunks) > 0, "No chunks produced from IFU pages"

    # Embed all chunks
    embedder = IFUEmbedder(
        cfg,
        batch_size=64,
        normalize=True,
        out_dtype="float32",
        filter_lang=None,
    )
    assert embedder.test_connection() is True

    result = embedder.embed_chunks(chunks)

    if isinstance(result, tuple) and len(result) == 2:
        records, _ = result
    else:
        records = result

    assert len(records) == len(chunks), "Embeddings count must match number of chunks"

    # Upsert into Chroma vector database

    print("endpoint:", cfg.chroma_endpoint)
    print("tenant:", cfg.chroma_tenant)
    print("database:", cfg.chroma_database)
    print("api key present:", bool(cfg.chroma_api_key))

    store: IFUVectorStore = ChromaIFUVectorStore(
        cfg=cfg,
        embedder=embedder,
        collection_name="ifu_chunks_test",
    )
    assert store.test_connection() is True

    store.upsert_chunk_embeddings(doc_id=doc_id, chunks=chunks, records=records)
    # Run a semantic query on Chroma vector database
    query = "maximum patient weight"
    res = store.query_text(query, n_results=3)

    assert "ids" in res and res["ids"], "No 'ids' in Chroma query result"
    assert len(res["ids"][0]) > 0, "No results returned from Chroma query"

    top_doc = (
        res["documents"][0][0]
        if res.get("documents") and res["documents"][0]
        else ""
    )
    print(f"\nTop match for query '{query}':\n{top_doc[:200].replace('\n', ' ')} …")

    # Clean up environment delete blob from storage and vectors from Chroma
    deleted = store.delete_by_doc_id(doc_id)
    assert deleted > 0

    try:
        container_client = loader.blob_service.get_container_client(container)
        blob_client = container_client.get_blob_client(uploaded_blob_name)
        blob_client.delete_blob()
    except Exception as e:
        # Best-effort cleanup; don't fail test on cleanup error
        print(f"Blob cleanup failed for {uploaded_blob_name}: {e}")
