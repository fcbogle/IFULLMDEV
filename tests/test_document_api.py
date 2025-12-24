# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-24
# Description: test_document_api.py
# -----------------------------------------------------------------------------
import os
import uuid
import time
import pytest
import httpx

API_BASE_URL = os.getenv("IFU_API_BASE_URL", "http://127.0.0.1:8000")
TEST_CONTAINER = os.getenv("IFU_TEST_CONTAINER", "ifu-docs-test")


def _unique_name(prefix: str, suffix: str = ".pdf") -> str:
    return f"{prefix}-{uuid.uuid4().hex}{suffix}"


def _require_env():
    # Keep this minimal; add others if your API requires auth.
    return True


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    if not _require_env():
        pytest.skip("Integration env not configured")
    return httpx.Client(base_url=API_BASE_URL, timeout=60.0)


@pytest.mark.integration
def test_documents_endpoints_smoke(client: httpx.Client):
    # 1) list docs (container must be passed)
    r = client.get("/documents", params={"container": TEST_CONTAINER})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["container"] == TEST_CONTAINER
    assert isinstance(payload["documents"], list)

    # pick a doc if any exist
    if not payload["documents"]:
        pytest.skip("No blobs in test container to validate GET/HEAD")

    doc0 = payload["documents"][0]
    doc_id = doc0["doc_id"]

    # 2) ids
    r = client.get("/documents/ids", params={"container": TEST_CONTAINER})
    assert r.status_code == 200, r.text
    ids = r.json()["doc_ids"]
    assert doc_id in ids

    # 3) get single
    r = client.get(f"/documents/{doc_id}", params={"container": TEST_CONTAINER})
    assert r.status_code == 200, r.text
    one = r.json()
    assert one["document"]["doc_id"] == doc_id

    # 4) head (exists)
    r = client.head(f"/documents/{doc_id}", params={"container": TEST_CONTAINER})
    assert r.status_code == 200, r.text

    # 5) head (missing)
    r = client.head("/documents/does-not-exist.pdf", params={"container": TEST_CONTAINER})
    assert r.status_code in (404, 422, 400), r.text
    # 404 expected; 422/400 only if your validation rejects it


@pytest.mark.integration
def test_ingest_reindex_delete_vectors_idempotent(client: httpx.Client):
    """
    This assumes your container already has known doc_ids to ingest.
    If you want fully self-contained tests, add an upload endpoint or
    upload test blobs directly to Azure in a fixture.
    """
    # list doc ids (need at least 1 to ingest)
    r = client.get("/documents/ids", params={"container": TEST_CONTAINER})
    assert r.status_code == 200, r.text
    doc_ids = r.json()["doc_ids"]
    if not doc_ids:
        pytest.skip("No doc_ids to ingest in test container")

    target = doc_ids[0]

    # ingest
    r = client.post(
        "/documents/ingest",
        json={"container": TEST_CONTAINER, "doc_ids": [target], "document_type": "IFU"},
    )
    assert r.status_code == 200, r.text
    ing = r.json()
    assert ing["requested"] == 1
    assert ing["ingested"] in (0, 1)  # 0 if your ingest pipeline decides to skip

    # reindex
    r = client.post(f"/documents/{target}/reindex", params={"container": TEST_CONTAINER, "document_type": "IFU"})
    assert r.status_code == 200, r.text

    # delete vectors (first time should normally delete >0 if indexed)
    r = client.delete(f"/documents/{target}/vectors")
    assert r.status_code == 200, r.text
    deleted1 = r.json()["deleted"]
    assert deleted1 >= 0

    # delete vectors again (should eventually be 0 if idempotent)
    deleted2 = None
    for attempt in range(10):
        r = client.delete(f"/documents/{target}/vectors")
        assert r.status_code == 200, r.text
        deleted2 = r.json()["deleted"]
        if deleted2 == 0:
            break
        time.sleep(0.2)

    assert deleted2 == 0, f"Expected idempotent delete to return 0, got {deleted2}"
