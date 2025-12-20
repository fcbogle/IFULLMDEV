# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-20
# Description: test_query_router.py
# -----------------------------------------------------------------------------
import pytest
from starlette.testclient import TestClient

from api.main import app


@pytest.mark.integration
def test_post_query_endpoint():
    client = TestClient(app)

    resp = client.post(
        "/query",
        json={
            "query": "noise from knee",
            "n_results": 3,
        },
    )

    # ---- diagnostics (crucial while stabilising the API) ----
    print("\nSTATUS:", resp.status_code)
    print("HEADERS:", resp.headers)
    try:
        print("RESPONSE JSON:", resp.json())
    except Exception:
        print("RESPONSE TEXT:", resp.text)

    # ---- assertions ----
    assert resp.status_code == 200

    data = resp.json()
    assert "results" in data, "No 'results' in response"
    assert isinstance(data["results"], list)

