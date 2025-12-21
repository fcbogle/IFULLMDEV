# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-20
# Description: test_query_router.py
# -----------------------------------------------------------------------------
import logging

import pytest
from starlette.testclient import TestClient

from api.main import app

logger = logging.getLogger(__name__)

@pytest.mark.integration
def test_post_query_endpoint():
    client = TestClient(app)

    resp = client.post(
        "/query",
        json={
            "query": "oil leak from device",
            "n_results": 3,
        },
    )

    # ---- diagnostics (crucial while stabilizing the API) ----
    print("\nSTATUS:", resp.status_code)
    print("HEADERS:", resp.headers)
    try:
        print("RESPONSE JSON:", resp.json())
    except Exception as e:
        logger.exception("Response failed: %s", e)
        print("RESPONSE TEXT:", resp.text)

    # ---- assertions ----
    assert resp.status_code == 200

    data = resp.json()
    assert "results" in data, "No 'results' in response"
    assert isinstance(data["results"], list)

