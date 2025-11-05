# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-05
# Description: SearchHealth
# -----------------------------------------------------------------------------
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

from AzureConfig import AzureConfig


class SearchHealth:
    def __init__(self, cfg: AzureConfig):
        self.cfg = cfg
        self.index_client = SearchIndexClient(
            endpoint=cfg.search_endpoint,
            credential=AzureKeyCredential(cfg.search_key)
        )

    def list_indexes(self) -> List[str]:
        return [idx.name for idx in self.index_client.list_indexes()]

if __name__ == "__main__":
    cfg = AzureConfig.from_env()
    sh = SearchHealth(cfg)
    try:
        indexes = sh.list_indexes()
        print("Search indexes:", indexes)
        if indexes:
            print("✅ SearchHealth: service reachable, indexes returned.")
        else:
            print("ℹ️ SearchHealth: service reachable, but no indexes defined yet.")
    except Exception as e:
        print("❌ SearchHealth: failed to list indexes:", e)
