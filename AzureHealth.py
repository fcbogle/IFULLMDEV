# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-01
# Description: AzureHealth
# -----------------------------------------------------------------------------
import uuid
from typing import List

from azure.core.credentials import AzureNamedKeyCredential, AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

from AzureConfig import AzureConfig
class AzureHealth:
    def __init__(self, cfg: AzureConfig):
        self.cfg = cfg
        if not all((cfg.storage_account, cfg.storage_key, cfg.search_endpoint, cfg.search_key)):
            missing = [k for k,v in {
                "AZURE_STORAGE_ACCOUNT": cfg.storage_account,
                "AZURE_STORAGE_KEY": cfg.storage_key,
                "AZURE_SEARCH_ENDPOINT": cfg.search_endpoint,
                "AZURE_SEARCH_KEY": cfg.search_key
            }.items() if not v]
            raise ValueError(f"Missing environment variables: {',  '.join(missing)}")

        # Create reusable clients
        blob_endpoint = f"https://{cfg.storage_account}.blob.core.windows.net"
        self.blob_client = BlobServiceClient(
            account_url=blob_endpoint,
            credential=AzureNamedKeyCredential(cfg.storage_account, cfg.storage_key)
        )
        self.search_indexes = SearchIndexClient(
            endpoint=cfg.search_endpoint,
            credential=AzureKeyCredential(cfg.search_key)
        )

    def check_blob(self) -> bool:
        container = "healthcheck"

        cc = self.blob_client.get_container_client(container)
        try:
            cc.create_container()  # first run creates it; later runs will hit ResourceExistsError
        except ResourceExistsError:
            pass  # container already exists, continue code execution

        data = b"ping"
        name = f"ping-{uuid.uuid4().hex[:8]}.txt"
        bc = self.blob_client.get_blob_client(container, name)
        bc.upload_blob(data, overwrite=True)
        return bc.download_blob().readall() == data

    def check_search_connect(self) -> List[str]:
        # returns list of index names; empty list is fine on a new service
        return [idx.name for idx in self.search_indexes.list_indexes()]

if __name__ == "__main__":
    cfg = AzureConfig.from_env()
    health = AzureHealth(cfg)

    ok_blob = health.check_blob()
    print("Blob OK:", ok_blob)

    indexes = health.check_search_connect()
    print(f"Search OK: endpoint={cfg.search_endpoint} indexes={indexes}")
