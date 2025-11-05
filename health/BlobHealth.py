# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-05
# Description: Tests the health of basic Blob connectivity
# -----------------------------------------------------------------------------
import uuid

from azure.core.credentials import AzureNamedKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

from AzureConfig import AzureConfig


class BlobHealth:
    def __init__(self, cfg: AzureConfig):
        self.cfg = cfg
        endpoint = f"https://{cfg.storage_account}.blob.core.windows.net"
        self.client = BlobServiceClient(
            account_url=endpoint,
            credential=AzureNamedKeyCredential(cfg.storage_account, cfg.storage_key),
        )

    def check_blob(self) -> bool:
        container = "healthcheck"
        cc = self.client.get_container_client(container)
        try:
            cc.create_container()
        except ResourceExistsError:
            pass

        data = b"ping"
        name = f"ping_{uuid.uuid4().hex[:8]}.txt"
        print(f"Uploading blob '{name}' with content: {data}")
        print(f"Raw byte values uploaded: {[b for b in data]}")
        bc = self.client.get_blob_client(container, name)
        bc.upload_blob(data, overwrite=True)
        downloaded = bc.download_blob().readall()
        print(f"Raw byte values returned: {[b for b in downloaded]}")
        return bc.download_blob().readall() == data

if __name__ == "__main__":
    # simple tests prior to orchestration
    cfg = AzureConfig.from_env()
    bh = BlobHealth(cfg)
    ok = bh.check_blob()
    print("BlobHealth.check_blob() -> ", ok)
