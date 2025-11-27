# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-05
# Description: Tests the health of basic Blob connectivity
# -----------------------------------------------------------------------------
import uuid
import logging


from azure.core.credentials import AzureNamedKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient
from utility.logging_utils import get_class_logger, get_logger

from config.Config import Config



class BlobHealth:
    def __init__(self, cfg: Config, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or get_class_logger(self.__class__)

        endpoint = f"https://{cfg.storage_account}.blob.core.windows.net"
        self.logger.info("Initialising BlobHealth with endpoint: %s", endpoint)

        self.client = BlobServiceClient(
            account_url=endpoint,
            credential=AzureNamedKeyCredential(cfg.storage_account, cfg.storage_key),
        )

    def check_blob(self) -> bool:
        """
        Simple healthcheck:
          1. Ensures a 'healthcheck' container exists
          2. Uploads a small blob
          3. Downloads it again and compares bytes
        """
        container = "healthcheck"
        cc = self.client.get_container_client(container)

        # Ensure container exists
        try:
            cc.create_container()
            self.logger.info("Created container '%s' for healthcheck", container)
        except ResourceExistsError:
            self.logger.debug("Container '%s' already exists", container)

        # Prepare test payload
        data = b"ping"
        name = f"ping_{uuid.uuid4().hex[:8]}.txt"

        self.logger.info("Uploading blob '%s' with content bytes=%s", name, list(data))

        bc = self.client.get_blob_client(container, name)
        bc.upload_blob(data, overwrite=True)

        downloaded = bc.download_blob().readall()
        self.logger.info(
            "Downloaded blob '%s' with content bytes=%s", name, list(downloaded)
        )

        ok = downloaded == data
        if ok:
            self.logger.info("Blob healthcheck PASSED for blob '%s'", name)
        else:
            self.logger.error("Blob healthcheck FAILED for blob '%s'", name)

        return ok


if __name__ == "__main__":
    # simple tests prior to orchestration
    import logging

    logging.basicConfig(level=logging.INFO)

    cfg = Config.from_env()
    logger = get_logger(__name__)
    bh = BlobHealth(cfg, logger=logger)
    ok = bh.check_blob()
    logger.info("BlobHealth.check_blob() -> %s", ok)
