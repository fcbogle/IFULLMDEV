# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-06
# Description: ChromaHealth
# -----------------------------------------------------------------------------

from typing import List, Optional
import logging

import chromadb

from AzureConfig import AzureConfig
from utility.logging_utils import get_logger  # adjust if this lives in another package


class ChromaHealth:
    """
    Healthcheck utility for Chroma Cloud.

    - list_indexes(): list existing collections
    - add_index():    create/verify a healthcheck collection and R/W test
    - remove_index(): delete a dedicated healthcheck collection
    """

    def __init__(self, cfg: AzureConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)

        self.logger.info(
            "Initialising ChromaHealth with tenant=%s, database=%s",
            cfg.chroma_tenant,
            cfg.chroma_database,
        )

        self.client = chromadb.CloudClient(
            api_key=cfg.chroma_api_key,
            tenant=cfg.chroma_tenant,
            database=cfg.chroma_database,
        )

    def list_indexes(self) -> List[str]:
        """
        Returns a list of collection names ("indexes") in Chroma.
        """
        self.logger.info("Listing Chroma collections (indexes)")
        collections = self.client.list_collections()
        names = [c.name for c in collections]
        self.logger.info("Found %d collection(s): %s", len(names), names)
        return names

    def add_index(self, index_name: str) -> None:
        """
        Creates (or reuses) a collection and performs a simple R/W healthcheck:
          - add a dummy document with a small embedding
          - query by embedding
          - ensure we get the same ID back
          - clean up the dummy record
        """
        self.logger.info("Starting Chroma healthcheck for collection '%s'", index_name)

        try:
            # 1. Get or create the collection
            collection = self.client.get_or_create_collection(name=index_name)
            self.logger.info("Collection '%s' is available.", index_name)

            # 2. Define a tiny dummy embedding (10 dimensions for healthcheck only)
            test_id = "healthcheck-doc-1"
            test_embedding = [0.1] * 10
            test_doc = "This is a Chroma healthcheck document."
            test_metadata = {"purpose": "healthcheck"}

            # In case we ran this before, try to delete any prior test doc
            try:
                collection.delete(ids=[test_id])
                self.logger.debug(
                    "Deleted existing healthcheck document '%s' (if it existed).",
                    test_id,
                )
            except Exception:
                # If it doesn't exist, that's fine
                self.logger.debug(
                    "No existing healthcheck document '%s' to delete.", test_id
                )

            # 3. Add dummy point
            self.logger.info(
                "Adding dummy document '%s' to collection '%s'.",
                test_id,
                index_name,
            )
            collection.add(
                ids=[test_id],
                documents=[test_doc],
                embeddings=[test_embedding],
                metadatas=[test_metadata],
            )

            # 4. Query it back
            self.logger.info(
                "Querying collection '%s' for the healthcheck document.", index_name
            )
            result = collection.query(
                query_embeddings=[test_embedding],
                n_results=1,
            )

            ids = result.get("ids", [[]])
            self.logger.debug("Query result ids: %s", ids)

            if ids and ids[0] and test_id in ids[0]:
                self.logger.info(
                    "Chroma healthcheck query returned the expected document '%s'.",
                    test_id,
                )
            else:
                self.logger.warning(
                    "Chroma healthcheck query did NOT return the expected document '%s'. "
                    "Result ids: %s",
                    test_id,
                    ids,
                )
                return

            # 5. Clean up the dummy record
            collection.delete(ids=[test_id])
            self.logger.info(
                "Cleaned up healthcheck document '%s' from collection '%s'.",
                test_id,
                index_name,
            )

            self.logger.info(
                "Chroma healthcheck for collection '%s' completed successfully.",
                index_name,
            )

        except Exception as e:
            # Common failure: embedding dimension mismatch if the collection is
            # already in use for real data with a different embedding size.
            self.logger.exception(
                "Chroma healthcheck for collection '%s' failed: %s", index_name, e
            )

    def remove_index(self, index_name: str) -> None:
        """
        Deletes a collection from Chroma.
        Intended for dedicated healthcheck collections.
        """
        self.logger.info("Attempting to delete collection '%s'.", index_name)

        try:
            existing = [c.name for c in self.client.list_collections()]
            if index_name not in existing:
                self.logger.info(
                    "Collection '%s' not found, nothing to delete.", index_name
                )
                return

            self.client.delete_collection(index_name)
            self.logger.info("Deleted collection '%s' successfully.", index_name)

        except Exception as e:
            self.logger.exception(
                "Failed to delete collection '%s': %s", index_name, e
            )


if __name__ == "__main__":
    import logging

    # Basic configuration for standalone runs
    logging.basicConfig(level=logging.INFO)

    cfg = AzureConfig.from_env()
    ch = ChromaHealth(cfg)

    test_index = "healthcheck-index"

    logger = ch.logger
    logger.info("CHROMA_TENANT: %s", cfg.chroma_tenant)
    logger.info("CHROMA_DATABASE: %s", cfg.chroma_database)
    logger.info("CHROMA_API_KEY set: %s", bool(cfg.chroma_api_key))

    logger.info("Existing indexes: %s", ch.list_indexes())

    ch.add_index(test_index)
    logger.info("After add & R/W test on '%s': %s", test_index, ch.list_indexes())

    ch.remove_index(test_index)
    logger.info("After remove: %s", ch.list_indexes())


