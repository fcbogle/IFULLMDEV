# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-01
# Description: AzureHealth (with local embedding store)
# -----------------------------------------------------------------------------
import uuid
from typing import List, Dict, Tuple
import os, requests
import numpy as np

from azure.core.credentials import AzureNamedKeyCredential, AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from openai import AzureOpenAI
from openai import OpenAI

from AzureConfig import AzureConfig

API_VER = "2024-10-21"


class AzureHealth:
    def __init__(self, cfg: AzureConfig):
        self.cfg = cfg
        if not all((
            cfg.storage_account,
            cfg.storage_key,
            cfg.search_endpoint,
            cfg.search_key,
            cfg.openai_endpoint,
            cfg.openai_azure_api_key,
            cfg.openai_azure_embed_deployment,
        )):
            missing = [k for k, v in {
                "AZURE_STORAGE_ACCOUNT": cfg.storage_account,
                "AZURE_STORAGE_KEY": cfg.storage_key,
                "AZURE_SEARCH_ENDPOINT": cfg.search_endpoint,
                "AZURE_SEARCH_KEY": cfg.search_key,
                "AZURE_OPENAI_ENDPOINT": cfg.openai_endpoint,
                "AZURE_OPENAI_API_KEY": cfg.openai_azure_api_key,
                "AZURE_OPENAI_EMBED_DEPLOYMENT": cfg.openai_azure_embed_deployment,
                "OPENAI_API_KEY": cfg.openai_api_key,
                "OPENAI_CHAT_MODEL": cfg.openai_azure_chat_deployment,
            }.items() if not v]
            raise ValueError(f"Missing environment variables: {',  '.join(missing)}")

        blob_endpoint = f"https://{cfg.storage_account}.blob.core.windows.net"
        self.blob_client = BlobServiceClient(
            account_url=blob_endpoint,
            credential=AzureNamedKeyCredential(cfg.storage_account, cfg.storage_key),
        )
        self.search_indexes = SearchIndexClient(
            endpoint=cfg.search_endpoint,
            credential=AzureKeyCredential(cfg.search_key),
        )

        endpoint = cfg.openai_endpoint.rstrip("/")
        self.oai = AzureOpenAI(
            api_key=cfg.openai_azure_api_key,
            api_version=API_VER,
            azure_endpoint=endpoint,
        )

        # 🔹 Local in-memory embedding store (label → np.array)
        self._embeddings: Dict[str, np.ndarray] = {}

    # ---------------- Blob & Search ---------------- #
    def check_blob(self) -> bool:
        """

        :return:
        """
        container = "healthcheck"
        cc = self.blob_client.get_container_client(container)
        try:
            cc.create_container()
        except ResourceExistsError:
            pass
        data = b"ping"
        name = f"ping-{uuid.uuid4().hex[:8]}.txt"
        bc = self.blob_client.get_blob_client(container, name)
        bc.upload_blob(data, overwrite=True)
        return bc.download_blob().readall() == data

    def check_search_connect(self) -> List[str]:
        return [idx.name for idx in self.search_indexes.list_indexes()]

    # ---------------- Azure OpenAI helpers ---------------- #
    def list_openai_deployments(self) -> List[str]:
        url = f"{self.cfg.openai_endpoint.rstrip('/')}/openai/deployments?api-version={API_VER}"
        r = requests.get(url, headers={"api-key": self.cfg.openai_api_key}, timeout=30)
        r.raise_for_status()
        items = r.json().get("data", [])
        names = [it.get("id") or it.get("name") for it in items if (it.get("id") or it.get("name"))]
        return names

    def ping_openai_embedding(self) -> int:
        endpoint = self.cfg.openai_endpoint.rstrip("/")
        dep = (self.cfg.openai_azure_embed_deployment or "").strip()
        url = f"{endpoint}/openai/deployments/{dep}/embeddings?api-version={API_VER}"
        body = {"input": "IFU smoke test"}
        headers = {"api-key": self.cfg.openai_azure_api_key, "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=body, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Embeddings ping failed: {r.status_code} {r.text[:300]}")
        data = r.json()
        emb = data["data"][0]["embedding"]
        return len(emb)

    def check_openai_embeddings(self) -> int:
        dep = (self.cfg.openai_azure_embed_deployment or "").strip()
        print("AOAI endpoint:", self.cfg.openai_endpoint)
        print("AOAI embed deployment (from .env):", dep)
        deployments = self.list_openai_deployments()
        print("AOAI deployments (names):", deployments)
        if dep not in deployments:
            raise RuntimeError(
                f"Embedding deployment '{dep}' not found at {self.cfg.openai_endpoint}. "
                f"Use one of: {deployments}"
            )
        vec = self.oai.embeddings.create(input="IFU smoke test", model=dep).data[0].embedding
        return len(vec)

    # ---------------- New functionality ---------------- #
    def add_embedding(self, label: str, text: str) -> Tuple[str, int]:
        """
        Generates an embedding for the given text and stores it locally under `label`.
        Returns (label, vector_length)
        """
        dep = (self.cfg.openai_azure_embed_deployment or "").strip()
        vec = self.oai.embeddings.create(input=text, model=dep).data[0].embedding
        self._embeddings[label] = np.array(vec, dtype=np.float32)
        print(f"✅ Added embedding '{label}' (dim={len(vec)})")
        return label, len(vec)

    def list_azure_openai_deployments(self) -> list[str]:
        """
        Best-effort attempt to list deployments on this Azure OpenAI resource.
        Some resources/API versions don't support GET /openai/deployments; in that
        case we log a warning and return [] instead of raising.
        """
        import requests, json

        endpoint = self.cfg.openai_endpoint.rstrip("/")
        key = self.cfg.openai_azure_api_key

        api_versions = [
            "2024-10-21",
            "2024-05-01-preview",
            "2023-12-01-preview",
            "2023-09-01-preview",
            "2023-03-15-preview",
        ]

        for ver in api_versions:
            url = f"{endpoint}/openai/deployments?api-version={ver}"
            r = requests.get(url, headers={"api-key": key}, timeout=30)
            if r.status_code == 404:
                # This api-version doesn't support /deployments on this resource
                print(f"[deployments] 404 for api-version={ver}, trying next...")
                continue

            # For any non-404, either succeed or raise a meaningful error
            try:
                r.raise_for_status()
            except Exception:
                print(f"[deployments] status={r.status_code} for api-version={ver}")
                print(r.text[:500])
                raise

            data = r.json().get("data", [])
            names = [d.get("id") or d.get("name") for d in data]
            print(f"🔍 Deployments visible at {endpoint} (api-version={ver}):")
            for d in data:
                dep = d.get("id") or d.get("name")
                model = d.get("model", "unknown")
                print(f"  - {dep}: model={model}")
            return names

        print("ℹ️ This resource/API combination does not support listing deployments via /openai/deployments.")
        return []

    def list_embeddings(self) -> List[str]:
        """
        Lists all stored embedding labels (and optionally prints norms).
        """
        if not self._embeddings:
            print("ℹ️ No embeddings stored yet.")
            return []
        print("\n📦 Stored embeddings:")
        for k, v in self._embeddings.items():
            print(f"  - {k}: dim={v.shape[0]}, norm={np.linalg.norm(v):.4f}")
        return list(self._embeddings.keys())

    # ---------------- Optional chat ---------------- #
    def check_openai_chat(self) -> str:
        dep = (self.cfg.openai_azure_chat_deployment or "").strip()
        if not dep:
            return "chat deployment not set"
        print("AOAI chat deployment:", dep)
        resp = self.oai.chat.completions.create(
            model=dep,
            messages=[
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Reply with OK"},
            ],
            max_tokens=5,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    cfg = AzureConfig.from_env()
    health = AzureHealth(cfg)

    # Run health checks
    print("Blob OK:", health.check_blob())
    print("Search indexes:", health.check_search_connect())
    print("Embedding platform: ", health.list_azure_openai_deployments())
    print("Embedding dim (REST):", health.ping_openai_embedding())

    # Add and list local embeddings
    health.add_embedding("Section A", "The device must be sterilized before use.")
    health.add_embedding("Section B", "The product is designed for single use only.")
    health.list_embeddings()