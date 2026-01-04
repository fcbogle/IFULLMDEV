# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-28
# Description: gradio_app.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import requests

from utility.assets import img_to_data_uri

# Environment configuration
API_BASE_URL = os.getenv("IFU_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LOG_FILE = os.getenv("IFU_LOG_FILE", "./logs/ifullmdev.log")
LOG_TAIL_LINES = int(os.getenv("IFU_UI_LOG_TAIL_LINES", "400"))
TIMEOUT_SECONDS = int(os.getenv("IFU_UI_TIMEOUT_SECONDS", "30"))

DEFAULT_CONTAINER = os.getenv("IFU_DEFAULT_CONTAINER", "ifu-docs-test")

# If your chunk metadata includes "container", you can enable this
# to constrain retrieval automatically.
USE_CONTAINER_FILTER = os.getenv("IFU_UI_USE_CONTAINER_FILTER", "0").strip() == "1"

# Languages (from your LangDetect logs)
LANG_CHOICES = ["any", "en", "fr", "de", "es", "it", "nl", "pt", "pl", "cs", "id", "ca", "ro"]


# ---------------------------
# HTTP helpers
# ---------------------------
def _url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def _get(path: str, params: Optional[dict] = None) -> Dict[str, Any]:
    try:
        r = requests.get(_url(path), params=params, timeout=TIMEOUT_SECONDS)
        if not r.ok:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"RequestException: {e}"}


def _post(path: str, payload: Dict[str, Any], params: Optional[dict] = None) -> Dict[str, Any]:
    try:
        r = requests.post(_url(path), params=params, json=payload, timeout=TIMEOUT_SECONDS)
        if not r.ok:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"RequestException: {e}"}


def _delete(path: str, params: Optional[dict] = None) -> Dict[str, Any]:
    try:
        r = requests.delete(_url(path), params=params, timeout=TIMEOUT_SECONDS)
        if not r.ok:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"RequestException: {e}"}


# ---------------------------
# Where-clause helper (Chroma-safe)
# ---------------------------
def _combine_where(parts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Chroma requires the top-level where to have exactly ONE operator when combining clauses.
    So:
      {"a": {"$eq": 1}, "b": {"$eq": 2}}  ❌
      {"$and": [{"a":{"$eq":1}}, {"b":{"$eq":2}}]} ✅
    """
    parts = [p for p in parts if p]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def _build_where(
    *,
    container: str,
    lang: str,
    where_json: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (where, error_message).

    language:
      - "any" / "" / None => no lang filter
      - otherwise => {"lang": {"$eq": <language>}}
    """
    base_where: Optional[Dict[str, Any]] = None
    where_json = (where_json or "").strip()

    if where_json:
        try:
            base_where = json.loads(where_json)
            if base_where is not None and not isinstance(base_where, dict):
                return None, "where JSON must be an object (dict)"
        except Exception as e:
            return None, f"Invalid where JSON: {e}"

    parts: List[Dict[str, Any]] = []
    if base_where:
        parts.append(base_where)

    # ---- Language filter ----
    lang = (lang or "").strip().lower()
    if lang and lang != "any":
        parts.append({"lang": {"$eq": lang}})

    # ---- Optional container filter ----
    if USE_CONTAINER_FILTER:
        parts.append({"container": {"$eq": container}})

    return _combine_where(parts), None


# ---------------------------
# Log tailing
# ---------------------------
def tail_log_file(path: str, n_lines: int = 200) -> str:
    try:
        if not path or not os.path.exists(path):
            return f"[log] file not found: {path}"
        with open(path, "rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-n_lines:])
    except Exception as e:
        return f"[log] failed to read log file: {e}"


# ---------------------------
# Ingest job helpers
# ---------------------------
def _extract_job_id(job_json: str) -> str:
    try:
        obj = json.loads(job_json or "{}")
        return (obj.get("job_id") or "").strip()
    except Exception:
        return ""


def ui_get_ingest_job(job_id: str) -> str:
    job_id = (job_id or "").strip()
    if not job_id:
        return json.dumps({"error": "job_id required"}, indent=2)
    out = _get(f"/documents/ingest/jobs/{job_id}")
    return json.dumps(out, indent=2)


# ---------------------------
# Blob UI functions
# ---------------------------
def ui_list_blobs(container: str) -> pd.DataFrame:
    container = (container or "").strip() or DEFAULT_CONTAINER
    out = _get("/blobs", params={"container": container})
    if out.get("error"):
        return pd.DataFrame([out])
    blobs = out.get("blobs") or out.get("items") or []
    return pd.DataFrame(blobs)


def ui_upload_blobs(container: str, blob_prefix: str, files) -> str:
    container = (container or "").strip() or DEFAULT_CONTAINER
    blob_prefix = (blob_prefix or "").strip()

    if not files:
        return json.dumps({"error": "No files selected"}, indent=2)

    if not isinstance(files, list):
        files = [files]

    multipart = []
    for f in files:
        p = getattr(f, "name", None)
        if not p:
            continue

        ctype, _ = mimetypes.guess_type(p)
        ctype = ctype or "application/octet-stream"

        multipart.append(("files", (Path(p).name, open(p, "rb"), ctype)))

    if not multipart:
        return json.dumps({"error": "No valid files found"}, indent=2)

    try:
        r = requests.post(
            _url("/blobs/upload"),
            params={"container": container, "blob_prefix": blob_prefix},
            files=multipart,
            timeout=TIMEOUT_SECONDS,
        )
        if not r.ok:
            return json.dumps(
                {"error": "upload failed", "status_code": r.status_code, "response": r.text},
                indent=2,
            )
        return json.dumps(r.json(), indent=2)

    except Exception as e:
        return json.dumps({"error": f"upload failed: {e}"}, indent=2)

    finally:
        for _, (_, fh, _) in multipart:
            try:
                fh.close()
            except Exception:
                pass


def ui_delete_blob(container: str, blob_name: str) -> str:
    container = (container or "").strip() or DEFAULT_CONTAINER
    blob_name = (blob_name or "").strip()
    if not blob_name:
        return json.dumps({"error": "blob_name required"}, indent=2)

    out = _delete(f"/blobs/{blob_name}", params={"container": container})
    return json.dumps(out, indent=2)


def ui_set_blob_metadata(container: str, blob_name: str, metadata_json: str) -> str:
    container = (container or "").strip() or DEFAULT_CONTAINER
    blob_name = (blob_name or "").strip()
    metadata_json = (metadata_json or "").strip()

    if not blob_name:
        return json.dumps({"error": "blob_name required"}, indent=2)

    meta: Dict[str, Any] = {}
    if metadata_json:
        try:
            meta = json.loads(metadata_json)
            if not isinstance(meta, dict):
                return json.dumps({"error": "metadata JSON must be an object (dict)"}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"invalid metadata JSON: {e}"}, indent=2)

    try:
        r = requests.patch(
            _url(f"/blobs/{blob_name}/metadata"),
            params={"container": container},
            json={"metadata": meta},
            timeout=TIMEOUT_SECONDS,
        )
        if not r.ok:
            return json.dumps({"error": f"HTTP {r.status_code}: {r.text}"}, indent=2)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return json.dumps({"error": f"metadata update failed: {e}"}, indent=2)


def ui_list_blobs_with_status(container: str) -> pd.DataFrame:
    container = (container or "").strip() or DEFAULT_CONTAINER

    blobs_out = _get("/blobs", params={"container": container})
    if blobs_out.get("error"):
        return pd.DataFrame([blobs_out])

    blobs = blobs_out.get("blobs") or blobs_out.get("items") or []
    if isinstance(blobs, dict):
        blobs = [blobs]
    if not blobs:
        return pd.DataFrame([])

    stats_out = _get("/stats", params={"blob_container": container})
    indexed_docs = stats_out.get("documents") or []
    indexed_ids = {d.get("doc_id") for d in indexed_docs if isinstance(d, dict)}
    indexed_ids.discard(None)

    df = pd.DataFrame(blobs)

    if "blob_metadata" in df.columns and len(df) > 0:
        print("blob_metadata type counts:",
              df["blob_metadata"].apply(lambda x: type(x).__name__).value_counts().to_dict())
        print("sample blob_metadata:", df["blob_metadata"].iloc[0])

    # Ensure we have a consistent blob name field
    if "blob_name" not in df.columns and "name" in df.columns:
        df["blob_name"] = df["name"]

    # MIME type handling
    if "content_type" in df.columns:
        df["mime_type"] = df["content_type"]
    else:
        df["mime_type"] = None

    if "blob_metadata" in df.columns:
        def _meta_mime(r):
            md = r.get("blob_metadata") or {}
            if isinstance(md, dict):
                return md.get("client_content_type") or md.get("content_type")
            return None

        df["mime_type"] = df["mime_type"].fillna(df.apply(_meta_mime, axis=1))

    # ------------------------------------------------------------------
    # 🔍 DEBUG: expose metadata evaluation (TEMPORARY)
    # ------------------------------------------------------------------
    # if "blob_metadata" in df.columns:
    #     df["debug_meta_type"] = df["blob_metadata"].apply(
    #         lambda x: type(x).__name__
    #     )
    #     df["debug_document_type"] = df["blob_metadata"].apply(
    #         lambda md: (md or {}).get("document_type") if isinstance(md, dict) else None
    #     )
    # else:
    #     df["debug_meta_type"] = None
    #     df["debug_document_type"] = None

    def _is_pdf(r) -> bool:
        # Prefer mime/content_type if present, fallback to filename extension.
        ct = (r.get("content_type") or r.get("mime_type") or "").lower()
        name = (r.get("blob_name") or r.get("name") or "").lower()

        if "pdf" in ct:
            return True
        return name.endswith(".pdf")

    def _has_required_metadata(r) -> bool:
        md = r.get("blob_metadata") or {}
        if not isinstance(md, dict):
            return False
        # adjust these keys to whatever you actually require
        return bool((md.get("document_type") or "").strip())

    def _status_row(r) -> tuple[str, str, str, str, bool]:
        name = r.get("blob_name")
        ingested = bool(name in indexed_ids) if isinstance(name, str) else False

        # Readiness (type + metadata)
        if not _is_pdf(r):
            status, status_detail = "🔴", "NOT SUPPORTED"
        elif not _has_required_metadata(r):
            status, status_detail = "🟠", "METADATA MISSING"
        else:
            status, status_detail = "🟢", "READY"

        # Ingestion (indexed or not)
        if ingested:
            ingest_status, ingest_status_detail = "🟢", "INGESTED"
        else:
            ingest_status, ingest_status_detail = "🔴", "NOT INGESTED"

        return status, status_detail, ingest_status, ingest_status_detail, ingested

    df[["status", "status_detail", "ingest_status", "ingest_status_detail", "is_ingested"]] = df.apply(
        lambda r: pd.Series(_status_row(r)),
        axis=1
    )

    df["container"] = container

    # Optional: drop noisy columns
    for col in ("blob_metadata",):
        if col in df.columns:
            df = df.drop(columns=[col])

    preferred = [
        "status", "status_detail",
        "ingest_status", "ingest_status_detail",
        "blob_name",
        "mime_type", "content_type",
        "size", "last_modified",
        "is_ingested", "container",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


# ---------------------------
# Documents UI functions
# ---------------------------
def ui_get_stats(container: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    container = (container or "").strip() or DEFAULT_CONTAINER
    stats = _get("/stats", params={"blob_container": container})
    stats_json = json.dumps(stats, indent=2)
    docs_df = pd.DataFrame(stats.get("documents") or [])
    blobs_df = pd.DataFrame(stats.get("blobs") or [])
    return stats_json, docs_df, blobs_df


def ui_list_documents(container: str) -> pd.DataFrame:
    container = (container or "").strip() or DEFAULT_CONTAINER
    out = _get("/documents", params={"container": container})
    docs = out.get("documents") or []
    if not docs:
        return pd.DataFrame([])

    df = pd.DataFrame(docs)

    if "is_indexed" not in df.columns:
        df["is_indexed"] = False
    if "indexed_last_modified" not in df.columns:
        df["indexed_last_modified"] = None
    if "blob_last_modified" not in df.columns:
        df["blob_last_modified"] = None

    def _status_row(r) -> tuple[str, str]:
        is_indexed = bool(r.get("is_indexed"))
        blob_lm = r.get("blob_last_modified")
        idx_lm = r.get("indexed_last_modified")
        is_stale = is_indexed and blob_lm and idx_lm and (str(blob_lm) != str(idx_lm))

        if not is_indexed:
            return "🔴", "NOT INDEXED"
        if is_stale:
            return "🟠", "STALE"
        return "🟢", "INDEXED"

    df[["status", "status_detail"]] = df.apply(lambda r: pd.Series(_status_row(r)), axis=1)

    for col in ("blob_metadata", "blob_name"):
        if col in df.columns:
            df = df.drop(columns=[col])

    preferred = [
        "status",
        "status_detail",
        "doc_id",
        "blob_name",
        "document_type",
        "page_count",
        "chunk_count",
        "blob_last_modified",
        "indexed_last_modified",
        "size",
        "content_type",
        "container",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def ui_list_document_ids(container: str) -> List[str]:
    container = (container or "").strip() or DEFAULT_CONTAINER
    out = _get("/documents/ids", params={"container": container})
    ids = out.get("doc_ids") or []
    return [x for x in ids if isinstance(x, str)]


def ui_refresh_documents_and_ids(container: str):
    df = ui_list_documents(container)
    ids = ui_list_document_ids(container)
    dd = gr.update(choices=ids, value=(ids[0] if ids else None))
    return df, dd


def ui_get_document(container: str, doc_id: str) -> str:
    container = (container or "").strip() or DEFAULT_CONTAINER
    doc_id = (doc_id or "").strip()
    out = _get(f"/documents/{doc_id}", params={"container": container})
    return json.dumps(out, indent=2)


def ui_reindex(container: str, doc_id: str, document_type: str) -> str:
    container = (container or "").strip() or DEFAULT_CONTAINER
    doc_id = (doc_id or "").strip()
    document_type = (document_type or "IFU").strip() or "IFU"

    try:
        r = requests.post(
            _url(f"/documents/{doc_id}/reindex"),
            params={"container": container, "document_type": document_type},
            timeout=TIMEOUT_SECONDS,
        )
        if not r.ok:
            return json.dumps({"error": f"HTTP {r.status_code}: {r.text}"}, indent=2)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return json.dumps({"error": f"reindex failed: {e}"}, indent=2)


def ui_ingest(container: str, doc_ids_text: str, document_type: str) -> str:
    """
    Bulk ingest now returns 202 with a job_id (queued), not a completed ingested count.
    """
    container = (container or "").strip() or DEFAULT_CONTAINER
    document_type = (document_type or "IFU").strip() or "IFU"

    raw_ids = (doc_ids_text or "").replace(",", "\n").splitlines()
    doc_ids = [x.strip() for x in raw_ids if x.strip()]

    payload = {"container": container, "doc_ids": doc_ids, "document_type": document_type}
    out = _post("/documents/ingest", payload=payload)
    return json.dumps(out, indent=2)


def ui_ingest_one(container: str, doc_id: str, document_type: str) -> str:
    """
    Single doc ingest endpoint:
      POST /documents/{doc_id}/ingest?container=...&document_type=...
    Returns 202 with a job_id.
    """
    container = (container or "").strip() or DEFAULT_CONTAINER
    doc_id = (doc_id or "").strip()
    document_type = (document_type or "IFU").strip() or "IFU"

    if not doc_id:
        return json.dumps({"error": "doc_id required"}, indent=2)

    out = _post(
        f"/documents/{doc_id}/ingest",
        payload={},
        params={"container": container, "document_type": document_type},
    )
    return json.dumps(out, indent=2)


def ui_delete_vectors(doc_id: str) -> str:
    doc_id = (doc_id or "").strip()
    out = _delete(f"/documents/{doc_id}/vectors")
    return json.dumps(out, indent=2)


# ---------------------------
# Query UI functions
# ---------------------------
def ui_query(container: str, query_text: str, n_results: int, lang: str, where_json: str) -> pd.DataFrame:
    container = (container or "").strip() or DEFAULT_CONTAINER
    query_text = (query_text or "").strip()
    if not query_text:
        return pd.DataFrame([{"error": "query must not be empty"}])

    where, err = _build_where(container=container, lang=lang, where_json=where_json)
    if err:
        return pd.DataFrame([{"error": err}])

    payload = {
        "query": query_text,
        "n_results": int(n_results),
        "where": where,
        "include_text": True,
        "include_scores": True,
        "include_metadata": True,
    }
    out = _post("/query", payload=payload)
    if out.get("error"):
        return pd.DataFrame([out])
    return pd.DataFrame(out.get("results") or [])


# ---------------------------
# Chat UI functions
# ---------------------------
def ui_chat(
    container: str,
    question: str,
    n_results: int,
    lang: str,
    where_json: str,
    temperature: float,
    max_tokens: int,
    chat_messages: List[Dict[str, str]] | None,
    api_history: List[Dict[str, str]] | None,
    tone: str,
) -> Tuple[List[Dict[str, str]], str, pd.DataFrame, List[Dict[str, str]]]:
    container = (container or "").strip() or DEFAULT_CONTAINER
    question = (question or "").strip()

    chat_messages = chat_messages or []
    api_history = api_history or []

    if not question:
        return chat_messages, "", pd.DataFrame(), api_history

    where, err = _build_where(container=container, lang=lang, where_json=where_json)
    if err:
        chat_messages = chat_messages + [{"role": "assistant", "content": err}]
        return chat_messages, "", pd.DataFrame(), api_history

    payload = {
        "question": question,
        "n_results": int(n_results),
        "where": where,
        "tone": tone,
        "language": lang,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "history": api_history or None,
    }

    out = _post("/chat", payload=payload)
    if out.get("error"):
        chat_messages = chat_messages + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": out["error"]},
        ]
        return chat_messages, "", pd.DataFrame(), api_history

    answer = out.get("answer") or ""
    sources = out.get("sources") or []

    chat_messages = chat_messages + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    api_history = api_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    sources_df = pd.DataFrame(sources)
    debug = json.dumps({"model": out.get("model"), "usage": out.get("usage"), "n_sources": len(sources)}, indent=2)

    return chat_messages, debug, sources_df, api_history


# ---------------------------
# Build Gradio UI
# ---------------------------
def build_gradio_app(api_base_url: str = API_BASE_URL) -> gr.Blocks:
    global API_BASE_URL
    API_BASE_URL = api_base_url.rstrip("/")

    with gr.Blocks(title="Blatchford IFU", analytics_enabled=False) as demo:
        logo_url = img_to_data_uri("./assets/blatchford.jpeg")

        gr.HTML(
            f"""
            <div class="ifu-header">
              <div class="ifu-left">
                <img class="ifu-logo-img" src="{logo_url}" alt="Blatchford logo" />
                <div class="ifu-title">
                  <h1>Blatchford IFU Large Language Model</h1>
                  <p>RAG + Chat UI</p>
                </div>
              </div>

              <div class="ifu-meta">
                <div class="ifu-chip"><b>API</b>: <code>{API_BASE_URL}</code></div>
                <div class="ifu-chip"><b>Container</b>: <code>{DEFAULT_CONTAINER}</code></div>
                <div class="ifu-chip"><b>Log</b>: <code>{LOG_FILE}</code></div>
                <div class="ifu-chip">
                  <b>Container filter</b>:
                  <code>{"ON" if USE_CONTAINER_FILTER else "OFF"}</code>
                </div>
              </div>
            </div>
            """
        )

        # ----------------
        # Blobs
        # ----------------
        with gr.Tab("Blobs"):
            gr.Markdown("""
            ### Blobs

            **Readiness legend**  
            🟢 READY — valid PDF and required metadata present; ready for ingestion  
            🟠 METADATA MISSING — PDF uploaded but required metadata is missing  
            🔴 NOT SUPPORTED — file is not a PDF or cannot be ingested

            **Ingestion legend**  
            🟢 INGESTED — vectors exist in the index for this blob  
            🔴 NOT INGESTED — no vectors exist yet
            """)
            with gr.Row():
                b_container = gr.Textbox(label="Blob container", value=DEFAULT_CONTAINER)
                b_refresh = gr.Button("Refresh blobs")

            blobs_df = gr.Dataframe(label="Blobs (with ingest status)", interactive=False)

            b_refresh.click(
                fn=ui_list_blobs_with_status,
                inputs=[b_container],
                outputs=[blobs_df],
            )

            gr.Markdown("### Upload PDFs to Blob Storage")
            with gr.Row():
                b_prefix = gr.Textbox(label="blob_prefix", value="")
                b_files = gr.File(label="Select PDF files", file_count="multiple", file_types=[".pdf", ".txt"])
            b_upload = gr.Button("Upload")
            b_upload_out = gr.Code(label="Upload response", language="json")
            b_upload.click(fn=ui_upload_blobs, inputs=[b_container, b_prefix, b_files], outputs=[b_upload_out])

            gr.Markdown("### Blob actions")
            with gr.Row():
                blob_name = gr.Textbox(label="blob_name", placeholder="BMK2IFU.pdf")
                b_delete = gr.Button("Delete blob")
            b_action_out = gr.Code(label="Blob action output", language="json")
            b_delete.click(fn=ui_delete_blob, inputs=[b_container, blob_name], outputs=[b_action_out])

            gr.Markdown("### Metadata")
            b_meta_json = gr.Textbox(
                label="metadata JSON",
                value='{"document_type":"IFU","source":"ui","owner":"Blatchford QARA"}',
                lines=4,
            )
            b_set_meta = gr.Button("Set metadata")
            b_set_meta.click(fn=ui_set_blob_metadata, inputs=[b_container, blob_name, b_meta_json], outputs=[b_action_out])

        # ----------------
        # Documents
        # ----------------
        with gr.Tab("Documents"):
            gr.Markdown(
                """
        ### Documents

        **Status legend**  
        🟢 INDEXED — indexed and up to date  
        🟠 STALE (REINDEX) — blob changed since last index  
        🔴 NOT INDEXED — blob exists but not indexed
        """
            )

            # Top controls
            with gr.Row():
                container = gr.Textbox(label="Blob container", value=DEFAULT_CONTAINER)
                refresh_docs_btn = gr.Button("Refresh documents + IDs")

            docs_df = gr.Dataframe(label="Documents (/documents)", interactive=False)

            # -----------------------
            # Bulk ingest (job-based)
            # -----------------------
            gr.Markdown("### Bulk ingest")

            with gr.Row():
                ingest_select = gr.Dropdown(
                    label="Select doc_ids to ingest",
                    choices=[],
                    multiselect=True,
                    value=[],
                    allow_custom_value=False,
                    scale=4,
                )
                ingest_ids_btn = gr.Button("Load IDs for bulk ingest", scale=1)

            ingest_ids = gr.Textbox(
                label="doc_ids (comma or newline separated) — optional",
                placeholder="BMK2IFU.pdf\nSAKLIFU.pdf",
                lines=4,
            )

            with gr.Row():
                ingest_use_selected_btn = gr.Button("Use selected IDs", scale=1)
                ingest_btn = gr.Button("Ingest (/documents/ingest)", scale=2)

            ingest_out = gr.Code(label="Bulk ingest response (JSON)", language="json")

            with gr.Row():
                ingest_job_id = gr.Textbox(label="job_id", placeholder="auto-filled after ingest", scale=4)
                poll_ingest_btn = gr.Button("Poll ingest job", scale=1)

            poll_ingest_out = gr.Code(label="Ingest job status (JSON)", language="json")

            # -----------------------
            # Single document ingest
            # -----------------------
            gr.Markdown("### Ingest single document")

            with gr.Row():
                single_doc_id = gr.Dropdown(
                    label="Select one doc_id",
                    choices=[],
                    multiselect=False,
                    value=None,
                    allow_custom_value=True,
                    scale=4,
                )
                single_ids_btn = gr.Button("Load IDs for single ingest", scale=1)
                ingest_one_btn = gr.Button("Ingest selected (single)", scale=2)

            single_ingest_out = gr.Code(label="Single ingest response (JSON)", language="json")

            with gr.Row():
                single_job_id = gr.Textbox(label="job_id (single)", placeholder="auto-filled after ingest", scale=4)
                poll_single_btn = gr.Button("Poll ingest job", scale=1)

            poll_single_out = gr.Code(label="Single ingest job status (JSON)", language="json")

            # ----------------
            # Document actions
            # ----------------
            gr.Markdown("### Document actions")

            with gr.Row():
                doc_id = gr.Dropdown(label="doc_id", choices=[], allow_custom_value=True)
                refresh_ids_btn = gr.Button("Load IDs only", scale=1)
                get_doc_btn = gr.Button("Get doc (/documents/{doc_id})", scale=2)

            doc_json = gr.Code(label="GetDocumentResponse (JSON)", language="json")

            with gr.Row():
                document_type = gr.Textbox(label="document_type", value="IFU", scale=2)
                delete_vectors_btn = gr.Button("Delete vectors (selected)", scale=2)

            action_out = gr.Code(label="Action output (JSON)", language="json")

            # ---------
            # Callbacks
            # ---------
            def _ids_update(c: str):
                ids = ui_list_document_ids(c)
                return gr.update(choices=ids, value=(ids[0] if ids else None))

            def _single_ids_update(c: str):
                ids = ui_list_document_ids(c)
                return gr.update(choices=ids, value=(ids[0] if ids else None))

            def _bulk_ids_update_not_indexed(c: str):
                # Use /documents so we can filter by is_indexed
                df = ui_list_documents(c)

                if df is None or df.empty:
                    return gr.update(choices=[], value=[])

                # Defensive: if is_indexed missing, treat as not indexed
                if "is_indexed" not in df.columns:
                    df["is_indexed"] = False

                # Only offer doc_ids that are NOT indexed
                candidates = df.loc[~df["is_indexed"].astype(bool), "doc_id"].dropna().astype(str).tolist()
                return gr.update(choices=candidates, value=[])

            def _selected_to_text(selected):
                selected = selected or []
                return "\n".join(selected)

            def _ingest_bulk_from_ui(c: str, selected, typed: str, doc_type: str) -> str:
                selected = selected or []
                doc_ids_text = "\n".join(selected) if selected else (typed or "")
                return ui_ingest(c, doc_ids_text, doc_type)  # should return JSON string incl job_id

            # -------------
            # Event wiring
            # -------------

            # Refresh docs table + dropdown ids together (doc actions only)
            refresh_docs_btn.click(
                fn=ui_refresh_documents_and_ids,
                inputs=[container],
                outputs=[docs_df, doc_id],
            )

            # Load IDs only (Document actions)
            refresh_ids_btn.click(
                fn=_ids_update,
                inputs=[container],
                outputs=[doc_id],
            )

            # Bulk ingest IDs: only NOT indexed
            ingest_ids_btn.click(
                fn=_bulk_ids_update_not_indexed,
                inputs=[container],
                outputs=[ingest_select],
            )

            ingest_use_selected_btn.click(
                fn=_selected_to_text,
                inputs=[ingest_select],
                outputs=[ingest_ids],
            )

            # Bulk ingest -> response + job_id textbox
            ingest_btn.click(
                fn=_ingest_bulk_from_ui,
                inputs=[container, ingest_select, ingest_ids, document_type],
                outputs=[ingest_out],
            ).then(
                fn=_extract_job_id,
                inputs=[ingest_out],
                outputs=[ingest_job_id],
            )

            # Bulk poll
            poll_ingest_btn.click(
                fn=ui_get_ingest_job,
                inputs=[ingest_job_id],
                outputs=[poll_ingest_out],
            )

            # Single IDs (explicit)
            single_ids_btn.click(
                fn=_single_ids_update,
                inputs=[container],
                outputs=[single_doc_id],
            )

            # Single ingest -> response + job_id textbox
            ingest_one_btn.click(
                fn=ui_ingest_one,
                inputs=[container, single_doc_id, document_type],
                outputs=[single_ingest_out],
            ).then(
                fn=_extract_job_id,
                inputs=[single_ingest_out],
                outputs=[single_job_id],
            )

            # Single poll
            poll_single_btn.click(
                fn=ui_get_ingest_job,
                inputs=[single_job_id],
                outputs=[poll_single_out],
            )

            # Document actions
            get_doc_btn.click(fn=ui_get_document, inputs=[container, doc_id], outputs=[doc_json])
            delete_vectors_btn.click(fn=ui_delete_vectors, inputs=[doc_id], outputs=[action_out])

        # ----------------
        # Stats
        # ----------------
        with gr.Tab("Stats"):
            with gr.Row():
                stats_container = gr.Textbox(label="Blob container", value=DEFAULT_CONTAINER)
                stats_btn = gr.Button("Refresh stats (/stats)")
            stats_json = gr.Code(label="Stats JSON", language="json")
            stats_docs_df = gr.Dataframe(label="Indexed documents", interactive=False)
            stats_blobs_df = gr.Dataframe(label="Blobs", interactive=False)
            stats_btn.click(fn=ui_get_stats, inputs=[stats_container], outputs=[stats_json, stats_docs_df, stats_blobs_df])

        # ----------------
        # Query
        # ----------------
        with gr.Tab("Query"):
            with gr.Row():
                q_container = gr.Textbox(label="Blob container", value=DEFAULT_CONTAINER)
                q_text = gr.Textbox(label="Query", placeholder="How do I align the prosthetic foot?")
            with gr.Row():
                q_n = gr.Slider(1, 20, value=5, step=1, label="n_results")
                q_lang = gr.Dropdown(label="Language", choices=LANG_CHOICES, value="en")
            q_where = gr.Textbox(label="where JSON (optional)", placeholder='{"doc_id":{"$eq":"BMK2IFU.pdf"}}', lines=2)
            q_btn = gr.Button("Run /query")
            q_results = gr.Dataframe(label="Query hits", interactive=False)
            q_btn.click(fn=ui_query, inputs=[q_container, q_text, q_n, q_lang, q_where], outputs=[q_results])

        # ----------------
        # Chat
        # ----------------
        with gr.Tab("Chat"):
            with gr.Row():
                c_container = gr.Textbox(label="Blob container", value=DEFAULT_CONTAINER)
                c_n = gr.Slider(1, 20, value=5, step=1, label="n_results")
            c_where = gr.Textbox(label="where JSON (optional)", placeholder='{"doc_id":{"$eq":"BMK2IFU.pdf"}}', lines=2)

            with gr.Row():
                c_temp = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="temperature")
                c_max = gr.Slider(64, 4096, value=512, step=64, label="max_tokens")
                c_tone = gr.Dropdown(
                    label="Tone",
                    choices=["neutral", "plain", "technical", "regulatory", "training"],
                    value="neutral",
                )
                c_lang = gr.Dropdown(label="Language", choices=LANG_CHOICES[1:], value="en")

            chat_messages_state = gr.State([])  # list[{role,content}]
            chat_api_state = gr.State([])  # list[{role,content}]

            chatbot = gr.Chatbot(label="Chat", height=420)
            question = gr.Textbox(label="Question", placeholder="Ask something about the IFU…")
            send_btn = gr.Button("Send")

            chat_debug = gr.Code(label="Chat debug (model/usage)", language="json")
            sources_df = gr.Dataframe(label="Sources", interactive=False)

            send_btn.click(
                fn=ui_chat,
                inputs=[
                    c_container,
                    question,
                    c_n,
                    c_lang,   # lang dropdown
                    c_where,
                    c_temp,
                    c_max,
                    chat_messages_state,
                    chat_api_state,
                    c_tone,
                ],
                outputs=[chatbot, chat_debug, sources_df, chat_api_state],
            ).then(
                fn=lambda chat_value: chat_value,
                inputs=[chatbot],
                outputs=[chat_messages_state],
            ).then(
                fn=lambda: "",
                outputs=[question],
            )

        # ----------------
        # Logs
        # ----------------
        with gr.Tab("Logs"):
            gr.Markdown("Reads your local log file (Option A).")
            with gr.Row():
                log_path = gr.Textbox(label="Log file path", value=LOG_FILE)
                tail_lines = gr.Slider(50, 2000, value=LOG_TAIL_LINES, step=50, label="Tail lines")
                refresh_logs_btn = gr.Button("Refresh logs")
            log_view = gr.Textbox(label="Logs", value="", lines=25, interactive=False)

            refresh_logs_btn.click(fn=tail_log_file, inputs=[log_path, tail_lines], outputs=[log_view])

            try:
                timer = gr.Timer(value=3.0)
                timer.tick(fn=tail_log_file, inputs=[log_path, tail_lines], outputs=[log_view])
            except Exception:
                pass

    return demo

