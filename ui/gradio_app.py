# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-28
# Description: gradio_app.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import requests

import inspect, os


# Environment configuration
API_BASE_URL = os.getenv("IFU_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LOG_FILE = os.getenv("IFU_LOG_FILE", "./logs/ifullmdev.log")
LOG_TAIL_LINES = int(os.getenv("IFU_UI_LOG_TAIL_LINES", "400"))
TIMEOUT_SECONDS = int(os.getenv("IFU_UI_TIMEOUT_SECONDS", "30"))

DEFAULT_CONTAINER = os.getenv("IFU_DEFAULT_CONTAINER", "ifu-docs-test")


# Small URL helpers
def _url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def _get(path: str, params: Optional[dict] = None) -> Dict[str, Any]:
    try:
        r = requests.get(_url(path), params=params, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}



def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(_url(path), json=payload, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}


def _delete(path: str) -> Dict[str, Any]:
    try:
        r = requests.delete(_url(path), timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}


# Log tailing for UI
def tail_log_file(path: str, n_lines: int = 200) -> str:
    """
    Tail last n_lines from a local log file path.
    (Option A: UI reads the file directly, no log API needed.)
    """
    try:
        if not path or not os.path.exists(path):
            return f"[log] file not found: {path}"
        with open(path, "rb") as f:
            # simple approach: read entire file if small-ish; ok for dev
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()[-n_lines:]
        return "\n".join(lines)
    except Exception as e:
        return f"[log] failed to read log file: {e}"


# Document UI functions
def ui_get_stats(container: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    container = (container or "").strip() or DEFAULT_CONTAINER
    stats = _get("/stats", params={"blob_container": container})

    # Pretty JSON for display
    stats_json = json.dumps(stats, indent=2)

    # Tables
    docs_df = pd.DataFrame(stats.get("documents") or [])
    blobs_df = pd.DataFrame(stats.get("blobs") or [])
    return stats_json, docs_df, blobs_df


def ui_list_documents(container: str) -> pd.DataFrame:
    container = (container or "").strip() or DEFAULT_CONTAINER
    out = _get("/documents", params={"container": container})
    docs = out.get("documents") or []
    return pd.DataFrame(docs)


def ui_list_document_ids(container: str) -> List[str]:
    container = (container or "").strip() or DEFAULT_CONTAINER
    out = _get("/documents/ids", params={"container": container})
    return out.get("doc_ids") or []


def ui_get_document(container: str, doc_id: str) -> str:
    container = (container or "").strip() or DEFAULT_CONTAINER
    doc_id = (doc_id or "").strip()
    out = _get(f"/documents/{doc_id}", params={"container": container})
    return json.dumps(out, indent=2)


def ui_reindex(container: str, doc_id: str, document_type: str) -> str:
    container = (container or "").strip() or DEFAULT_CONTAINER
    doc_id = (doc_id or "").strip()
    document_type = (document_type or "IFU").strip() or "IFU"
    out = _post(f"/documents/{doc_id}/reindex", payload={}, ) if False else None  # placeholder

    # Your router defines POST /documents/{doc_id}/reindex with query params, not JSON body.
    # So use requests.post directly:
    r = requests.post(
        _url(f"/documents/{doc_id}/reindex"),
        params={"container": container, "document_type": document_type},
        timeout=TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    out = r.json()
    return json.dumps(out, indent=2)


def ui_ingest(container: str, doc_ids_text: str, document_type: str) -> str:
    container = (container or "").strip() or DEFAULT_CONTAINER
    document_type = (document_type or "IFU").strip() or "IFU"

    # comma / newline separated list
    raw_ids = (doc_ids_text or "").replace(",", "\n").splitlines()
    doc_ids = [x.strip() for x in raw_ids if x.strip()]

    payload = {"container": container, "doc_ids": doc_ids, "document_type": document_type}
    out = _post("/documents/ingest", payload=payload)
    return json.dumps(out, indent=2)


def ui_delete_vectors(doc_id: str) -> str:
    doc_id = (doc_id or "").strip()
    out = _delete(f"/documents/{doc_id}/vectors")
    return json.dumps(out, indent=2)


# Query UI functions
def ui_query(container: str, query_text: str, n_results: int, lang_en_only: bool, where_json: str) -> pd.DataFrame:
    query_text = (query_text or "").strip()
    if not query_text:
        return pd.DataFrame([{"error": "query must not be empty"}])

    where: Optional[Dict[str, Any]] = None

    # Base where from JSON (optional)
    where_json = (where_json or "").strip()
    if where_json:
        try:
            where = json.loads(where_json)
        except Exception as e:
            return pd.DataFrame([{"error": f"invalid where JSON: {e}"}])

    # Merge in english filter if requested
    if lang_en_only:
        lang_filter = {"lang": {"$eq": "en"}}
        if where is None:
            where = lang_filter
        else:
            # naive merge; if user sets lang too, user wins
            where = {**lang_filter, **where}

    payload = {
        "query": query_text,
        "n_results": int(n_results),
        "where": where,
        "include_text": True,
        "include_scores": True,
        "include_metadata": True,
    }
    out = _post("/query", payload=payload)
    rows = [h for h in (out.get("results") or [])]
    return pd.DataFrame(rows)


# Chat UI functions
def ui_chat(
        container: str,
        question: str,
        n_results: int,
        lang_en_only: bool,
        where_json: str,
        temperature: float,
        max_tokens: int,
        history: List[List[str]],  # Gradio Chatbot format: [[user, assistant], ...]
) -> Tuple[List[List[str]], str, pd.DataFrame]:
    question = (question or "").strip()
    if not question:
        return history, "", pd.DataFrame()

    where: Optional[Dict[str, Any]] = None
    where_json = (where_json or "").strip()
    if where_json:
        try:
            where = json.loads(where_json)
        except Exception as e:
            # show error as assistant response
            history = history + [[question, f"Invalid where JSON: {e}"]]
            return history, "", pd.DataFrame()

    if lang_en_only:
        lang_filter = {"lang": {"$eq": "en"}}
        where = lang_filter if where is None else {**lang_filter, **where}

    # Convert history to OpenAI-style history for your API (optional)
    # Your ChatRequest.history is Optional[List[Dict[str,str]]]
    api_history: List[Dict[str, str]] = []
    for u, a in history:
        if u:
            api_history.append({"role": "user", "content": u})
        if a:
            api_history.append({"role": "assistant", "content": a})

    payload = {
        "question": question,
        "n_results": int(n_results),
        "where": where,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "history": api_history or None,
    }

    out = _post("/chat", payload=payload)
    answer = out.get("answer") or ""

    # Add to chat UI history
    history = history + [[question, answer]]

    # Sources table
    sources = out.get("sources") or []
    sources_df = pd.DataFrame(sources)

    # Optional debug panel text
    debug = json.dumps(
        {"model": out.get("model"), "usage": out.get("usage"), "n_sources": len(sources)},
        indent=2,
    )

    return history, debug, sources_df


# Build Gradio UI
def build_gradio_app(api_base_url: str = API_BASE_URL) -> gr.Blocks:
    global API_BASE_URL
    API_BASE_URL = api_base_url

    with gr.Blocks(title="IFULLMDEV UI", analytics_enabled=False) as demo:
        gr.Markdown(
            f"""
# IFULLMDEV (RAG + Chat)
**API:** `{API_BASE_URL}`  
**Default container:** `{DEFAULT_CONTAINER}`  
**Log file:** `{LOG_FILE}`
"""
        )

        with gr.Tab("Documents"):
            with gr.Row():
                container = gr.Textbox(label="Blob container", value=DEFAULT_CONTAINER)
                refresh_docs_btn = gr.Button("Refresh documents")
            docs_df = gr.Dataframe(label="Documents (/documents)", interactive=False)
            refresh_docs_btn.click(fn=ui_list_documents, inputs=[container], outputs=[docs_df])

            gr.Markdown("### Document actions")
            with gr.Row():
                doc_id = gr.Dropdown(label="doc_id", choices=[], allow_custom_value=True)
                refresh_ids_btn = gr.Button("Load IDs (/documents/ids)")
                get_doc_btn = gr.Button("Get doc (/documents/{doc_id})")
            doc_json = gr.Code(label="GetDocumentResponse (JSON)", language="json")

            def _reload_ids(c: str) -> List[str]:
                return ui_list_document_ids(c)

            refresh_ids_btn.click(fn=_reload_ids, inputs=[container], outputs=[doc_id])
            get_doc_btn.click(fn=ui_get_document, inputs=[container, doc_id], outputs=[doc_json])

            with gr.Row():
                document_type = gr.Textbox(label="document_type", value="IFU")
                reindex_btn = gr.Button("Reindex selected")
                delete_vectors_btn = gr.Button("Delete vectors (selected)")
            action_out = gr.Code(label="Action output (JSON)", language="json")
            reindex_btn.click(fn=ui_reindex, inputs=[container, doc_id, document_type], outputs=[action_out])
            delete_vectors_btn.click(fn=ui_delete_vectors, inputs=[doc_id], outputs=[action_out])

            gr.Markdown("### Bulk ingest")
            ingest_ids = gr.Textbox(
                label="doc_ids (comma or newline separated)",
                placeholder="BMK2IFU.pdf\nSAKLIFU.pdf",
                lines=4,
            )
            ingest_btn = gr.Button("Ingest (/documents/ingest)")
            ingest_out = gr.Code(label="Ingest output (JSON)", language="json")
            ingest_btn.click(fn=ui_ingest, inputs=[container, ingest_ids, document_type], outputs=[ingest_out])

        with gr.Tab("Stats"):
            with gr.Row():
                stats_container = gr.Textbox(label="Blob container", value=DEFAULT_CONTAINER)
                stats_btn = gr.Button("Refresh stats (/stats)")
            stats_json = gr.Code(label="Stats JSON", language="json")
            stats_docs_df = gr.Dataframe(label="Indexed documents (from vector store)", interactive=False)
            stats_blobs_df = gr.Dataframe(label="Blobs (from storage)", interactive=False)
            stats_btn.click(fn=ui_get_stats, inputs=[stats_container],
                            outputs=[stats_json, stats_docs_df, stats_blobs_df])

        with gr.Tab("Query"):
            with gr.Row():
                q_container = gr.Textbox(label="Blob container (for your context)", value=DEFAULT_CONTAINER)
                q_text = gr.Textbox(label="Query", placeholder="How do I align the prosthetic foot?")
            with gr.Row():
                q_n = gr.Slider(1, 20, value=5, step=1, label="n_results")
                q_en = gr.Checkbox(value=True, label="English only (where: lang == en)")
            q_where = gr.Textbox(label="where JSON (optional)", placeholder='{"doc_id":{"$eq":"BMK2IFU.pdf"}}', lines=2)
            q_btn = gr.Button("Run /query")
            q_results = gr.Dataframe(label="Query hits", interactive=False)
            q_btn.click(fn=ui_query, inputs=[q_container, q_text, q_n, q_en, q_where], outputs=[q_results])

        with gr.Tab("Chat"):
            with gr.Row():
                c_container = gr.Textbox(label="Blob container (for your context)", value=DEFAULT_CONTAINER)
                c_n = gr.Slider(1, 20, value=5, step=1, label="n_results")
                c_en = gr.Checkbox(value=True, label="English only (where: lang == en)")
            c_where = gr.Textbox(label="where JSON (optional)", placeholder='{"doc_id":{"$eq":"BMK2IFU.pdf"}}', lines=2)

            with gr.Row():
                c_temp = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="temperature")
                c_max = gr.Slider(64, 4096, value=512, step=64, label="max_tokens")

            chatbot = gr.Chatbot(label="Chat", height=420)
            question = gr.Textbox(label="Question", placeholder="Ask something about the IFU…")
            send_btn = gr.Button("Send")

            chat_debug = gr.Code(label="Chat debug (model/usage)", language="json")
            sources_df = gr.Dataframe(label="Sources", interactive=False)

            send_btn.click(
                fn=ui_chat,
                inputs=[c_container, question, c_n, c_en, c_where, c_temp, c_max, chatbot],
                outputs=[chatbot, chat_debug, sources_df],
            ).then(lambda: "", outputs=[question])  # clear input after send

        with gr.Tab("Logs"):
            gr.Markdown(
                "Reads your local log file (Option A). "
                "If you later deploy remotely, we can switch to a log API or a websocket."
            )
            with gr.Row():
                log_path = gr.Textbox(label="Log file path", value=LOG_FILE)
                tail_lines = gr.Slider(50, 2000, value=LOG_TAIL_LINES, step=50, label="Tail lines")
                refresh_logs_btn = gr.Button("Refresh logs")
            log_view = gr.Textbox(label="Logs", value="", lines=25, interactive=False)

            refresh_logs_btn.click(fn=tail_log_file, inputs=[log_path, tail_lines], outputs=[log_view])

            # Optional: auto-refresh every few seconds (works on recent Gradio versions)
            try:
                timer = gr.Timer(value=3.0)  # seconds
                timer.tick(fn=tail_log_file, inputs=[log_path, tail_lines], outputs=[log_view])
            except Exception:
                pass

        # Initial load
        refresh_docs_btn.click(fn=ui_list_documents, inputs=[container], outputs=[docs_df])

    return demo


if __name__ == "__main__":
    import os
    import threading
    import uvicorn

    API_HOST = os.getenv("IFU_API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("IFU_API_PORT", "8000"))

    UI_HOST = os.getenv("IFU_UI_HOST", "127.0.0.1")
    UI_PORT = int(os.getenv("IFU_UI_PORT", "7860"))

    def run_api() -> None:
        # IMPORTANT: adjust this import path to wherever your FastAPI app lives
        # e.g. "api.main:app" or "main:app"
        uvicorn.run(
            "api.main:app",
            host=API_HOST,
            port=API_PORT,
            log_level="info",
            reload=False,  # set True only if you accept double-reload weirdness
        )

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    print("GRADIO_APP LOADED FROM:", __file__)
    print("build_gradio_app signature:",
          inspect.signature(build_gradio_app) if "build_gradio_app" in globals() else "not yet defined")

    demo = build_gradio_app()
    demo.launch(server_name=UI_HOST, server_port=UI_PORT)


