# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: main.py
# -----------------------------------------------------------------------------
import logging
import os

import gradio as gr
from fastapi import FastAPI

from api.routers import health, ifu_stats, query, documents, chat, blobs
from ui.gradio_app import build_gradio_app

CUSTOM_CSS = """
/* =========================================================
   DataFrame styling (compact, readable tables)
   ========================================================= */

.gr-dataframe,
.gr-dataframe table,
.gr-dataframe .cell,
.gr-dataframe td,
.gr-dataframe th {
  font-size: 10px !important;
  line-height: 1.2 !important;
}

/* Tighten row padding so rows are shorter */
.gr-dataframe td,
.gr-dataframe th {
  padding-top: 4px !important;
  padding-bottom: 4px !important;
}

/* Prevent status / icon column from expanding */
.gr-dataframe td:first-child,
.gr-dataframe th:first-child {
  white-space: nowrap !important;
}


/* =========================================================
   IFULLMDEV Header
   ========================================================= */

.ifu-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 18px;
  margin-bottom: 14px;

  border-radius: 14px;
  border: 1px solid rgba(120, 120, 120, 0.25);
  background: rgba(250, 250, 250, 0.9);
}

.ifu-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.ifu-logo {
  width: 38px;
  height: 38px;

  display: grid;
  place-items: center;

  font-weight: 700;
  border-radius: 10px;
  border: 1px solid rgba(120, 120, 120, 0.25);
  background: #ffffff;
}

.ifu-title h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
}

.ifu-title p {
  margin: 2px 0 0 0;
  font-size: 12px;
  opacity: 0.75;
}

.ifu-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

.ifu-chip {
  font-size: 12px;
  padding: 6px 10px;

  border-radius: 999px;
  border: 1px solid rgba(120, 120, 120, 0.25);
  background: #ffffff;

  white-space: nowrap;
}

.ifu-chip code {
  font-size: 12px;
}

.ifu-logo-img {
  width: 60px;
  height: 60px;
  object-fit: contain;
  border-radius: 10px;
  border: 1px solid rgba(120,120,120,.25);
  background: white;
  padding: 4px;
}
"""


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", )
app = FastAPI(title="IFULLMDEV API")
app.include_router(health.router)
app.include_router(ifu_stats.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(blobs.router)

# Mount Gradio (served by the SAME uvicorn process/port)
API_BASE_URL = os.getenv("IFU_API_BASE_URL", "http://127.0.0.1:8000")
gradio_blocks = build_gradio_app(api_base_url=API_BASE_URL)
app = gr.mount_gradio_app(app, gradio_blocks, path="/ui", css=CUSTOM_CSS)
