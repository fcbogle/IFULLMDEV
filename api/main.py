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
   Global knobs (easy to tweak)
   ========================================================= */
:root {
  --ifu-df-font-size: 10px;
  --ifu-df-line-height: 1.2;
  --ifu-df-pad-y: 4px;

  --ifu-header-radius: 14px;
  --ifu-border: 1px solid rgba(120, 120, 120, 0.25);
}


/* =========================================================
   DataFrame styling (compact, readable tables)
   Notes:
   - Gradio DataFrame markup can vary by version.
   - These selectors intentionally “over-match” a bit.
   ========================================================= */

/* Outer dataframe container(s) */
.gr-dataframe,
.gr-dataframe * {
  font-size: var(--ifu-df-font-size) !important;
  line-height: var(--ifu-df-line-height) !important;
}

/* Table cells (padding/row height) */
.gr-dataframe td,
.gr-dataframe th {
  padding-top: var(--ifu-df-pad-y) !important;
  padding-bottom: var(--ifu-df-pad-y) !important;
}

/* Prevent status/icon column from expanding */
.gr-dataframe td:first-child,
.gr-dataframe th:first-child {
  white-space: nowrap !important;
}

/* Optional: keep long text from exploding column widths */
.gr-dataframe td,
.gr-dataframe th {
  max-width: 520px;
  overflow: hidden;
  text-overflow: ellipsis;
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

  border-radius: var(--ifu-header-radius);
  border: var(--ifu-border);
  background: rgba(250, 250, 250, 0.9);
}

.ifu-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* If you use text-only logo block */
.ifu-logo {
  width: 38px;
  height: 38px;

  display: grid;
  place-items: center;

  font-weight: 700;
  border-radius: 10px;
  border: var(--ifu-border);
  background: #ffffff;
}

/* Title */
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

/* Chips on the right */
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
  border: var(--ifu-border);
  background: #ffffff;

  white-space: nowrap;
}

.ifu-chip code {
  font-size: 12px;
}

/* If you use an <img class="ifu-logo-img" ...> */
.ifu-logo-img {
  width: 60px;
  height: 60px;
  object-fit: contain;

  border-radius: 10px;
  border: var(--ifu-border);
  background: #ffffff;
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
