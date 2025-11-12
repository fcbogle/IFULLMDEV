# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-08
# Description: IFUTextExtractor
# -----------------------------------------------------------------------------
import time
from typing import List

import fitz
from utility.logging_utils import get_class_logger

class IFUTextExtractor:
    def __init__(self, logger = None):
        self.logger = logger or get_class_logger(self.__class__)

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> List[str]:
        """
        Extracts text from PDF bytes using PyMuPDF (fitz).
        Returns: list of page texts (page 1 = index 0)
        """
        start = time.time()
        page_texts: List[str] = []
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    text = page.get_text("text") or ""
                    page_texts.append(text.strip())

                elapsed = (time.time() - start) * 1000.0
                self.logger.info(
                    "Extracted text from PDF (%d pages, %.1f ms)", len(doc), elapsed
                )

            return page_texts

        except Exception as e:
            self.logger.error("Failed to extract text from PDF: %s", e)
            raise


