# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-08
# Description: IFUTextExtractor
# -----------------------------------------------------------------------------
import time
import fitz
from utility.logging_utils import get_class_logger

class IFUTextExtractor:
    def __init__(self, logger = None):
        self.logger = logger or get_class_logger(self.__class__)

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extracts text from a PDF document pages using page loop

        """
        start = time.time()
        text = ""
        try:
            with fitz.open(stream = pdf_bytes, filetype = "pdf") as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                elapsed = (time.time() - start) * 1000.0
                self.logger.info(
                    "Extracted text from PDF (%d pages, %.1f ms)", len(doc), elapsed
                )
                return text.strip()
        except Exception as e:
            self.logger.error("Failed to extract text from PDF: %s", e)
            raise


