# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-11
# Description: PDFPageExtractor
# -----------------------------------------------------------------------------
from io import BytesIO
from typing import List
from pypdf import PdfReader

class PDFPageExtractor:
    """
    Utility for extracting page text from PDF bytes.
    """
    def extract_pages(self, pdf_bytes: bytes) -> List[str]:
        reader = PdfReader(BytesIO(pdf_bytes))
        return [page.extract_text() or "" for page in reader.pages]