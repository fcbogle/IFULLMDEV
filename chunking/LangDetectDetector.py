# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-11
# Description: LangDetectDetector
# -----------------------------------------------------------------------------

from langdetect import detect_langs

class LangDetectDetector:
    def detect(self, text: str):
        if not text or len(text.strip()) < 20:
            return "und", 0.0, None

        try:
            detections = detect_langs(text)
            if not detections:
                return "und", 0.0, None

            top = detections[0] # most probable language from detections list
            lang_code = top.lang  # e.g. "en", "fr"
            confidence = top.prob  # 0.0–1.0
            script = "Latn" if lang_code in ("en", "fr", "de", "es", "it") else None
            return lang_code, confidence, script

        except Exception:
            return "und", 0.0, None
