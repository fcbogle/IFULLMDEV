# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-11
# Description: LangDetectDetector
# -----------------------------------------------------------------------------

from langdetect import detect_langs, DetectorFactory
import langid
import re

# determinism for langdetect
DetectorFactory.seed = 0

_WS = re.compile(r"\s+")

def _prep(s: str | None) -> str:
    if not s:
        return ""
    return _WS.sub(" ", s).strip()

class LangDetectDetector:
    """
    Robust language detector:
      - cleans text
      - tries page-level fallback when chunk is short
      - primary: langdetect
      - secondary: langid (for short/noisy text)
    """
    def __init__(self, *, min_len: int = 40, min_conf: float = 0.60):
        self.min_len = min_len
        self.min_conf = min_conf

    def _detect_langdetect(self, text: str):
        text = _prep(text)
        if len(text) < self.min_len:
            return None
        try:
            dets = detect_langs(text)
            if not dets:
                return None
            top = dets[0]
            lang = top.lang
            conf = float(top.prob)
            script = "Latn"  # simple heuristic
            return lang, conf, script
        except Exception:
            return None

    def _detect_langid(self, text: str):
        text = _prep(text)
        if len(text) < max(20, self.min_len // 2):
            return None

        code, score = langid.classify(text)

        import math
        s = float(score)

        # Numerically stable sigmoid
        if s >= 0:
            z = math.exp(-s)
            conf = 1.0 / (1.0 + z)
        else:
            z = math.exp(s)
            conf = z / (1.0 + z)

        script = "Latn"
        return code, conf, script

    def detect(self, text: str, fallback: str | None = None):
        # 1) Try langdetect on chunk
        got = self._detect_langdetect(text)
        if got:
            lang, conf, script = got
            if conf >= self.min_conf:
                return lang, conf, script

        # 2) If chunk weak/short, try fallback (e.g., full page)
        if fallback:
            got = self._detect_langdetect(fallback)
            if got:
                lang, conf, script = got
                if conf >= self.min_conf:
                    return lang, conf, script

        # 3) Secondary: langid on the best available text
        for candidate in (text or "", fallback or ""):
            got = self._detect_langid(candidate)
            if got:
                return got

        # 4) Give up
        return "und", 0.0, None
