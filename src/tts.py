"""Text-to-Speech module – gTTS (online)."""
from __future__ import annotations

import io
from typing import Optional


def speak(text: str, engine: str = "gtts") -> Optional[bytes]:
    """Convert *text* to MP3 audio bytes via gTTS. Returns None on failure."""
    if not text or not text.strip():
        return None
    try:
        return _gtts(text)
    except Exception:
        return None


def tts_engines_available() -> dict[str, bool]:
    """Return which TTS backends can be imported."""
    try:
        from gtts import gTTS  # noqa: F401
        return {"gtts": True}
    except Exception:
        return {"gtts": False}


# ── Private helpers ──────────────────────────────────────────────────────────

def _gtts(text: str) -> bytes:
    from gtts import gTTS
    buf = io.BytesIO()
    tts = gTTS(text=text, lang="en", slow=False)
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()
