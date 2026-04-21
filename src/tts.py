"""Text-to-Speech module – gTTS (online)."""
from __future__ import annotations

import io
import re
from typing import Optional


_ABBREVIATIONS = {
    "CA": "Canada",
    "U.S.A.": "United States",
    "USA": "United States",
    "U.S.": "United States",
    "US": "United States",
    "UK": "United Kingdom",
}


def normalize_tts_text(text: str) -> str:
    """Normalize text before TTS so it sounds more natural when spoken."""
    if not text:
        return ""

    text = str(text)

    # Expand a few common abbreviations used in the app's answers.
    for token, expanded in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(token)}\b", expanded, text)

    # Convert compact timer/duration fragments like "2m 0s" or "2 m 0 s".
    def _duration_repl(match: re.Match[str]) -> str:
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        parts: list[str] = []
        if minutes:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
        if not parts:
            return "0 seconds"
        return " and ".join(parts)

    text = re.sub(
        r"\b(?P<minutes>\d+)\s*m\s*(?P<seconds>\d+)\s*s\b",
        _duration_repl,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?P<minutes>\d+)\s*min(?:utes?)?\s*(?P<seconds>\d+)\s*sec(?:onds?)?\b",
        _duration_repl,
        text,
        flags=re.IGNORECASE,
    )

    # Read rating-style fractions naturally, e.g. "8.8/10" -> "8.8 out of 10".
    # Avoid converting slash-separated dates like "3/10/2026".
    text = re.sub(
        r"\b(?P<num>\d+(?:\.\d+)?)\s*/\s*(?P<den>10)\b(?!\s*/\s*\d)",
        r"\g<num> out of \g<den>",
        text,
    )

    # Spell out 19xx years so TTS doesn't read them digit-by-digit.
    def _year_repl(match: re.Match[str]) -> str:
        year = int(match.group(0))
        if 1900 <= year <= 1999:
            last_two = year % 100
            tens, ones = divmod(last_two, 10)
            tens_words = {
                0: "oh",
                1: "ten",
                2: "twenty",
                3: "thirty",
                4: "forty",
                5: "fifty",
                6: "sixty",
                7: "seventy",
                8: "eighty",
                9: "ninety",
            }
            if last_two == 0:
                return "nineteen hundred"
            if last_two < 10:
                return f"nineteen oh {last_two}"
            if ones == 0:
                return f"nineteen {tens_words[tens]}"
            return f"nineteen {tens_words[tens]} {ones}"
        return match.group(0)

    text = re.sub(r"\b(19\d{2})\b", _year_repl, text)

    return text.strip()


def speak(text: str, engine: str = "gtts") -> Optional[bytes]:
    """Convert *text* to MP3 audio bytes via gTTS. Returns None on failure."""
    text = normalize_tts_text(text)
    if not text:
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
