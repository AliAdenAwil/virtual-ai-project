"""LLM-based answer generation using Groq (free tier, no credit card required).

Strategy (per professor's guidance — explore where each approach excels):
- Control intents (Play/Pause/Stop/etc.) and SetTimer → always use templates.
  These are functional commands where the template message IS the answer.
- All informational/conversational intents → use LLM when enabled.
  LLM receives structured fulfillment data and crafts a natural spoken response.

Model: llama-3.3-70b-versatile via Groq (fast, free, excellent quality).
Get a free API key at https://console.groq.com — no credit card required.

Falls back to None on any failure (missing key, network error, import error),
so the caller silently falls back to the template-based answer.
"""
from __future__ import annotations

import json
import os
from typing import Optional

# These intents produce functional command confirmations — templates win here.
# The "message" from the media controller is the answer; LLM adds no value.
_TEMPLATE_ONLY_INTENTS = {
    "SetTimer",
    "PlayMusic",
    "PlayMovie",
    "PauseMedia",
    "ResumeMedia",
    "StopMedia",
    "NextTrack",
    "ChangeVolume",
    "AddToWatchlist",
    "AddToPlaylist",
    "ShufflePlaylist",
}

# Keys to strip from fulfillment data before sending to LLM
_STRIP_KEYS = {"thread", "daily", "hourly", "canned"}

# For OMDb responses, keep only the most useful fields to stay within token limits
_OMDB_KEEP = {
    "Title", "Year", "Genre", "Director", "Actors", "Plot",
    "imdbRating", "Metascore", "imdbVotes", "Runtime",
    "Response", "Error", "Season", "Episodes", "Ratings",
}


def _trim_fulfillment(fr: dict) -> dict:
    """Return a trimmed copy of fulfillment data safe to send to the LLM."""
    trimmed: dict = {}
    for k, v in fr.items():
        if k in _STRIP_KEYS or callable(v):
            continue
        if isinstance(v, str) and len(v) > 500:
            trimmed[k] = v[:500] + "…"
        else:
            trimmed[k] = v

    # If it looks like an OMDb response, drop unused fields
    if "imdbID" in trimmed or "Title" in trimmed:
        trimmed = {k: v for k, v in trimmed.items() if k in _OMDB_KEEP}

    return trimmed


_SYSTEM_PROMPT = (
    "You are Atlas, a friendly and concise voice assistant. "
    "Respond in 1–2 natural spoken sentences. "
    "Use only the data provided — never invent facts or ratings. "
    "If the data contains an error field, apologise briefly and state the reason. "
    "Do not start with 'Atlas here' or repeat the intent name. "
    "Sound warm and helpful, like a knowledgeable friend."
)

_GROQ_MODEL = "llama-3.3-70b-versatile"


def generate_answer_llm(
    intent: str,
    fulfillment_result: dict,
    slots: dict,
    question: str = "",
) -> Optional[str]:
    """
    Generate a spoken answer using Llama 3.3 70B via Groq (free tier).

    Returns a string on success, or None if:
    - The intent is in _TEMPLATE_ONLY_INTENTS
    - GROQ_API_KEY is not set
    - The `groq` package is not installed
    - Any API / network error occurs
    """
    if intent in _TEMPLATE_ONLY_INTENTS:
        return None

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    fr_trimmed = _trim_fulfillment(fulfillment_result)

    question_line = f"User asked: \"{question}\"\n" if question.strip() else ""
    user_prompt = (
        f"{question_line}"
        f"Intent: {intent}\n"
        f"Slots: {json.dumps(slots, ensure_ascii=False)}\n"
        f"Data from APIs: {json.dumps(fr_trimmed, ensure_ascii=False, default=str)}\n\n"
        "Generate a natural spoken response for Atlas the voice assistant."
    )

    try:
        client = Groq(api_key=api_key)
        chat = client.chat.completions.create(
            model=_GROQ_MODEL,
            max_tokens=180,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = chat.choices[0].message.content.strip()
        return text if text else None
    except Exception:
        return None
