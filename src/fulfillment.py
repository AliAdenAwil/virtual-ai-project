"""Fulfillment module – calls external APIs and the media-center controller."""
from __future__ import annotations

import re
import threading
import time
from datetime import date as _date, timedelta as _timedelta
from typing import Optional

import requests

from src.control_system import MediaCenterController

# ── API keys / base URLs ────────────────────────────────────────────────────
import os as _os
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass
OMDB_API_KEY = _os.environ.get("OMDB_API_KEY", "")
OMDB_BASE_URL = "http://www.omdbapi.com/"
MB_BASE_URL = "https://musicbrainz.org/ws/2/"
MB_HEADERS = {"User-Agent": "VirtualAssistant/1.0 (csi5180-group14; student-project)"}
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# ── In-process timer registry ───────────────────────────────────────────────
_timers: dict[str, dict] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Movies & TV  (OMDb)
# ══════════════════════════════════════════════════════════════════════════════

def search_movie_by_title(
    title: str,
    year: Optional[str] = None,
    media_type: Optional[str] = None,
) -> dict:
    params: dict = {"apikey": OMDB_API_KEY, "t": title, "r": "json"}
    if year:
        params["y"] = _extract_year(year)
    if media_type and media_type in ("movie", "series", "episode"):
        params["type"] = media_type
    return _omdb_get(params) or _canned_movie(title)


def search_by_keyword(
    search_term: str,
    media_type: Optional[str] = None,
    year: Optional[str] = None,
) -> dict:
    params: dict = {"apikey": OMDB_API_KEY, "s": search_term, "r": "json"}
    if media_type and media_type in ("movie", "series", "episode"):
        params["type"] = media_type
    if year:
        params["y"] = _extract_year(year)
    return _omdb_get(params) or {"Search": [], "Response": "False", "Error": "API unavailable"}


def get_ratings_and_score(title: str, year: Optional[str] = None) -> dict:
    return search_movie_by_title(title, year=year)


def get_series_season_info(title: str, season: str) -> dict:
    season_num = _extract_digits(season) or "1"
    params = {"apikey": OMDB_API_KEY, "t": title, "Season": season_num, "r": "json"}
    return _omdb_get(params) or {"Response": "False", "Error": "API unavailable"}


def recommend_similar_by_keyword(
    search_term: str,
    media_type: Optional[str] = None,
    year: Optional[str] = None,
) -> dict:
    return search_by_keyword(search_term, media_type=media_type, year=year)


# ══════════════════════════════════════════════════════════════════════════════
# Music  (MusicBrainz)
# ══════════════════════════════════════════════════════════════════════════════

def search_artist_by_name(artist_name: str) -> dict:
    params = {"query": f"artist:{artist_name}", "fmt": "json", "limit": 3}
    return _mb_get("artist/", params) or {"artists": []}


def search_song_by_title(
    song_title: str, artist_name: Optional[str] = None
) -> dict:
    query = f"recording:{song_title}"
    if artist_name:
        query += f" AND artist:{artist_name}"
    params = {"query": query, "fmt": "json", "limit": 3}
    return _mb_get("recording/", params) or {"recordings": []}


def search_album_by_title(
    album_title: str, artist_name: Optional[str] = None
) -> dict:
    query = f"release:{album_title}"
    if artist_name:
        query += f" AND artist:{artist_name}"
    params = {"query": query, "fmt": "json", "limit": 3}
    return _mb_get("release/", params) or {"releases": []}


def browse_artist_albums(
    artist_name: str, release_type: Optional[str] = None
) -> dict:
    query = f"artist:{artist_name} AND primarytype:album"
    params = {"query": query, "fmt": "json", "limit": 10}
    return _mb_get("release-group/", params) or {"release-groups": []}


def search_music_by_keyword(search_term: str) -> dict:
    params = {"query": search_term, "fmt": "json", "limit": 5}
    return _mb_get("recording/", params) or {"recordings": []}


def get_track_artist(song_title: str) -> dict:
    return search_song_by_title(song_title)


# ══════════════════════════════════════════════════════════════════════════════
# Weather  (Open-Meteo, no key required)
# ══════════════════════════════════════════════════════════════════════════════

def _get_current_location() -> str:
    """Return city name from IP geolocation, fallback to 'Ottawa'."""
    try:
        data = requests.get("https://ipinfo.io/json", timeout=4).json()
        return data.get("city") or "Ottawa"
    except Exception:
        return "Ottawa"


def _resolve_date(date_str: Optional[str]) -> _date:
    """Resolve a relative date string to a calendar date. Defaults to today."""
    today = _date.today()
    if not date_str:
        return today
    s = date_str.lower().strip()
    if not s or s == "today":
        return today
    if s == "tomorrow":
        return today + _timedelta(days=1)
    if "weekend" in s:
        days_until_sat = (5 - today.weekday()) % 7
        return today + _timedelta(days=days_until_sat if days_until_sat > 0 else 7)
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for day_name, day_num in day_map.items():
        if day_name in s:
            delta = (day_num - today.weekday()) % 7
            if delta == 0:
                delta = 7  # "next Monday" when today is Monday
            if "next" in s:
                delta = (day_num - today.weekday()) % 7
                if delta <= 0:
                    delta += 7
                delta += 7
            return today + _timedelta(days=delta)
    return today


def _clean_location(location: str) -> str:
    """Normalize user-provided location text for geocoding."""
    loc = location or ""
    loc = (
        loc.replace("\u201c", '"').replace("\u201d", '"')
           .replace("\u2018", "'").replace("\u2019", "'")
    )
    loc = re.sub(r"\s+", " ", loc).strip()
    loc = loc.strip(" .,!?:;\"'`")
    loc = re.sub(r"[?!.,;:\"'`]+$", "", loc).strip()
    return loc


def get_weather(location: str, date: Optional[str] = None) -> dict:
    try:
        location = _clean_location(location)
        if not location:
            return {"error": "Location is empty."}

        # Split query into parts: city is first, rest are disambiguation hints
        # e.g. "Halifax, Nova Scotia, Canada" → city="Halifax", hints=["Nova Scotia","Canada"]
        parts = [p.strip() for p in location.split(",")]
        city = parts[0]
        hints = [h.lower() for h in parts[1:]]  # province, country, continent, etc.

        # Search with the city name, fetch up to 10 results for disambiguation
        geo = requests.get(
            GEOCODING_URL,
            params={"name": city, "count": 10, "format": "json"},
            timeout=8,
        ).json()
        results = geo.get("results") or []

        # If no results even for just the city, fail
        if not results:
            return {"error": f"Location '{location}' not found."}

        def _hint_score(result: dict) -> int:
            """Score a result by how many of the user's hints it matches."""
            haystack = " ".join([
                (result.get("name") or ""),
                (result.get("admin1") or ""),       # province / state
                (result.get("admin2") or ""),       # county / district
                (result.get("country") or ""),
                (result.get("country_code") or ""),
                (result.get("timezone") or ""),
            ]).lower()
            return sum(1 for h in hints if h in haystack)

        if hints:
            # Pick highest hint score, break ties by population
            r = max(results, key=lambda x: (_hint_score(x), x.get("population") or 0))
        else:
            # No hints — pick by population (most prominent city with that name)
            r = max(results, key=lambda x: x.get("population") or 0)

        lat, lon = r["latitude"], r["longitude"]
        # Build a readable location name
        loc_name = r.get("name", city)
        admin1 = r.get("admin1", "")
        country = r.get("country", "")
        if admin1 and admin1.lower() != loc_name.lower():
            loc_name = f"{loc_name}, {admin1}"
        elif country:
            loc_name = f"{loc_name}, {country}"

        resolved = _resolve_date(date)
        today = _date.today()
        is_today = (resolved == today)

        # Always fetch both current weather (for today) and daily forecast
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,weathercode"
            ),
            "forecast_days": 8,
            "timezone": "auto",
        }
        weather = requests.get(OPENMETEO_URL, params=params, timeout=8).json()
        weather["location_name"] = loc_name
        weather["requested_date"] = date or "today"
        weather["resolved_date"] = resolved.isoformat()
        weather["is_today"] = is_today

        # Extract the specific day's forecast from the daily arrays
        daily = weather.get("daily", {})
        times = daily.get("time", [])
        target = resolved.isoformat()
        if target in times:
            idx = times.index(target)
            weather["forecast_day"] = {
                "date": target,
                "temp_max": daily["temperature_2m_max"][idx],
                "temp_min": daily["temperature_2m_min"][idx],
                "weathercode": daily["weathercode"][idx],
                "precipitation_probability": daily["precipitation_probability_max"][idx],
            }

        return weather
    except Exception as exc:
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# Timer
# ══════════════════════════════════════════════════════════════════════════════

def set_timer(duration: str) -> dict:
    seconds = _parse_duration(duration)
    if seconds <= 0:
        return {"error": f"Could not parse duration from '{duration}'."}

    timer_id = f"timer_{int(time.time())}"

    def _run() -> None:
        time.sleep(seconds)
        if timer_id in _timers:
            _timers[timer_id]["status"] = "done"

    t = threading.Thread(target=_run, daemon=True)
    _timers[timer_id] = {"seconds": seconds, "status": "running", "thread": t}
    t.start()

    mins, secs = divmod(seconds, 60)
    if mins and secs:
        label = f"{mins} minute{'s' if mins != 1 else ''} and {secs} second{'s' if secs != 1 else ''}"
    elif mins:
        label = f"{mins} minute{'s' if mins != 1 else ''}"
    else:
        label = f"{secs} second{'s' if secs != 1 else ''}"
    return {"timer_id": timer_id, "seconds": seconds, "label": label, "status": "running"}


def get_timer_statuses() -> dict:
    return {tid: {"seconds": v["seconds"], "status": v["status"]} for tid, v in _timers.items()}


# ══════════════════���═══════════════════════════════════════════════════════════
# Control system – delegates to MediaCenterController
# ══════════════════════════════════════════════════════════════════════════════

def execute_control_intent(
    intent: str,
    slots: dict,
    media_controller: MediaCenterController,
) -> dict:
    """Execute a control-system intent and return a result dict with 'message'."""

    def _msg(m: str) -> dict:
        return {"message": m, "state": media_controller.get_status_dict()}

    if intent == "PlayMusic":
        song = (slots.get("song_title", "") or "").strip()
        artist = (slots.get("artist_name") or "").strip() or None

        if song:
            music = search_song_by_title(song, artist_name=artist)
            recs = music.get("recordings", [])
            if not recs:
                return _msg("Not available.")
            # Verify the returned title actually matches the query — MusicBrainz
            # fuzzy-matches so a query like "voice music request" can score 100
            # by matching only the word "request".
            top = recs[0]
            top_title = top.get("title", "")
            if not _music_title_matches(song, top_title):
                return _msg("Not available.")
        elif artist:
            music = search_music_by_keyword(artist)
            recs = music.get("recordings", [])
            if not recs:
                return _msg("Not available.")
            top = recs[0]
            top_title = top.get("title", "")
            if not _music_title_matches(artist, top_title):
                return _msg("Not available.")
        else:
            return _msg("Not available.")

        # Use API-returned data for what's shown in the media center
        api_title = top.get("title", song or artist or "Unknown")
        credits = top.get("artist-credit", [])
        api_artist = credits[0].get("name", "") if credits else (artist or "")
        releases = top.get("releases", [])
        api_album = releases[0].get("title", "") if releases else ""
        duration_ms = top.get("length")  # MusicBrainz returns duration in milliseconds

        msg = media_controller.play_music(api_title, artist_name=api_artist, album=api_album, duration_ms=duration_ms)
        return _msg(msg)

    if intent == "PlayMovie":
        title = (slots.get("title", "") or "").strip()
        if not title:
            return _msg("Not available.")

        # Only play if OMDb actually finds a result.
        params: dict = {"apikey": OMDB_API_KEY, "t": title, "r": "json"}
        if slots.get("year"):
            params["y"] = _extract_year(slots.get("year"))
        omdb = _omdb_get(params)
        if not omdb:
            return _msg("Not available.")

        # Require a meaningful vote count — blocks generic phrases that happen
        # to match obscure/accidental IMDb titles (e.g. "A Movie", "Some Film").
        # 5,000 is conservative enough to pass any movie a user would actually
        # want to watch while filtering noise.
        votes_raw = omdb.get("imdbVotes", "N/A")
        try:
            votes = int(votes_raw.replace(",", "")) if votes_raw != "N/A" else 0
        except (ValueError, AttributeError):
            votes = 0
        if votes < 5000:
            return _msg("Not available.")

        # Use API-returned data — canonical title, year, genre, rating, runtime
        api_title = omdb.get("Title") or title
        year = omdb.get("Year")
        rating = omdb.get("imdbRating")
        genre = omdb.get("Genre", "").split(",")[0].strip() or None  # first genre only
        runtime_min: Optional[int] = None
        runtime_raw = omdb.get("Runtime", "")  # e.g. "152 min"
        if runtime_raw and runtime_raw != "N/A":
            m = re.search(r"(\d+)", runtime_raw)
            if m:
                runtime_min = int(m.group(1))
        msg = media_controller.play_movie(api_title, year=year, rating=rating, genre=genre, runtime_min=runtime_min)
        result = _msg(msg)
        result["omdb"] = omdb
        return result

    if intent == "PauseMedia":
        return _msg(media_controller.pause())

    if intent == "ResumeMedia":
        return _msg(media_controller.resume())

    if intent == "StopMedia":
        return _msg(media_controller.stop())

    if intent == "NextTrack":
        return _msg(media_controller.next_track())

    if intent == "ChangeVolume":
        vol_str = slots.get("volume", "50")
        modifier = slots.get("volume_modifier", "set")
        return _msg(media_controller.change_volume(vol_str, modifier))

    if intent == "AddToWatchlist":
        title = (slots.get("title", "") or "").strip()
        if not title or title.lower() in {"my", "the", "a", "an", "watchlist"}:
            return _msg("Please say the movie title to add to your watchlist.")
        return _msg(media_controller.add_to_watchlist(title))

    if intent == "AddToPlaylist":
        song = (slots.get("song_title", slots.get("title", "")) or "").strip()
        if not song or song.lower() in {"my", "the", "a", "an", "playlist"}:
            return _msg("Please say the song title to add to your playlist.")
        return _msg(media_controller.add_to_playlist(song))

    if intent == "ShufflePlaylist":
        return _msg(media_controller.shuffle_playlist())

    return _msg(f"Unhandled control intent: {intent}")


# ══════════════════════════════════════════════════════════════════════════════
# Unified dispatcher
# ══════════════════════════════════════════════════════════════════════════════

CONTROL_INTENTS = {
    "PlayMusic", "PlayMovie", "PauseMedia", "ResumeMedia", "StopMedia",
    "NextTrack", "ChangeVolume", "AddToWatchlist", "AddToPlaylist", "ShufflePlaylist",
}

BASIC_INTENTS = {"Greetings", "Goodbye", "LockSystem", "SetTimer", "GetWeather", "OOS"}


def fulfill(
    intent: str,
    slots: dict,
    media_controller: Optional[MediaCenterController] = None,
) -> dict:
    """Single entry-point: returns a fulfillment result dict."""

    if intent == "OOS":
        return {"message": "Out of scope."}

    if intent == "Greetings":
        return {"message": "Hello!"}

    if intent == "Goodbye":
        return {"message": "Goodbye!"}

    if intent == "LockSystem":
        return {"message": "I can't lock the system by voice. Use the System Control panel to lock."}

    if intent == "SetTimer":
        duration = slots.get("duration", "")
        return set_timer(duration) if duration else {"error": "No duration specified."}

    if intent == "GetWeather":
        location = slots.get("location", "").strip()
        if not location:
            location = _get_current_location()
        return get_weather(location, date=slots.get("date"))

    # ── Specialized domain: Movies ──
    if intent == "SearchMovieByTitle":
        return search_movie_by_title(
            slots.get("title", ""),
            year=slots.get("year"),
            media_type=slots.get("type"),
        )
    if intent == "SearchByKeyword":
        return search_by_keyword(
            slots.get("search_term", ""),
            media_type=slots.get("type"),
            year=slots.get("year"),
        )
    if intent == "GetRatingsAndScore":
        return get_ratings_and_score(slots.get("title", ""), year=slots.get("year"))

    if intent == "GetSeriesSeasonInfo":
        return get_series_season_info(
            slots.get("title", ""), slots.get("season", "1")
        )
    if intent == "RecommendSimilarMovieByKeyword":
        return recommend_similar_by_keyword(
            slots.get("search_term", ""),
            media_type=slots.get("type"),
        )

    # ── Specialized domain: Music ──
    if intent == "SearchArtistByName":
        return search_artist_by_name(slots.get("artist_name", ""))

    if intent == "SearchSongByTitle":
        return search_song_by_title(
            slots.get("song_title", ""), artist_name=slots.get("artist_name")
        )
    if intent == "SearchAlbumByTitle":
        return search_album_by_title(
            slots.get("album_title", ""), artist_name=slots.get("artist_name")
        )
    if intent == "BrowseArtistAlbums":
        return browse_artist_albums(
            slots.get("artist_name", ""), release_type=slots.get("release_type")
        )
    if intent == "SearchMusicByKeyword":
        return search_music_by_keyword(slots.get("search_term", ""))
    if intent == "GetTrackArtist":
        return get_track_artist(slots.get("song_title", ""))

    # ── Control system ──
    if intent in CONTROL_INTENTS:
        if media_controller is None:
            media_controller = MediaCenterController()
        return execute_control_intent(intent, slots, media_controller)

    return {"message": f"No fulfillment handler for intent '{intent}'."}


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════

def _omdb_get(params: dict) -> Optional[dict]:
    try:
        resp = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        data = resp.json()
        return data if data.get("Response") == "True" else None
    except Exception:
        return None


def _mb_get(endpoint: str, params: dict) -> Optional[dict]:
    try:
        resp = requests.get(
            MB_BASE_URL + endpoint, params=params, headers=MB_HEADERS, timeout=10
        )
        return resp.json()
    except Exception:
        return None


_MUSIC_STOPWORDS = {
    # Articles / prepositions
    "a", "an", "the", "in", "on", "of", "to", "for", "with", "by",
    "is", "are", "was", "were", "and", "or", "it", "its",
    # Voice-command words — appear in requests, never in real song/artist titles
    "play", "pause", "stop", "start", "listen",
    # Generic music-domain words — too broad to validate a specific title
    "music", "song", "songs", "track", "audio", "sound",
    # Vague quantifiers
    "some", "any", "that", "this", "those",
}


def _meaningful_words(text: str) -> set:
    """Return lowercase words longer than 3 chars that are not stopwords."""
    return {
        w for w in re.findall(r"[a-z]+", text.lower())
        if len(w) > 3 and w not in _MUSIC_STOPWORDS
    }


def _music_title_matches(query: str, returned_title: str) -> bool:
    """Return True only if all meaningful query words appear in the returned title.

    This prevents MusicBrainz fuzzy matches from playing unrelated songs —
    e.g. the query 'voice music request' should NOT match 'request (Music Video)'
    because 'voice' is absent from the returned title.
    Returns False when the query has no meaningful words (generic phrases).
    """
    q_words = _meaningful_words(query)
    if not q_words:
        return False  # Only stopwords / very short words — nothing specific enough
    t_words = _meaningful_words(returned_title)
    return q_words.issubset(t_words)


def _canned_movie(title: str) -> dict:
    return {
        "Title": title, "Year": "N/A", "Genre": "N/A",
        "Director": "N/A", "imdbRating": "N/A",
        "Plot": "No data available (API offline).",
        "Response": "True", "canned": True,
    }


def _extract_year(text: str) -> str:
    m = re.search(r"\b(19|20)\d{2}\b", str(text))
    return m.group(0) if m else str(text)


def _extract_digits(text: str) -> str:
    m = re.search(r"\d+", str(text))
    return m.group(0) if m else ""


_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "a": 1, "an": 1, "half": 0,
}


def _words_to_digits(text: str) -> str:
    """Replace number words with digits, e.g. 'five minutes' → '5 minutes'."""
    def _replace(m):
        word = m.group(0).lower()
        return str(_WORD_NUMS[word]) if word in _WORD_NUMS else m.group(0)
    pattern = r"\b(" + "|".join(re.escape(w) for w in _WORD_NUMS) + r")\b"
    return re.sub(pattern, _replace, text, flags=re.IGNORECASE)


def _parse_duration(text: str) -> int:
    """Return total seconds parsed from a human duration string."""
    text = _words_to_digits(text.lower())
    total = 0
    for val, unit in re.findall(r"(\d+)\s*(seconds?|minutes?|hours?)", text):
        val = int(val)
        if unit.startswith("hour"):
            total += val * 3600
        elif unit.startswith("minute"):
            total += val * 60
        else:
            total += val
    # Fallback: plain number treated as minutes
    if total == 0:
        m = re.search(r"\b(\d+)\b", text)
        if m:
            total = int(m.group(1)) * 60
    return total
