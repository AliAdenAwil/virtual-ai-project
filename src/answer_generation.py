"""Answer generation – template-based with optional LLM override.

When use_llm=True, informational/conversational intents are handled by Llama
3.3 70B via Groq (src/llm_answer.py). Control intents (Play/Pause/etc.) and
SetTimer always use templates because the controller message IS the answer and
LLM adds no value. Falls back to templates silently if the LLM call fails or
the API key is absent.
"""
from __future__ import annotations

import random
import re
from datetime import date as _d
from typing import Optional


def generate_answer(
    intent: str,
    fulfillment_result: dict,
    slots: dict,
    use_llm: bool = False,
    question: str = "",
) -> str:
    """Return a spoken answer for the given intent.

    If use_llm=True, attempts LLM generation first and falls back to templates.
    question is the original user transcript, passed to the LLM for context.
    """
    if use_llm:
        from src.llm_answer import generate_answer_llm
        llm_answer = generate_answer_llm(intent, fulfillment_result, slots, question=question)
        if llm_answer:
            return llm_answer
        # Fall through to template on failure / unsupported intent

    gen = _GENERATORS.get(intent)
    if gen:
        try:
            return gen(fulfillment_result, slots)
        except Exception:
            pass
    return _fallback(intent, fulfillment_result, slots)


# ── Basic intents ────────────────────────────────────────────────────────────

def _greetings(r, s):
    return random.choice([
        "Hello! I'm Atlas, your voice assistant. How can I help you today?",
        "Hey there! Atlas here — what can I do for you?",
        "Hi! I'm Atlas. Ready to help. What do you need?",
        "Good to hear from you! I'm Atlas. How can I assist?",
    ])


def _goodbye(r, s):
    return random.choice([
        "Goodbye! Have a great day!",
        "See you later! Take care!",
        "Bye! Don't hesitate to ask if you need anything.",
        "Until next time! Have a wonderful day.",
    ])


def _lock_system(r, s):
    return "I can't lock the system by voice. Use the System Control panel to lock."


def _oos(r, s):
    return random.choice([
        "Sorry, I don't know how to help with that. Try asking about movies, music, weather, or media playback.",
        "That's outside what I can do. I'm best with movies, music, weather, or controlling media.",
        "I'm not sure I can help with that one. Ask me about movies, music, the weather, or media control.",
        "Hmm, I didn't catch that as something I can handle. Try movies, music, weather, or media commands.",
    ])


def _set_timer(r, s):
    if "error" in r:
        return random.choice([
            f"Sorry, I couldn't set the timer: {r['error']}",
            f"I had trouble with that timer: {r['error']}",
            f"Timer setup failed — {r['error']}",
        ])
    label = r.get("label", s.get("duration", "the specified time"))
    return random.choice([
        f"Timer set for {label}! I'll track it for you.",
        f"Got it! Your {label} timer is now running.",
        f"Sure! I've started a {label} timer.",
        f"Done — {label} timer is counting down.",
    ])


def _get_weather(r, s):
    if "error" in r:
        return random.choice([
            f"Sorry, I couldn't fetch the weather: {r['error']}",
            f"Weather lookup failed: {r['error']}",
            f"I wasn't able to get the weather right now: {r['error']}",
        ])

    loc = r.get("location_name", s.get("location", "your location"))
    is_today = r.get("is_today", True)
    forecast_day = r.get("forecast_day")

    if not is_today and forecast_day:
        resolved_iso = r.get("resolved_date", "")
        try:
            dt = _d.fromisoformat(resolved_iso)
            date_label = dt.strftime("%A, %B %-d")
        except Exception:
            date_label = r.get("requested_date", "that day").capitalize()

        code = int(forecast_day.get("weathercode", 0))
        desc = _wmo_description(code)
        t_max = forecast_day.get("temp_max", "N/A")
        t_min = forecast_day.get("temp_min", "N/A")
        precip = forecast_day.get("precipitation_probability", "")
        precip_str = f", {precip}% chance of precipitation" if precip not in ("", None) else ""

        return random.choice([
            f"Weather in {loc} on {date_label}: {desc}, high of {t_max}°C and low of {t_min}°C{precip_str}.",
            f"On {date_label} in {loc}, expect {desc} with a high of {t_max}°C and low of {t_min}°C{precip_str}.",
            f"Forecast for {loc} on {date_label}: {desc}, between {t_min}°C and {t_max}°C{precip_str}.",
            f"{loc} on {date_label} looks {desc} — highs around {t_max}°C, lows near {t_min}°C{precip_str}.",
        ])
    else:
        cw = r.get("current_weather", {})
        temp = cw.get("temperature", "N/A")
        code = int(cw.get("weathercode", 0))
        desc = _wmo_description(code)
        wind = cw.get("windspeed", "")
        wind_str = f", wind {wind} km/h" if wind else ""

        return random.choice([
            f"Current weather in {loc}: {temp}°C, {desc}{wind_str}.",
            f"Right now in {loc}: {desc}, {temp}°C{wind_str}.",
            f"It's {temp}°C and {desc} in {loc} at the moment{wind_str}.",
            f"In {loc} right now: {temp}°C with {desc}{wind_str}.",
        ])


def _wmo_description(code: int) -> str:
    if code == 0:             return "clear sky"
    if code in (1, 2, 3):    return "partly cloudy"
    if code in (45, 48):     return "foggy"
    if code in (51, 53, 55): return "drizzle"
    if code in (61, 63, 65): return "rain"
    if code in (71, 73, 75): return "snow"
    if code in (80, 81, 82): return "rain showers"
    if code in (95, 96, 99): return "thunderstorm"
    return "mixed conditions"


_SENTENCE_END_RE = re.compile(r"[.!?](?:['\")\]]+)?(?=\s|$)")


def _truncate_at_sentence(text: str, max_chars: int = 120) -> str:
    """Trim text at a sentence boundary near max_chars when possible."""
    text = str(text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    head = text[:max_chars]
    matches = list(_SENTENCE_END_RE.finditer(head))
    if matches:
        return head[:matches[-1].end()].strip()

    # If no sentence end inside max_chars, allow a small lookahead window.
    lookahead = text[max_chars:max_chars + 90]
    next_end = _SENTENCE_END_RE.search(lookahead)
    if next_end:
        return text[:max_chars + next_end.end()].strip()

    # No sentence boundary nearby: do not force a mid-sentence cut.
    return text


# ── Movies & TV (OMDb) ───────────────────────────────────────────────────────

def _search_movie(r, s):
    if r.get("Response") == "False" or not r.get("Title"):
        title = s.get("title", "that movie")
        return random.choice([
            f"Sorry, I couldn't find '{title}'.",
            f"No results for '{title}'. Try a different title.",
            f"I didn't find anything for '{title}'. Maybe check the spelling?",
        ])
    title    = r.get("Title", "Unknown")
    year     = r.get("Year", "")
    genre    = r.get("Genre", "")
    director = r.get("Director", "")
    rating   = r.get("imdbRating", "N/A")
    plot     = r.get("Plot", "")
    plot_snip = ""
    if plot and plot not in ("N/A", ""):
        plot_trimmed = _truncate_at_sentence(plot, max_chars=120)
        if plot_trimmed:
            plot_snip = f" {plot_trimmed}"

    genre_str    = f" a {genre} film" if genre and genre != "N/A" else ""
    dir_str      = f" directed by {director}" if director and director != "N/A" else ""
    rating_str   = f", rated {rating}/10 on IMDb" if rating and rating != "N/A" else ""
    year_str     = f" ({year})" if year else ""

    return random.choice([
        f"{title}{year_str} is{genre_str}{dir_str}{rating_str}.{plot_snip}",
        f"'{title}'{year_str} —{genre_str}{dir_str}{rating_str}.{plot_snip}",
        f"Here's what I found: {title}{year_str},{genre_str}{dir_str}{rating_str}.{plot_snip}",
    ])


def _search_by_keyword(r, s):
    items = r.get("Search", [])
    if not items:
        term = s.get("search_term", "that keyword")
        return random.choice([
            f"No results found for '{term}'.",
            f"I couldn't find anything matching '{term}'.",
            f"Nothing came up for '{term}'. Try a broader search.",
        ])
    titles = [f"{i['Title']} ({i.get('Year', '?')})" for i in items[:5]]
    joined = ", ".join(titles)
    return random.choice([
        f"Here are some results: {joined}.",
        f"I found these titles: {joined}.",
        f"Matching titles: {joined}.",
    ])


def _get_ratings(r, s):
    if r.get("Response") == "False":
        title = s.get("title", "that title")
        return random.choice([
            f"Sorry, couldn't find ratings for '{title}'.",
            f"No rating info available for '{title}'.",
            f"I wasn't able to find scores for '{title}'.",
        ])
    title = r.get("Title", "Unknown")
    imdb  = r.get("imdbRating", "N/A")
    meta  = r.get("Metascore", "N/A")
    votes = r.get("imdbVotes", "")

    meta_str  = f" and a Metascore of {meta}" if meta and meta != "N/A" else ""
    votes_str = f" ({votes} votes)" if votes and votes != "N/A" else ""

    return random.choice([
        f"{title} has an IMDb score of {imdb}/10{meta_str}{votes_str}.",
        f"On IMDb, {title} scores {imdb}/10{meta_str}{votes_str}.",
        f"{title} is rated {imdb}/10 on IMDb{meta_str}{votes_str}.",
    ])


def _series_season(r, s):
    if r.get("Response") == "False":
        title = s.get("title", "that series")
        return random.choice([
            f"Couldn't find season info for '{title}'.",
            f"No season data available for '{title}'.",
            f"I didn't find season details for '{title}'.",
        ])
    title    = r.get("Title", s.get("title", "the series"))
    season   = r.get("Season", s.get("season", "1"))
    episodes = r.get("Episodes", [])
    count    = len(episodes)
    first    = episodes[0].get("Title", "") if episodes else ""
    first_str = f" The first episode is '{first}'." if first else ""

    return random.choice([
        f"{title} Season {season} has {count} episode(s).{first_str}",
        f"Season {season} of {title} contains {count} episodes.{first_str}",
        f"There are {count} episodes in {title} Season {season}.{first_str}",
    ])


def _recommend(r, s):
    items = r.get("Search", [])
    if not items:
        term = s.get("search_term", "that topic")
        return random.choice([
            f"No recommendations found for '{term}'.",
            f"I couldn't find anything to recommend for '{term}'.",
            f"Nothing matched '{term}' for recommendations.",
        ])
    titles = [f"{i['Title']} ({i.get('Year', '?')})" for i in items[:5]]
    joined = ", ".join(titles)
    return random.choice([
        f"You might enjoy: {joined}.",
        f"Based on that, try: {joined}.",
        f"Here are some recommendations: {joined}.",
        f"How about these: {joined}.",
    ])


# ── Music (MusicBrainz) ──────────────────────────────────────────────────────

def _search_artist(r, s):
    artists = r.get("artists", [])
    if not artists:
        name = s.get("artist_name", "that artist")
        return random.choice([
            f"No info found for '{name}'.",
            f"I couldn't find an artist named '{name}'.",
            f"Nothing came up for '{name}'. Try the full name.",
        ])
    a       = artists[0]
    name    = a.get("name", "Unknown")
    country = a.get("country", "")
    tags    = [t["name"] for t in a.get("tags", [])[:3]]

    country_str = f" from {country}" if country else ""
    tags_str    = f" and known for {', '.join(tags)}" if tags else ""

    return random.choice([
        f"{name} is{country_str}{tags_str}.",
        f"{name} —{country_str} artist{tags_str}.",
        f"Here's what I found: {name}{country_str}{tags_str}.",
    ])


def _search_song(r, s):
    recs = r.get("recordings", [])
    if not recs:
        song = s.get("song_title", "that song")
        return random.choice([
            f"No song found for '{song}'.",
            f"I couldn't find '{song}'.",
            f"Nothing came up for '{song}'. Try a different spelling.",
        ])
    rec      = recs[0]
    title    = rec.get("title", "Unknown")
    credits  = rec.get("artist-credit", [])
    artist   = credits[0].get("name", "") if credits else ""
    releases = rec.get("releases", [])
    album    = releases[0].get("title", "") if releases else ""

    artist_str = f" by {artist}" if artist else ""
    album_str  = f", from the album '{album}'" if album else ""

    return random.choice([
        f"'{title}'{artist_str}{album_str}.",
        f"That song is '{title}'{artist_str}{album_str}.",
        f"I found '{title}'{artist_str}{album_str}.",
    ])


def _search_album(r, s):
    releases = r.get("releases", [])
    if not releases:
        album = s.get("album_title", "that album")
        return random.choice([
            f"No album found for '{album}'.",
            f"I couldn't find an album called '{album}'.",
            f"Nothing came up for the album '{album}'.",
        ])
    rel    = releases[0]
    title  = rel.get("title", "Unknown")
    credit = rel.get("artist-credit", [])
    artist = credit[0].get("name", "") if credit else ""
    date   = rel.get("date", "")

    artist_str = f" by {artist}" if artist else ""
    date_str   = f", released {date}" if date else ""

    return random.choice([
        f"'{title}'{artist_str}{date_str}.",
        f"The album '{title}'{artist_str}{date_str}.",
        f"Found it: '{title}'{artist_str}{date_str}.",
    ])


def _browse_albums(r, s):
    groups = r.get("release-groups", [])
    artist = s.get("artist_name", "that artist")
    if not groups:
        return random.choice([
            f"No albums found for {artist}.",
            f"I couldn't find any albums by {artist}.",
            f"Nothing came up for {artist}'s albums.",
        ])
    titles = [g.get("title", "?") for g in groups[:6]]
    joined = ", ".join(titles)
    return random.choice([
        f"{artist.title()} has released: {joined}.",
        f"Albums by {artist.title()}: {joined}.",
        f"Here are {artist.title()}'s albums: {joined}.",
    ])


def _search_music_keyword(r, s):
    recs = r.get("recordings", [])
    if not recs:
        term = s.get("search_term", "that genre")
        return random.choice([
            f"No music found for '{term}'.",
            f"I couldn't find any music matching '{term}'.",
            f"Nothing came up for '{term}'. Try a different keyword.",
        ])
    results = []
    for rec in recs[:4]:
        title = rec.get("title", "?")
        credits = rec.get("artist-credit", [])
        artist = credits[0].get("name", "") if credits else ""
        results.append(f"'{title}'" + (f" by {artist}" if artist else ""))
    joined = ", ".join(results)
    return random.choice([
        f"Here are some tracks: {joined}.",
        f"I found these songs: {joined}.",
        f"Matching music: {joined}.",
    ])


def _get_track_artist(r, s):
    return _search_song(r, s)


# ── Control system ───────────────────────────────────────────────────────────

def _control_passthrough(r, s):
    return r.get("message", "Done.")


# ── Fallback ─────────────────────────────────────────────────────────────────

def _fallback(intent, r, s):
    msg = r.get("message", "")
    return msg if msg else f"Action completed ({intent})."


# ── Dispatch table ───────────────────────────────────────────────────────────

_GENERATORS = {
    "Greetings":                _greetings,
    "Goodbye":                  _goodbye,
    "OOS":                      _oos,
    "LockSystem":               _lock_system,
    "SetTimer":                 _set_timer,
    "GetWeather":               _get_weather,
    # Movies
    "SearchMovieByTitle":       _search_movie,
    "SearchByKeyword":          _search_by_keyword,
    "GetRatingsAndScore":       _get_ratings,
    "GetSeriesSeasonInfo":      _series_season,
    "RecommendSimilarMovieByKeyword": _recommend,
    # Music
    "SearchArtistByName":       _search_artist,
    "SearchSongByTitle":        _search_song,
    "SearchAlbumByTitle":       _search_album,
    "BrowseArtistAlbums":       _browse_albums,
    "SearchMusicByKeyword":     _search_music_keyword,
    "GetTrackArtist":           _get_track_artist,
    # Control system
    "PlayMusic":                _control_passthrough,
    "PlayMovie":                _control_passthrough,
    "PauseMedia":               _control_passthrough,
    "ResumeMedia":              _control_passthrough,
    "StopMedia":                _control_passthrough,
    "NextTrack":                _control_passthrough,
    "ChangeVolume":             _control_passthrough,
    "AddToWatchlist":           _control_passthrough,
    "AddToPlaylist":            _control_passthrough,
    "ShufflePlaylist":          _control_passthrough,
}
