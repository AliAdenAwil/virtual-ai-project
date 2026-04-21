from __future__ import annotations

import json
from pathlib import Path

INTENTS = [
    # Basic
    "OOS",
    "Greetings",
    "Goodbye",
    "SetTimer",
    "GetWeather",
    # Movies & TV
    "SearchMovieByTitle",
    "SearchByKeyword",
    "GetRatingsAndScore",
    "GetSeriesSeasonInfo",
    "RecommendSimilarMovieByKeyword",
    # Music
    "SearchArtistByName",
    "SearchSongByTitle",
    "SearchAlbumByTitle",
    "BrowseArtistAlbums",
    "GetTrackArtist",
    # Control system
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
]

MOVIE_TITLES = [
    "inception",
    "interstellar",
    "the dark knight",
    "arrival",
    "the matrix",
    "blade runner",
    "dune",
    "parasite",
    "oppenheimer",
    "avatar",
]

SONG_TITLES = [
    "blinding lights",
    "shape of you",
    "bad guy",
    "perfect",
    "hello",
    "levitating",
    "viva la vida",
    "believer",
    "halo",
    "yellow",
]

ALBUM_TITLES = [
    "after hours",
    "divide",
    "future nostalgia",
    "thriller",
    "25",
    "random access memories",
    "midnights",
    "fine line",
    "ghost stories",
    "starboy",
]

ARTISTS = [
    "the weeknd",
    "ed sheeran",
    "dua lipa",
    "coldplay",
    "beyonce",
    "drake",
    "taylor swift",
    "bruno mars",
    "ariana grande",
    "billie eilish",
]

KEYWORDS = [
    "space",
    "romance",
    "comedy",
    "thriller",
    "sci fi",
    "adventure",
    "crime",
    "mystery",
    "animation",
    "drama",
]

SEASONS = [
    "season 1",
    "season 2",
    "season 3",
    "season 4",
    "season 5",
]

YEARS = ["2019", "2020", "2021", "2022", "2023", "2024"]
TYPES = ["movie", "series", "song", "album", "playlist", "show"]
RELEASE_TYPES = ["single", "album", "ep", "live album", "remix"]
VOLUMES = ["20 percent", "30 percent", "50 percent", "70 percent", "80 percent", "maximum"]
DURATIONS = ["5 minutes", "10 minutes", "15 minutes", "20 minutes", "30 minutes", "1 hour"]
LOCATIONS = ["seattle", "new york", "london", "dubai", "toronto", "paris", "tokyo"]
DATES = ["today", "tomorrow", "this weekend", "monday", "friday", "next week"]


def render_template(template: str, values: dict[str, str], intent: str) -> dict:
    tokens: list[str] = []
    slots: list[str] = []

    for token in template.split():
        if token.startswith("{") and token.endswith("}"):
            slot_name = token[1:-1]
            value_tokens = values[slot_name].split()
            for idx, value_token in enumerate(value_tokens):
                tokens.append(value_token)
                slots.append(f"B-{slot_name}" if idx == 0 else f"I-{slot_name}")
        else:
            tokens.append(token)
            slots.append("O")

    return {
        "tokens": tokens,
        "intent": intent,
        "slots": slots,
    }


def pick(pool: list[str], i: int, shift: int = 0) -> str:
    return pool[(i + shift) % len(pool)]


def generate_intent_examples(intent: str, n: int = 30) -> list[dict]:
    examples: list[dict] = []

    templates_by_intent: dict[str, list[str]] = {
        "OOS": [
            "what is the capital of mars",
            "how many wheels does a cloud have",
            "teach me quantum gravity in one sentence",
            "calculate the meaning of life now",
            "who invented the color blue",
            "open the portal to another dimension",
        ],
        "Greetings": [
            "hello",
            "hi there",
            "hey assistant",
            "good morning",
            "good evening",
            "how are you",
        ],
        "Goodbye": [
            "goodbye",
            "bye",
            "see you later",
            "talk to you soon",
            "catch you later",
            "have a good night",
        ],
        "SetTimer": [
            "set a timer for {duration}",
            "start a timer for {duration}",
            "create a timer of {duration}",
            "set timer {duration}",
            "please set a timer for {duration}",
            "i need a timer for {duration}",
        ],
        "GetWeather": [
            "what is the weather in {location} {date}",
            "show weather for {location} {date}",
            "weather in {location} on {date}",
            "tell me the forecast for {location} {date}",
            "how is weather in {location} {date}",
            "give weather update for {location} {date}",
        ],
        "SearchMovieByTitle": [
            "find movie {title}",
            "search for movie {title}",
            "look up {title} movie",
            "get details for movie {title}",
            "show me movie {title} from {year}",
            "search movie {title} {year}",
        ],
        "SearchByKeyword": [
            "find {type} about {search_term}",
            "search {type} with keyword {search_term}",
            "show {type} related to {search_term}",
            "look for {type} on {search_term}",
            "browse {type} about {search_term}",
            "get {type} by keyword {search_term}",
        ],
        "GetRatingsAndScore": [
            "what is rating for {title}",
            "show score of {title}",
            "get ratings for {title}",
            "how well rated is {title}",
            "rating of {title} from {year}",
            "score for {title} {year}",
        ],
        "GetSeriesSeasonInfo": [
            "show {season} info for {title}",
            "get {season} details of {title}",
            "what happens in {season} of {title}",
            "series {title} {season} summary",
            "tell me about {title} {season}",
            "open {season} for {title}",
        ],
        "RecommendSimilarMovieByKeyword": [
            "recommend {type} similar to {search_term}",
            "suggest {type} like {search_term}",
            "give me similar {type} for {search_term}",
            "find related {type} to {search_term}",
            "recommend based on {search_term} {type}",
            "what {type} are similar to {search_term}",
        ],
        "SearchArtistByName": [
            "find artist {artist_name}",
            "search for artist {artist_name}",
            "look up artist {artist_name}",
            "show profile of {artist_name}",
            "tell me about artist {artist_name}",
            "open artist page for {artist_name}",
        ],
        "SearchSongByTitle": [
            "find song {song_title}",
            "search for song {song_title}",
            "look up track {song_title}",
            "show me song {song_title}",
            "get details of song {song_title}",
            "open song {song_title}",
        ],
        "SearchAlbumByTitle": [
            "find album {album_title}",
            "search album {album_title}",
            "look up album {album_title}",
            "show me album {album_title}",
            "get album details {album_title}",
            "open album {album_title}",
        ],
        "BrowseArtistAlbums": [
            "show {release_type} from {artist_name}",
            "browse albums by {artist_name}",
            "list {release_type} by {artist_name}",
            "get releases by {artist_name} type {release_type}",
            "open {artist_name} {release_type}",
            "find {release_type} of {artist_name}",
        ],
        "GetTrackArtist": [
            "who sings {song_title}",
            "who is artist of {song_title}",
            "track artist for {song_title}",
            "get performer of {song_title}",
            "tell me artist behind {song_title}",
            "show artist for song {song_title}",
        ],
        "PlayMusic": [
            "play {song_title}",
            "start playing {song_title}",
            "play music by {artist_name}",
            "put on {song_title}",
            "play a song from {artist_name}",
            "queue {song_title}",
        ],
        "PlayMovie": [
            "play movie {title}",
            "start movie {title}",
            "put on {title}",
            "play {title} movie",
            "open movie {title}",
            "watch {title}",
        ],
        "PauseMedia": [
            "pause",
            "pause media",
            "pause playback",
            "pause the {type}",
            "hold on pause this",
            "stop for a moment",
        ],
        "ResumeMedia": [
            "resume",
            "resume playback",
            "continue playing",
            "continue the {type}",
            "keep going",
            "play again",
        ],
        "StopMedia": [
            "stop",
            "stop playback",
            "stop the {type}",
            "end this media",
            "turn it off",
            "stop now",
        ],
        "NextTrack": [
            "next track",
            "skip",
            "play next",
            "skip this song",
            "go to next",
            "next",
        ],
        "ChangeVolume": [
            "set volume to {volume}",
            "change volume to {volume}",
            "make volume {volume}",
            "turn volume to {volume}",
            "adjust sound to {volume}",
            "volume {volume}",
        ],
        "AddToWatchlist": [
            "add {title} to watchlist",
            "put {title} in my watchlist",
            "save {title} to watchlist",
            "add {type} {title} to watchlist",
            "watchlist add {title}",
            "include {title} in watchlist",
        ],
        "AddToPlaylist": [
            "add {song_title} to playlist",
            "put {song_title} in my playlist",
            "save {song_title} to playlist",
            "add album {album_title} to playlist",
            "include {song_title} in playlist",
            "playlist add {song_title}",
        ],
        "ShufflePlaylist": [
            "shuffle playlist",
            "shuffle my {type}",
            "turn on shuffle",
            "mix up this playlist",
            "shuffle songs",
            "randomize playback",
        ],
    }

    templates = templates_by_intent[intent]

    for i in range(n):
        template = templates[i % len(templates)]
        values = {
            "title": pick(MOVIE_TITLES, i, 0),
            "search_term": pick(KEYWORDS, i, 1),
            "season": pick(SEASONS, i, 2),
            "year": pick(YEARS, i, 3),
            "type": pick(TYPES, i, 4),
            "artist_name": pick(ARTISTS, i, 5),
            "song_title": pick(SONG_TITLES, i, 6),
            "album_title": pick(ALBUM_TITLES, i, 7),
            "release_type": pick(RELEASE_TYPES, i, 8),
            "volume": pick(VOLUMES, i, 9),
            "duration": pick(DURATIONS, i, 10),
            "location": pick(LOCATIONS, i, 11),
            "date": pick(DATES, i, 12),
        }

        examples.append(render_template(template, values, intent))

    return examples


def main() -> None:
    all_examples: list[dict] = []
    for intent in INTENTS:
        all_examples.extend(generate_intent_examples(intent, n=40))

    output_path = Path("data/nlu/train.full.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(all_examples)} examples ({len(INTENTS)} intents × 40) at {output_path}")
    print("Run training with:")
    print("  python -m nlu.train --data_path data/nlu/train.full.json --output_dir models/joint_nlu")


if __name__ == "__main__":
    main()
