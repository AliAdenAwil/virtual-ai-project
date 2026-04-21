from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

INTENTS = [
    # ── Basic ──────────────────────────────────────────────────────────────
    "OOS",
    "Greetings",
    "Goodbye",
    "LockSystem",
    "SetTimer",
    "GetWeather",
    # ── Movies & TV (OMDb) ─────────────────────────────────────────────────
    "SearchMovieByTitle",
    "SearchByKeyword",
    "GetRatingsAndScore",
    "GetSeriesSeasonInfo",
    "RecommendSimilarMovieByKeyword",
    # ── Music (MusicBrainz) ────────────────────────────────────────────────
    "SearchArtistByName",
    "SearchSongByTitle",
    "SearchAlbumByTitle",
    "BrowseArtistAlbums",
    "SearchMusicByKeyword",
    "GetTrackArtist",
    # ── Control system ─────────────────────────────────────────────────────
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

SLOTS = [
    "title",
    "search_term",
    "season",
    "year",
    "type",
    "artist_name",
    "song_title",
    "album_title",
    "release_type",
    "volume",
    "duration",
    "location",
    "date",
]

BIO_TAGS = ["O"]
for slot in SLOTS:
    BIO_TAGS.append(f"B-{slot}")
    BIO_TAGS.append(f"I-{slot}")


def build_label_maps() -> dict[str, dict]:
    intent2id = {label: idx for idx, label in enumerate(INTENTS)}
    id2intent = {idx: label for label, idx in intent2id.items()}

    slot2id = {label: idx for idx, label in enumerate(BIO_TAGS)}
    id2slot = {idx: label for label, idx in slot2id.items()}

    return {
        "intent2id": intent2id,
        "id2intent": id2intent,
        "slot2id": slot2id,
        "id2slot": id2slot,
    }


def save_label_maps(output_dir: str | Path, maps: dict[str, dict]) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mappings_path = output_path / "label_mappings.json"
    with mappings_path.open("w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False, indent=2)
    return mappings_path


def load_label_maps(model_dir: str | Path) -> dict[str, dict]:
    path = Path(model_dir) / "label_mappings.json"
    with path.open("r", encoding="utf-8") as f:
        maps = json.load(f)

    maps["id2intent"] = {int(k): v for k, v in maps["id2intent"].items()}
    maps["id2slot"] = {int(k): v for k, v in maps["id2slot"].items()}
    return maps


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bio_tags_to_slots(tokens: list[str], tags: list[str]) -> dict[str, str]:
    slots: dict[str, list[str]] = {}
    current_slot = None
    current_tokens: list[str] = []

    def flush_current() -> None:
        nonlocal current_slot, current_tokens
        if current_slot and current_tokens:
            value = " ".join(current_tokens).strip()
            if value:
                if current_slot in slots:
                    slots[current_slot].append(value)
                else:
                    slots[current_slot] = [value]
        current_slot = None
        current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag == "O":
            flush_current()
            continue

        if "-" not in tag:
            flush_current()
            continue

        prefix, slot_name = tag.split("-", 1)

        if prefix == "B":
            flush_current()
            current_slot = slot_name
            current_tokens = [token]
        elif prefix == "I":
            if current_slot == slot_name:
                current_tokens.append(token)
            else:
                flush_current()
                current_slot = slot_name
                current_tokens = [token]
        else:
            flush_current()

    flush_current()

    flattened: dict[str, str] = {}
    for slot_name, values in slots.items():
        flattened[slot_name] = " ".join(values)
    return flattened
