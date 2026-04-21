from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer

from nlu.model import JointIntentSlotModel
from nlu.utils import bio_tags_to_slots, load_label_maps


@dataclass
class ManualOverride:
    intent: str
    slots: dict[str, str]


class JointNLUPredictor:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        maps = load_label_maps(self.model_dir)
        self.intent2id = maps["intent2id"]
        self.id2intent  = maps["id2intent"]
        self.slot2id    = maps["slot2id"]
        self.id2slot    = maps["id2slot"]

        self.model = JointIntentSlotModel.from_pretrained_local(
            self.model_dir, device=self.device
        )
        self.intent_confidence_threshold = 0.55

    # ------------------------------------------------------------------ #
    # Public predict
    # ------------------------------------------------------------------ #

    def predict(
        self,
        sentence: str,
        manual_override: ManualOverride | None = None,
    ) -> dict:
        if manual_override is not None:
            return {
                "intent": manual_override.intent,
                "slots": manual_override.slots,
                "bypass_used": True,
            }

        words = sentence.strip().split()
        if not words:
            return {"intent": "OOS", "slots": {}, "bypass_used": False}

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        intent_logits = outputs["intent_logits"]
        slot_logits   = outputs["slot_logits"]
        intent_probs  = torch.softmax(intent_logits, dim=-1)

        intent_id          = int(torch.argmax(intent_logits, dim=-1).item())
        predicted_intent   = self.id2intent[intent_id]
        intent_confidence  = float(torch.max(intent_probs, dim=-1).values.item())

        heuristic = self._heuristic_intent(sentence)
        if heuristic is not None and intent_confidence < self.intent_confidence_threshold:
            predicted_intent = heuristic

        token_slot_ids = torch.argmax(slot_logits, dim=-1)[0].tolist()

        word_ids = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=64,
        ).word_ids()

        word_level_tags = ["O"] * len(words)
        seen_word_ids: set[int] = set()
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen_word_ids:
                continue
            seen_word_ids.add(word_idx)
            word_level_tags[word_idx] = self.id2slot[int(token_slot_ids[token_idx])]

        slots = bio_tags_to_slots(words, word_level_tags)
        # Sanitize all slot values from BIO decoder (strip trailing punctuation)
        slots = {k: self._clean(v) for k, v in slots.items()}
        slots = self._apply_slot_fallbacks(
            sentence=sentence, intent=predicted_intent, slots=slots
        )

        return {
            "intent": predicted_intent,
            "slots": slots,
            "intent_confidence": intent_confidence,
            "bypass_used": False,
        }

    # ------------------------------------------------------------------ #
    # Heuristic intent classifier (rule-based safety net)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _heuristic_intent(sentence: str) -> str | None:
        t = sentence.lower().strip()
        if not t:
            return None

        # ── Basic ──
        if re.search(r"\b(hello|hi\b|hey|good morning|good evening|how are you)\b", t):
            return "Greetings"
        if re.search(r"\b(bye|goodbye|see you|talk to you|catch you later|good night)\b", t):
            return "Goodbye"
        if re.search(r"\b(timer|remind me|alarm)\b", t):
            return "SetTimer"
        if re.search(r"\b(weather|forecast|temperature|rain|sunny|snow|humidity)\b", t):
            return "GetWeather"

        # ── Control system – high priority ──
        if re.search(r"\b(pause)\b", t):
            return "PauseMedia"
        if re.search(r"\b(resume|unpause|continue playing)\b", t):
            return "ResumeMedia"
        if re.search(r"\b(stop\b(?! timer))\b", t) and not re.search(r"\btimer\b", t):
            return "StopMedia"
        if re.search(r"\b(next track|skip|next song|play next)\b", t):
            return "NextTrack"
        if re.search(r"\b(shuffle|randomize)\b", t):
            return "ShufflePlaylist"
        if re.search(r"\b(volume|percent|louder|quieter|mute|unmute)\b", t):
            return "ChangeVolume"
        if re.search(r"\b(watchlist)\b", t):
            if re.search(r"\b(add|save|put|include)\b", t):
                return "AddToWatchlist"
        if re.search(r"\b(playlist)\b", t):
            if re.search(r"\b(add|save|put|include)\b", t):
                return "AddToPlaylist"
        if re.search(r"\b(play|watch|start)\b", t) and re.search(r"\b(movie|film)\b", t):
            return "PlayMovie"
        if re.search(r"\b(play|listen|queue|put on)\b", t) and re.search(
            r"\b(song|track|music|artist|album|by)\b", t
        ):
            return "PlayMusic"

        # ── Movies & TV domain ──
        if re.search(r"\b(rating|score|imdb|metascore|rotten|rated)\b", t):
            return "GetRatingsAndScore"
        if re.search(r"\b(season|episode)\b", t) and re.search(
            r"\b(show|series|tv|info|detail|tell|about)\b", t
        ):
            return "GetSeriesSeasonInfo"
        if re.search(r"\b(recommend|suggest|similar|like|related)\b", t):
            return "RecommendSimilarMovieByKeyword"
        if re.search(r"\b(find|search|look up|get)\b", t) and re.search(
            r"\b(movie|film)\b", t
        ):
            return "SearchMovieByTitle"
        if re.search(r"\b(find|search|look for|browse|keyword|about)\b", t) and re.search(
            r"\b(movie|series|show|film)\b", t
        ):
            return "SearchByKeyword"

        # ── Music domain ──
        if re.search(r"\b(who (sings|performs|sang|recorded|made)|artist (of|for|behind))\b", t):
            return "GetTrackArtist"
        if re.search(r"\b(albums|discography)\b", t) and re.search(r"\b(by|from)\b", t):
            return "BrowseArtistAlbums"
        if re.search(r"\b(album)\b", t) and re.search(r"\b(find|search|look|about)\b", t):
            return "SearchAlbumByTitle"
        if re.search(r"\b(song|track|recording)\b", t) and re.search(
            r"\b(find|search|look|about)\b", t
        ):
            return "SearchSongByTitle"
        if re.search(r"\b(artist|band|musician|singer)\b", t) and re.search(
            r"\b(find|search|look|tell|about|info)\b", t
        ):
            return "SearchArtistByName"

        # ── Generic keyword / search fallback ──
        if re.search(r"\b(find|search|look for)\b", t):
            return "SearchByKeyword"

        return None

    # ------------------------------------------------------------------ #
    # Slot fallbacks (regex-based, applied after BIO decoding)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean(value: str) -> str:
        v = value
        # Normalize common smart quotes from ASR/LLM transcripts before stripping.
        v = (
            v.replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2018", "'").replace("\u2019", "'")
        )
        v = v.strip(" .,!?:;\"'`")
        v = re.sub(r"\s+", " ", v)
        # Remove trailing punctuation that may have been absorbed into the slot value
        v = re.sub(r"[?!.,;:\"'`]+$", "", v).strip()
        v = re.sub(r"\b(on|for|at|in|of|the)$", "", v, flags=re.IGNORECASE).strip()
        return v

    @classmethod
    def _apply_slot_fallbacks(
        cls, sentence: str, intent: str, slots: dict[str, str]
    ) -> dict[str, str]:
        t = sentence.strip()
        p = dict(slots)

        # ── SetTimer ──
        if intent == "SetTimer" and "duration" not in p:
            m = re.search(
                r"\b(\d+)\s*[-]?\s*(seconds?|minutes?|hours?)\b", t, re.IGNORECASE
            )
            if not m:
                m = re.search(
                    r"\b(one|two|three|four|five|ten|fifteen|twenty|thirty)\s*[-]?\s*(seconds?|minutes?|hours?)\b",
                    t, re.IGNORECASE,
                )
            if m:
                p["duration"] = m.group(0)

        # ── ChangeVolume ──
        if intent == "ChangeVolume" and "volume" not in p:
            m = re.search(r"\b(\d{1,3})\s*(percent|%)\b", t, re.IGNORECASE)
            if m:
                p["volume"] = m.group(0)
            elif re.search(r"\b(max|maximum)\b", t, re.IGNORECASE):
                p["volume"] = "maximum"
            elif re.search(r"\b(min|mute)\b", t, re.IGNORECASE):
                p["volume"] = "0"

        # Detect volume modifier (decrease, increase, set) for ChangeVolume.
        # "by" indicates a relative delta, while "to" indicates an absolute target.
        if intent == "ChangeVolume" and "volume" in p:
            has_decrease = re.search(r"\b(decrease|down|lower|reduce)\b", t, re.IGNORECASE)
            has_increase = re.search(r"\b(increase|up|raise|boost)\b", t, re.IGNORECASE)
            has_by = re.search(r"\bby\b", t, re.IGNORECASE)
            has_to = re.search(r"\bto\b", t, re.IGNORECASE)

            if has_to:
                p["volume_modifier"] = "set"
            elif has_decrease and has_by:
                p["volume_modifier"] = "decrease"
            elif has_increase and has_by:
                p["volume_modifier"] = "increase"
            elif has_decrease:
                p["volume_modifier"] = "decrease"
            elif has_increase:
                p["volume_modifier"] = "increase"
            else:
                p["volume_modifier"] = "set"

        # ── GetWeather ──
        if intent == "GetWeather":
            if "location" not in p:
                for pat in [
                    r"\bweather\s+in\s+([A-Za-z][A-Za-z\s]{1,30}?)(?=\s+(today|tomorrow|on|this|next|\?|$))",
                    r"\bforecast\s+for\s+([A-Za-z][A-Za-z\s]{1,30}?)(?=\s+|\?|$)",
                    r"\bin\s+([A-Za-z][A-Za-z\s]{1,30}?)(?=\s+(today|tomorrow|on|this|next|\?|$))",
                ]:
                    m = re.search(pat, t, re.IGNORECASE)
                    if m:
                        cand = cls._clean(m.group(1))
                        cand = re.sub(
                            r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|next\s+week)\b",
                            "", cand, flags=re.IGNORECASE,
                        ).strip()
                        if cand:
                            p["location"] = cand
                            break
            if "date" not in p:
                m = re.search(
                    r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|this\s+weekend|next\s+week)\b",
                    t, re.IGNORECASE,
                )
                if m:
                    p["date"] = m.group(0)

        # ── SearchMovieByTitle ──
        if intent == "SearchMovieByTitle" and "title" not in p:
            for pat in [
                r"\b(?:find|search for|look up|get details for|show me)\s+(?:movie\s+)?(.+?)(?:\s+(?:from\s+)?\d{4})?$",
                r"\b(?:movie|film)\s+(.+?)(?:\s+\d{4})?$",
            ]:
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    cand = cls._clean(m.group(1))
                    cand = re.sub(r"\b(movie|film)$", "", cand, flags=re.IGNORECASE).strip()
                    if cand:
                        p["title"] = cand
                        break
            _extract_year_slot(t, p)

        # ── GetRatingsAndScore ──
        if intent == "GetRatingsAndScore" and "title" not in p:
            for pat in [
                r"\b(?:rating|score|rated|imdb)\s+(?:of|for)?\s+(.+?)(?:\s+\d{4})?$",
                r"\b(?:how\s+(?:good|well)\s+is)\s+(.+?)(?:\s+\d{4})?$",
            ]:
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    p["title"] = cls._clean(m.group(1))
                    break
            _extract_year_slot(t, p)

        # ── GetSeriesSeasonInfo ──
        if intent == "GetSeriesSeasonInfo":
            if "title" not in p:
                m = re.search(
                    r"\b(?:details?|info|about|of)\s+(.+?)\s+season\b", t, re.IGNORECASE
                )
                if m:
                    p["title"] = cls._clean(m.group(1))
                else:
                    m = re.search(r"\b([A-Za-z][A-Za-z\s]{2,40}?)\s+season\s+\d+\b", t, re.IGNORECASE)
                    if m:
                        p["title"] = cls._clean(m.group(1))
            if "season" not in p:
                m = re.search(r"\bseason\s+(\d+)\b", t, re.IGNORECASE)
                if m:
                    p["season"] = m.group(1)

        # ── RecommendSimilarMovieByKeyword / SearchByKeyword ──
        if intent in {"RecommendSimilarMovieByKeyword", "SearchByKeyword"}:
            if "search_term" not in p:
                for pat in [
                    r"\b(?:similar to|like|about|keyword|related to|on)\s+(.+?)(?:\s+\d{4})?$",
                    r"\b(?:recommend|suggest)\s+(?:\w+\s+)?(?:about|for|like)\s+(.+?)$",
                ]:
                    m = re.search(pat, t, re.IGNORECASE)
                    if m:
                        p["search_term"] = cls._clean(m.group(1))
                        break
            if "type" not in p:
                m = re.search(r"\b(movie|series|show|film|album|song)\b", t, re.IGNORECASE)
                if m:
                    p["type"] = m.group(1).lower()

        # ── SearchArtistByName ──
        if intent == "SearchArtistByName" and "artist_name" not in p:
            for pat in [
                r"\b(?:artist|band|singer|musician)\s+(.+?)$",
                r"\b(?:tell me about|about|find|search for|look up)\s+(.+?)(?:\s+artist)?$",
                r"\b(?:info on|information on)\s+(.+?)$",
            ]:
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    p["artist_name"] = cls._clean(m.group(1))
                    break

        # ── SearchSongByTitle ──
        if intent == "SearchSongByTitle" and "song_title" not in p:
            for pat in [
                r"\b(?:song|track|recording)\s+(.+?)(?:\s+by\s+\w+)?$",
                r"\b(?:find|search for|look up)\s+(?:song\s+)?(.+?)$",
            ]:
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    cand = cls._clean(m.group(1))
                    if not re.search(r"\b(song|track|recording|find|search)\b", cand, re.IGNORECASE):
                        p["song_title"] = cand
                        break
            if "artist_name" not in p:
                m = re.search(r"\bby\s+([A-Za-z][A-Za-z\s]{1,30})\b", t, re.IGNORECASE)
                if m:
                    p["artist_name"] = cls._clean(m.group(1))

        # ── SearchAlbumByTitle ──
        if intent == "SearchAlbumByTitle" and "album_title" not in p:
            for pat in [
                r"\b(?:album)\s+(.+?)(?:\s+by\s+\w+)?$",
                r"\b(?:find|search for|look up)\s+(?:album\s+)?(.+?)$",
            ]:
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    cand = cls._clean(m.group(1))
                    if cand.lower() not in {"album"}:
                        p["album_title"] = cand
                        break
            if "artist_name" not in p:
                m = re.search(r"\bby\s+([A-Za-z][A-Za-z\s]{1,30})\b", t, re.IGNORECASE)
                if m:
                    p["artist_name"] = cls._clean(m.group(1))

        # ── BrowseArtistAlbums ──
        if intent == "BrowseArtistAlbums" and "artist_name" not in p:
            for pat in [
                r"\b(?:albums?|discography|releases?)\s+(?:of|by|from)\s+(.+?)$",
                r"\b(?:by|from)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\s+\w+)?$",
                r"\b(?:show me|list|browse)\s+(?:albums?\s+(?:by|from)\s+)?(.+?)$",
            ]:
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    p["artist_name"] = cls._clean(m.group(1))
                    break

        # ── GetTrackArtist ──
        if intent == "GetTrackArtist" and "song_title" not in p:
            for pat in [
                r"\b(?:who sings|who performs|who sang|who recorded)\s+(.+?)$",
                r"\b(?:artist of|artist for|artist behind)\s+(.+?)$",
                r"\b(?:what artist|who made)\s+(.+?)$",
            ]:
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    p["song_title"] = cls._clean(m.group(1))
                    break

        # ── PlayMovie ──
        if intent == "PlayMovie" and "title" not in p:
            for pat in [
                r"\b(?:play|watch|start)\s+(?:movie\s+)?(.+?)(?:\s+movie)?$",
            ]:
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    cand = cls._clean(m.group(1))
                    cand = re.sub(r"\b(movie|film)$", "", cand, re.IGNORECASE).strip()
                    if cand and not re.fullmatch(r"(?:my|the|a|an)\b", cand, re.IGNORECASE):
                        p["title"] = cand
                        break

        # ── AddToWatchlist ──
        if intent == "AddToWatchlist" and "title" not in p:
            for pat in [
                r"\b(?:add|save|put|include)\s+(.+?)\s+(?:to\s+watchlist|in\s+watchlist)",
            ]:
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    cand = cls._clean(m.group(1))
                    cand = re.sub(r"\b(movie|film|watchlist)$", "", cand, re.IGNORECASE).strip()
                    if cand and not re.fullmatch(r"(?:my|the|a|an)\b", cand, re.IGNORECASE):
                        p["title"] = cand
                        break

        # ── PlayMusic / AddToPlaylist ──
        if intent in {"PlayMusic", "AddToPlaylist"}:
            if "song_title" not in p:
                for pat in [
                    r"\b(?:play|queue|put on|start playing)\s+(.+?)(?:\s+by\s+\w+)?$",
                    r"\b(?:add|save|include)\s+(.+?)\s+(?:to\s+playlist|in\s+playlist|playlist)",
                ]:
                    m = re.search(pat, t, re.IGNORECASE)
                    if m:
                        song = cls._clean(m.group(1))
                        if not re.search(r"\b(playlist|music|song|track|artist|album)\b", song, re.IGNORECASE) and not re.fullmatch(r"(?:my|the|a|an)\b", song, re.IGNORECASE):
                            p["song_title"] = song
                            break
            if "artist_name" not in p:
                m = re.search(r"\bby\s+([A-Za-z][A-Za-z\s]{1,30})\b", t, re.IGNORECASE)
                if m:
                    p["artist_name"] = cls._clean(m.group(1))

        return p


# ── Year extraction helper ────────────────────────────────────────────────────

def _extract_year_slot(text: str, slots: dict) -> None:
    if "year" not in slots:
        m = re.search(r"\b(19|20)\d{2}\b", text)
        if m:
            slots["year"] = m.group(0)


# ══════════════════════════════════════════════════════════════════════════════
# Module-level convenience function
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_PREDICTOR: JointNLUPredictor | None = None


def predict(
    sentence: str,
    model_dir: str = "models/joint_nlu",
    manual_intent: str | None = None,
    manual_slots: dict[str, str] | None = None,
) -> dict:
    global _DEFAULT_PREDICTOR

    if _DEFAULT_PREDICTOR is None or str(_DEFAULT_PREDICTOR.model_dir) != str(Path(model_dir)):
        _DEFAULT_PREDICTOR = JointNLUPredictor(model_dir=model_dir)

    override = None
    if manual_intent is not None and manual_slots is not None:
        override = ManualOverride(intent=manual_intent, slots=manual_slots)

    return _DEFAULT_PREDICTOR.predict(sentence, manual_override=override)


if __name__ == "__main__":
    predictor = JointNLUPredictor(model_dir="models/joint_nlu")
    for sent in [
        "play blinding lights by the weeknd",
        "what is the weather in Paris today",
        "set a timer for 10 minutes",
        "pause",
        "what is the IMDb rating of inception",
        "tell me about Daft Punk",
    ]:
        result = predictor.predict(sent)
        print(f"{sent!r:60s} → intent={result['intent']}, slots={result['slots']}")
