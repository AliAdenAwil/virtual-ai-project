"""
NLU Evaluation Script — eval_nlu.py
Evaluates the JointNLUPredictor against 100 labelled test samples.
"""

from nlu.inference import JointNLUPredictor

# ---------------------------------------------------------------------------
# Test samples
# ---------------------------------------------------------------------------
SAMPLES = [
    # Weather (0-24)
    {"input": "What is the weather in Ottawa tomorrow?", "intent": "GetWeather", "slots": {"location": "Ottawa", "date": "tomorrow"}},
    {"input": "What's the weather in Toronto today?", "intent": "GetWeather", "slots": {"location": "Toronto", "date": "today"}},
    {"input": "Will it rain in Ottawa tomorrow?", "intent": "GetWeather", "slots": {"location": "Ottawa", "date": "tomorrow"}},
    {"input": "What's the forecast for Ottawa next Monday?", "intent": "GetWeather", "slots": {"location": "Ottawa", "date": "next Monday"}},
    {"input": "Weather in Montreal this weekend", "intent": "GetWeather", "slots": {"location": "Montreal", "date": "this weekend"}},
    {"input": "Do I need a jacket tomorrow in Ottawa?", "intent": "GetWeather", "slots": {"location": "Ottawa", "date": "tomorrow"}},
    {"input": "Is it colder today or tomorrow in Ottawa?", "intent": "GetWeather", "slots": {"location": "Ottawa"}},
    {"input": "Weather for Vancouver tonight", "intent": "GetWeather", "slots": {"location": "Vancouver"}},
    {"input": "What's the temperature in Ottawa in two days?", "intent": "GetWeather", "slots": {"location": "Ottawa"}},
    {"input": "Will it snow tomorrow?", "intent": "GetWeather", "slots": {"date": "tomorrow"}},
    {"input": "What about tomorrow's weather?", "intent": "GetWeather", "slots": {"date": "tomorrow"}},
    {"input": "Is it sunny today?", "intent": "GetWeather", "slots": {}},
    {"input": "Weather next Friday", "intent": "GetWeather", "slots": {"date": "next Friday"}},
    {"input": "Will it be hot this weekend in Ottawa?", "intent": "GetWeather", "slots": {"location": "Ottawa"}},
    {"input": "Forecast for Ottawa April 5th", "intent": "GetWeather", "slots": {"location": "Ottawa"}},
    {"input": "What's the weather tomorrow morning?", "intent": "GetWeather", "slots": {"date": "tomorrow"}},
    {"input": "Rain forecast for tomorrow", "intent": "GetWeather", "slots": {"date": "tomorrow"}},
    {"input": "Will it rain this afternoon?", "intent": "GetWeather", "slots": {}},
    {"input": "Weather in Ottawa right now", "intent": "GetWeather", "slots": {"location": "Ottawa"}},
    {"input": "Temperature in Ottawa tomorrow night", "intent": "GetWeather", "slots": {"location": "Ottawa", "date": "tomorrow"}},
    {"input": "Weather for next week in Ottawa", "intent": "GetWeather", "slots": {"location": "Ottawa"}},
    {"input": "Is tomorrow warmer than today?", "intent": "GetWeather", "slots": {}},
    {"input": "Forecast for Sunday", "intent": "GetWeather", "slots": {"date": "Sunday"}},
    {"input": "What's the weather in Ottawa the day after tomorrow?", "intent": "GetWeather", "slots": {"location": "Ottawa"}},
    {"input": "How windy will it be tomorrow?", "intent": "GetWeather", "slots": {"date": "tomorrow"}},
    # Movies (25-44)
    {"input": "Find the movie Inception", "intent": "SearchMovieByTitle", "slots": {"title": "Inception"}},
    {"input": "Tell me about Titanic", "intent": "SearchMovieByTitle", "slots": {"title": "Titanic"}},
    {"input": "Show me Interstellar", "intent": "SearchMovieByTitle", "slots": {"title": "Interstellar"}},
    {"input": "Find action movies", "intent": "SearchByKeyword", "slots": {"search_term": "action"}},
    {"input": "Search movies about space", "intent": "SearchByKeyword", "slots": {"search_term": "space"}},
    {"input": "What is the rating of Fight Club?", "intent": "GetRatingsAndScore", "slots": {"title": "Fight Club"}},
    {"input": "IMDb score of Avatar", "intent": "GetRatingsAndScore", "slots": {"title": "Avatar"}},
    {"input": "Give me Breaking Bad season 1", "intent": "GetSeriesSeasonInfo", "slots": {"title": "Breaking Bad", "season": "1"}},
    {"input": "Episodes of Friends season 2", "intent": "GetSeriesSeasonInfo", "slots": {"title": "Friends", "season": "2"}},
    {"input": "Recommend movies like Inception", "intent": "RecommendSimilarMovieByKeyword", "slots": {"search_term": "Inception"}},
    {"input": "Suggest horror movies", "intent": "RecommendSimilarMovieByKeyword", "slots": {"search_term": "horror"}},
    {"input": "Find comedy films from 2010", "intent": "SearchByKeyword", "slots": {"search_term": "comedy"}},
    {"input": "Show sci-fi movies", "intent": "SearchByKeyword", "slots": {"search_term": "sci-fi"}},
    {"input": "Rating of The Dark Knight", "intent": "GetRatingsAndScore", "slots": {"title": "The Dark Knight"}},
    {"input": "Tell me about Avatar 2009", "intent": "SearchMovieByTitle", "slots": {"title": "Avatar"}},
    {"input": "Recommend movies about space exploration", "intent": "RecommendSimilarMovieByKeyword", "slots": {"search_term": "space exploration"}},
    {"input": "Find movies similar to Titanic", "intent": "RecommendSimilarMovieByKeyword", "slots": {"search_term": "Titanic"}},
    {"input": "What's the score of Joker", "intent": "GetRatingsAndScore", "slots": {"title": "Joker"}},
    {"input": "Show Breaking Bad season 3", "intent": "GetSeriesSeasonInfo", "slots": {"title": "Breaking Bad", "season": "3"}},
    {"input": "Find thriller movies", "intent": "SearchByKeyword", "slots": {"search_term": "thriller"}},
    # Music (45-64)
    {"input": "Tell me about Drake", "intent": "SearchArtistByName", "slots": {"artist_name": "Drake"}},
    {"input": "Who is Taylor Swift?", "intent": "SearchArtistByName", "slots": {"artist_name": "Taylor Swift"}},
    {"input": "Find the song Blinding Lights", "intent": "SearchSongByTitle", "slots": {"song_title": "Blinding Lights"}},
    {"input": "Play Shape of You", "intent": "SearchSongByTitle", "slots": {"song_title": "Shape of You"}},
    {"input": "Show album Thriller", "intent": "SearchAlbumByTitle", "slots": {"album_title": "Thriller"}},
    {"input": "Find album After Hours", "intent": "SearchAlbumByTitle", "slots": {"album_title": "After Hours"}},
    {"input": "Show albums by Adele", "intent": "BrowseArtistAlbums", "slots": {"artist_name": "Adele"}},
    {"input": "List albums from Eminem", "intent": "BrowseArtistAlbums", "slots": {"artist_name": "Eminem"}},
    {"input": "Who sings Blinding Lights?", "intent": "GetTrackArtist", "slots": {"song_title": "Blinding Lights"}},
    {"input": "Artist of Starboy", "intent": "GetTrackArtist", "slots": {"song_title": "Starboy"}},
    {"input": "Find song Starboy by The Weeknd", "intent": "SearchSongByTitle", "slots": {"song_title": "Starboy", "artist_name": "The Weeknd"}},
    {"input": "Show albums released by Drake", "intent": "BrowseArtistAlbums", "slots": {"artist_name": "Drake"}},
    {"input": "What is the album 1989?", "intent": "SearchAlbumByTitle", "slots": {"album_title": "1989"}},
    {"input": "Who made Shape of You", "intent": "GetTrackArtist", "slots": {"song_title": "Shape of You"}},
    {"input": "Info about The Weeknd", "intent": "SearchArtistByName", "slots": {"artist_name": "The Weeknd"}},
    {"input": "Find song Believer", "intent": "SearchSongByTitle", "slots": {"song_title": "Believer"}},
    {"input": "Albums by Coldplay", "intent": "BrowseArtistAlbums", "slots": {"artist_name": "Coldplay"}},
    {"input": "Artist of Rolling in the Deep", "intent": "GetTrackArtist", "slots": {"song_title": "Rolling in the Deep"}},
    {"input": "Find album DAMN", "intent": "SearchAlbumByTitle", "slots": {"album_title": "DAMN"}},
    {"input": "Show songs by Drake", "intent": "SearchSongByTitle", "slots": {"artist_name": "Drake"}},
    # Control (65-79)
    {"input": "Play Blinding Lights", "intent": "PlayMusic", "slots": {}},
    {"input": "Play Inception", "intent": "PlayMovie", "slots": {"title": "Inception"}},
    {"input": "Pause", "intent": "PauseMedia", "slots": {}},
    {"input": "Resume", "intent": "ResumeMedia", "slots": {}},
    {"input": "Stop", "intent": "StopMedia", "slots": {}},
    {"input": "Next song", "intent": "NextTrack", "slots": {}},
    {"input": "Skip track", "intent": "NextTrack", "slots": {}},
    {"input": "Increase volume", "intent": "ChangeVolume", "slots": {}},
    {"input": "Set volume to 50", "intent": "ChangeVolume", "slots": {}},
    {"input": "Lower volume", "intent": "ChangeVolume", "slots": {}},
    {"input": "Add Inception to watchlist", "intent": "AddToWatchlist", "slots": {"title": "Inception"}},
    {"input": "Add this song to playlist", "intent": "AddToPlaylist", "slots": {}},
    {"input": "Add Blinding Lights to playlist", "intent": "AddToPlaylist", "slots": {}},
    {"input": "Shuffle playlist", "intent": "ShufflePlaylist", "slots": {}},
    {"input": "Turn on shuffle", "intent": "ShufflePlaylist", "slots": {}},
    # Basic (80-85)
    {"input": "Hello", "intent": "Greetings", "slots": {}},
    {"input": "Hi", "intent": "Greetings", "slots": {}},
    {"input": "Good morning", "intent": "Greetings", "slots": {}},
    {"input": "Bye", "intent": "Goodbye", "slots": {}},
    {"input": "See you later", "intent": "Goodbye", "slots": {}},
    {"input": "Goodbye", "intent": "Goodbye", "slots": {}},
    # OOS (86-99)
    {"input": "What is the capital of France?", "intent": "OOS", "slots": {}},
    {"input": "Solve 2 + 2", "intent": "OOS", "slots": {}},
    {"input": "Tell me a joke", "intent": "OOS", "slots": {}},
    {"input": "Explain AI", "intent": "OOS", "slots": {}},
    {"input": "Who is the president of Canada?", "intent": "OOS", "slots": {}},
    {"input": "What is machine learning?", "intent": "OOS", "slots": {}},
    {"input": "Define gravity", "intent": "OOS", "slots": {}},
    {"input": "Translate hello to French", "intent": "OOS", "slots": {}},
    {"input": "What time is it?", "intent": "OOS", "slots": {}},
    {"input": "What is 5 times 5", "intent": "OOS", "slots": {}},
]

# ---------------------------------------------------------------------------
# Group definitions
# ---------------------------------------------------------------------------
GROUPS = {
    "Weather":    (0, 24),
    "Movies":     (25, 44),
    "Music":      (45, 64),
    "Control":    (65, 79),
    "Basic+OOS":  (80, 99),
}

# ---------------------------------------------------------------------------
# Slot matching helper
# ---------------------------------------------------------------------------
def slots_correct(expected_slots: dict, predicted_slots: dict) -> bool:
    """
    Returns True if:
      - expected_slots is empty (nothing to check), OR
      - for every expected key, the key exists in predicted_slots AND
        the expected value is a case-insensitive substring of the predicted value.
    """
    if not expected_slots:
        return True
    for key, exp_val in expected_slots.items():
        if key not in predicted_slots:
            return False
        pred_val = predicted_slots[key]
        if exp_val.lower() not in pred_val.lower():
            return False
    return True


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    print("Loading NLU model...")
    model = JointNLUPredictor(model_dir="models/joint_nlu")
    print("Model loaded.\n")

    total = len(SAMPLES)
    intent_correct = 0
    slot_correct = 0
    failures = []

    group_results = {g: {"intent_correct": 0, "slot_correct": 0, "total": 0} for g in GROUPS}

    for idx, sample in enumerate(SAMPLES):
        text = sample["input"]
        expected_intent = sample["intent"]
        expected_slots = sample["slots"]

        result = model.predict(text)
        predicted_intent = result["intent"]
        predicted_slots = result["slots"]

        intent_ok = (predicted_intent == expected_intent)
        slot_ok = slots_correct(expected_slots, predicted_slots)

        if intent_ok:
            intent_correct += 1
        if slot_ok:
            slot_correct += 1

        # Assign to group
        for g_name, (g_start, g_end) in GROUPS.items():
            if g_start <= idx <= g_end:
                group_results[g_name]["total"] += 1
                if intent_ok:
                    group_results[g_name]["intent_correct"] += 1
                if slot_ok:
                    group_results[g_name]["slot_correct"] += 1
                break

        if not intent_ok or not slot_ok:
            failures.append({
                "idx": idx,
                "input": text,
                "expected_intent": expected_intent,
                "predicted_intent": predicted_intent,
                "intent_ok": intent_ok,
                "expected_slots": expected_slots,
                "predicted_slots": predicted_slots,
                "slot_ok": slot_ok,
            })

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("NLU EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nOverall Intent Accuracy : {intent_correct}/{total}  ({100*intent_correct/total:.1f}%)")
    print(f"Overall Slot  Accuracy  : {slot_correct}/{total}  ({100*slot_correct/total:.1f}%)")

    print("\n--- Per-Group Intent Accuracy ---")
    for g_name, stats in group_results.items():
        t = stats["total"]
        ic = stats["intent_correct"]
        sc = stats["slot_correct"]
        print(f"  {g_name:<12}: intent {ic}/{t} ({100*ic/t:.1f}%)  |  slots {sc}/{t} ({100*sc/t:.1f}%)")

    print(f"\n--- Failures ({len(failures)} total) ---")
    if not failures:
        print("  No failures! Perfect score.")
    else:
        for f in failures:
            print(f"\n  [{f['idx']:>3}] Input   : {f['input']}")
            intent_marker = "OK" if f["intent_ok"] else "FAIL"
            slot_marker   = "OK" if f["slot_ok"]   else "FAIL"
            print(f"       Intent  : expected={f['expected_intent']!r:30s}  predicted={f['predicted_intent']!r}  [{intent_marker}]")
            print(f"       Slots   : expected={str(f['expected_slots']):<45}  predicted={f['predicted_slots']}  [{slot_marker}]")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
