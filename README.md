---
title: Atlas Voice Assistant
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.44.0"
app_file: app.py
pinned: false
---

# Atlas — Voice Assistant

**CSI5180 Group 14** | Pipeline: User Verification → Wake Word → ASR → NLU → Fulfillment → Answer Generation → TTS

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Quick Start (Pre-trained Models)](#quick-start-pre-trained-models)
4. [Environment Variables & API Keys](#environment-variables--api-keys)
5. [Re-training Models](#re-training-models)
6. [Pipeline Modules](#pipeline-modules)
7. [Large Files & Git LFS](#large-files--git-lfs)

---

## Project Overview

Atlas is a fully local voice assistant with the following capabilities:

- **User Verification** — speaker-identity check via SpeechBrain ECAPA-TDNN embeddings
- **Wake Word Detection** — custom CNN trained on "Hey Atlas" recordings
- **ASR** — OpenAI Whisper (offline, no API key required)
- **NLU** — fine-tuned BERT for joint intent classification + BIO slot filling (26 intents)
- **Fulfillment** — OMDb (movies), MusicBrainz (music), Open-Meteo (weather)
- **Answer Generation** — template-based with random paraphrase selection
- **TTS** — Google Text-to-Speech (gTTS)

---

## File Structure

```
virtual-ai-project/
├── app.py                        # Streamlit UI (main entry point)
├── requirements.txt
├── .env.example                  # Copy to .env and fill in your keys
│
├── src/
│   ├── config.py                 # All tuneable constants
│   ├── audio.py                  # Microphone capture + preprocessing
│   ├── asr.py                    # Whisper ASR wrapper
│   ├── wakeword.py               # Wake word detection logic
│   ├── wakeword_model.py         # CNN model definition
│   ├── wakeword_features.py      # Mel-spectrogram feature extraction
│   ├── verifier.py               # Speaker verification
│   ├── embeddings.py             # SpeechBrain ECAPA-TDNN embeddings
│   ├── voiceprint_store.py       # Voiceprint persistence (pickle)
│   ├── fulfillment.py            # External API calls (OMDb, MusicBrainz, weather)
│   ├── answer_generation.py      # Template-based response generation
│   ├── tts.py                    # gTTS wrapper
│   ├── state_machine.py          # Assistant state (Locked/Listening/Processing)
│   └── control_system.py         # Media center controller
│
├── nlu/
│   ├── model.py                  # BERT + intent head + slot head
│   ├── dataset.py                # Training data loader + BIO alignment
│   ├── train.py                  # HuggingFace Trainer training loop
│   ├── inference.py              # Prediction + heuristic fallbacks
│   └── utils.py                  # Intent/slot label maps, BIO decoder
│
├── scripts/
│   ├── generate_nlu_dataset.py   # Generate NLU training data from templates
│   ├── train_wakeword.py         # Train wake word CNN
│   ├── tune_threshold.py         # Tune speaker verification threshold
│   └── enroll.py                 # Enroll users from voice recordings
│
├── models/
│   └── joint_nlu/                # Trained NLU model (stored in Git LFS)
│       ├── model_state.pt
│       ├── model_config.json
│       ├── label_mappings.json
│       └── tokenizer*.json
│
├── data/
│   ├── nlu/
│   │   ├── train.full.json       # Full NLU training set (~4000 samples)
│   │   └── train.sample.json     # Small sample for quick testing
│   ├── wakeword_models/
│   │   ├── wakeword_cnn.pt       # Trained wake word CNN (Git LFS)
│   │   └── wakeword_config.pkl   # Wake word config (Git LFS)
│   └── voiceprints/
│       └── voiceprint_store.pkl  # Enrolled speaker voiceprints (Git LFS)
│
├── raw_recordings/               # Voice enrollment WAV files (Git LFS)
│   ├── positve/                  # Authorized users (positive + near + other)
│   └── Negative/                 # Unauthorized users
│
├── wakeword_dataset/             # Wake word training audio (Git LFS)
│   ├── positive/
│   ├── near/
│   └── other/
│
└── pretrained_models/
    └── spkrec-ecapa-voxceleb/    # SpeechBrain speaker model (auto-downloaded on first run)
```

---

## Quick Start (Pre-trained Models)

> Use this path to **run the assistant** without retraining anything. All models are included via Git LFS.

### 1. Prerequisites

- Python 3.11–3.13
- macOS or Linux (microphone access required)
- Internet connection (for gTTS, weather, and movie/music APIs)
- [Git LFS](https://git-lfs.com/) installed

Install Git LFS:
```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs
```

### 2. Clone the repository

```bash
git lfs install
git clone https://github.com/Mohumeddahir/virtual-ai-project.git
cd virtual-ai-project
```

Git LFS automatically downloads large model files (`.pt`, `.pkl`, `.wav`, `.m4a`) during clone.

### 3. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
OMDB_API_KEY=your_omdb_api_key_here   # Free key: https://www.omdbapi.com/apikey.aspx
VA_BYPASS_PIN=1234                     # Optional: PIN to bypass voice verification in UI
```

### 5. Run the app

```bash
source .venv/bin/activate
python -m streamlit run app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

> **Note:** First launch takes ~30 seconds as Whisper and BERT load into memory. Subsequent interactions are fast thanks to model caching.

---

## Environment Variables & API Keys

| Variable | Required | Description | Where to get it |
|---|---|---|---|
| `OMDB_API_KEY` | **Yes** | Movies & TV data (OMDb) | [omdbapi.com/apikey.aspx](https://www.omdbapi.com/apikey.aspx) — free |
| `VA_BYPASS_PIN` | No | PIN to skip voice verification in the UI | Any number you choose |

**No key needed for:**
- Weather — Open-Meteo (free, open API)
- Music — MusicBrainz (free, open API)
- ASR — Whisper runs fully offline

---

## Re-training Models

> Use this path if you want to retrain from scratch or add new data.

### Re-train the NLU model

```bash
source .venv/bin/activate

# (Optional) Regenerate training data from templates
python scripts/generate_nlu_dataset.py

# Train — takes ~10 min on CPU (M1/M2 Mac)
python -m nlu.train \
  --data_path data/nlu/train.full.json \
  --output_dir models/joint_nlu

# Evaluate on 96 test samples
python eval_nlu.py
```

Expected results: ~99% intent accuracy, ~99% slot accuracy.

### Re-train the Wake Word model

```bash
source .venv/bin/activate

# Requires wakeword_dataset/ with positive/, near/, other/ subdirectories
python scripts/train_wakeword.py
```

Model saved to `data/wakeword_models/wakeword_cnn.pt`.

### Re-enroll speaker voiceprints

```bash
source .venv/bin/activate

# Requires WAV files in raw_recordings/positve/ named <Name>-positive-<N>.wav
python scripts/enroll.py

# Tune per-user verification thresholds
python scripts/tune_threshold.py
```

Voiceprints saved to `data/voiceprints/voiceprint_store.pkl`.

---

## Pipeline Modules

### 1. User Verification
- SpeechBrain ECAPA-TDNN (192-dim speaker embeddings)
- Cosine similarity against enrolled centroids
- Per-user tuned threshold (~0.46)
- Fallback: PIN bypass via `VA_BYPASS_PIN` env variable

### 2. Wake Word Detection
- Custom CNN with Global Average Pooling trained on "Hey Atlas"
- Mel-spectrogram features (128 mel bins)
- Detection threshold: 0.58 (configurable in `src/config.py`)

### 3. ASR
- OpenAI Whisper (`base` model, configurable in `src/config.py`)
- Fully offline — no API key required
- Anti-hallucination: `no_speech_threshold=0.6`, `condition_on_previous_text=False`

### 4. NLU
- Fine-tuned `bert-base-uncased` with dual output heads
- **26 intents**: GetWeather, SearchMovieByTitle, SearchByKeyword, GetRatingsAndScore, GetSeriesSeasonInfo, RecommendSimilarMovieByKeyword, SearchArtistByName, SearchSongByTitle, SearchAlbumByTitle, BrowseArtistAlbums, SearchMusicByKeyword, GetTrackArtist, PlayMusic, PlayMovie, PauseMedia, ResumeMedia, StopMedia, NextTrack, ChangeVolume, AddToWatchlist, AddToPlaylist, ShufflePlaylist, SetTimer, Greetings, Goodbye, OOS
- BIO slot tagging with 15 slot types (location, date, title, artist_name, song_title, etc.)

### 5. Fulfillment

| Domain | API | Key Required |
|---|---|---|
| Weather | Open-Meteo | No |
| Movies/TV | OMDb | **Yes** (free) |
| Music | MusicBrainz | No |
| Media control | Internal controller | No |

### 6. Answer Generation
- Fully template-based, no LLM required
- 3–4 random paraphrase variants per intent
- Hit **Regenerate** in the UI to get a different phrasing

### 7. TTS
- Google Text-to-Speech (gTTS) — requires internet

---

## Large Files & Git LFS

The following file types are tracked by Git LFS:

| Pattern | Description |
|---|---|
| `*.pt` | PyTorch model weights |
| `*.safetensors` | HuggingFace model weights |
| `*.pth` | PyTorch checkpoints |
| `*.pkl` | Pickle files (voiceprints, configs) |
| `*.wav` | Voice enrollment recordings |
| `*.m4a` | Wake word dataset audio |
| `*.mp3` | Audio files |

**Pull LFS files after cloning:**
```bash
git lfs pull
```

**Skip LFS and retrain everything yourself:**
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Mohumeddahir/virtual-ai-project.git
cd virtual-ai-project
# Then follow the Re-training Models section above
```
