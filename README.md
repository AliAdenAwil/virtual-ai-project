# Atlas Voice Assistant

Local voice assistant pipeline:
User Verification -> Wake Word -> ASR -> NLU -> Fulfillment -> Answer -> TTS

## Quick Run (Pretrained)

1. Unzip and enter project folder:

```bash
cd virtual-ai-project
```

2. Create environment and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Create env file:

```bash
cp .env.example .env
```

4. Add your keys in `.env`:

```env
# Required
OMDB_API_KEY=your_key_here        # https://www.omdbapi.com/apikey.aspx

# Optional — LLM answers (free, no credit card)
GROQ_API_KEY=your_groq_key_here   # https://console.groq.com

# Optional — HF Spaces deployment
HF_TOKEN=your_hf_token_here       # https://huggingface.co/settings/tokens

# Optional — UI bypass PIN for demos/testing
VA_BYPASS_PIN=1234
```

5. Run app:

```bash
python -m streamlit run app.py --server.port 8501
```

Open: http://localhost:8501

**Note:** First run takes ~30 seconds as models load into memory. Subsequent interactions are fast thanks to caching.

## Available Options

### Option A: Run with existing models (fastest)

Use the Quick Run steps above.

### Option B: Retrain wake word model

```bash
source .venv/bin/activate
python scripts/train_wakeword.py
```

Notes:
- Uses original files from class activity `wakeword_dataset/positive`, `wakeword_dataset/near`, `wakeword_dataset/other`
- Fast defaults: 30 epochs, early stopping patience 8, batch size 32

### Option C: Retrain NLU model

```bash
source .venv/bin/activate
python scripts/generate_nlu_dataset.py
python -m nlu.train --data_path data/nlu/train.full.json --output_dir models/joint_nlu
python eval_nlu.py
```

### Option D: Re-enroll speakers

```bash
source .venv/bin/activate
python scripts/enroll.py
python scripts/tune_threshold.py
```

## Required / Optional APIs

| Variable | Required | Where to get |
|---|---|---|
| `OMDB_API_KEY` | Yes | https://www.omdbapi.com/apikey.aspx |
| `GROQ_API_KEY` | No | https://console.groq.com (free, no card) |
| `HF_TOKEN` | No (HF deploy only) | https://huggingface.co/settings/tokens |
| `VA_BYPASS_PIN` | No | Set any PIN, default `1234` |

- No key needed: Open-Meteo (weather), MusicBrainz (music), Whisper ASR (offline)
- Internet required for gTTS output

## Main Commands

```bash
# run app
python -m streamlit run app.py --server.port 8501

# train wakeword
python scripts/train_wakeword.py

# train NLU
python -m nlu.train --data_path data/nlu/train.full.json --output_dir models/joint_nlu
```
