from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_RECORDINGS_DIR = ROOT_DIR / "raw_recordings"
VOICEPRINT_STORE_PATH = ROOT_DIR / "data" / "voiceprints" / "voiceprint_store.pkl"
LOG_PATH = ROOT_DIR / "logs" / "verification_log.jsonl"

# Wake word detection paths
WAKEWORD_DATASET_DIR = ROOT_DIR / "wakeword_dataset"
WAKEWORD_MODEL_PATH = ROOT_DIR / "data" / "wakeword_models" / "wakeword_cnn.pt"
WAKEWORD_CONFIG_PATH = ROOT_DIR / "data" / "wakeword_models" / "wakeword_config.pkl"
WAKEWORD_DETECTION_THRESHOLD = 0.45
WAKEWORD_PHRASE = "Hey Atlas"

# ASR (Whisper)
ASR_MODEL_NAME = "small"
ASR_RECORD_SECONDS = 5
ASR_MULTILINGUAL_ENABLED = False
ASR_TRANSLATE_TO_ENGLISH = False

SAMPLE_RATE = 16000
CHANNELS = 1
DEFAULT_RECORD_SECONDS = 3
DEFAULT_THRESHOLD = 0.30
BYPASS_PIN_ENV_KEY = "VA_BYPASS_PIN"
