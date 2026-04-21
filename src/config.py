from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_RECORDINGS_DIR = ROOT_DIR / "raw_recordings"
VOICEPRINT_STORE_PATH = ROOT_DIR / "data" / "voiceprints" / "voiceprint_store.pkl"
GUEST_VOICEPRINT_STORE_PATH = ROOT_DIR / "data" / "voiceprints" / "guest_store.pkl"
GUEST_ENROLLMENT_TTL_DAYS = 7
LOG_PATH = ROOT_DIR / "logs" / "verification_log.jsonl"

# Wake word detection paths
WAKEWORD_DATASET_DIR = ROOT_DIR / "wakeword_dataset"
WAKEWORD_MODEL_PATH = ROOT_DIR / "data" / "wakeword_models" / "wakeword_cnn.pt"
WAKEWORD_CONFIG_PATH = ROOT_DIR / "data" / "wakeword_models" / "wakeword_config.pkl"
WAKEWORD_DETECTION_THRESHOLD = 0.52
WAKEWORD_PHRASE = "Hey Atlas"
# Use Whisper transcript as a gate for wakeword acceptance.
# Keep True to reduce false positives like "hey google/alexa".
WHISPER_WAKEWORD_GATE = True

# ASR (Whisper)
ASR_MODEL_NAME = "small"
ASR_RECORD_SECONDS = 5
ASR_MULTILINGUAL_ENABLED = False
ASR_TRANSLATE_TO_ENGLISH = False

SAMPLE_RATE = 16000
CHANNELS = 1
DEFAULT_RECORD_SECONDS = 3
# Voice verification fallback used only when no tuned threshold is found in voiceprint_store.pkl.
VOICE_VERIFICATION_FALLBACK_THRESHOLD = 0.30
# Backward compatibility alias; prefer VOICE_VERIFICATION_FALLBACK_THRESHOLD in code.
DEFAULT_THRESHOLD = VOICE_VERIFICATION_FALLBACK_THRESHOLD
BYPASS_PIN_ENV_KEY = "VA_BYPASS_PIN"
