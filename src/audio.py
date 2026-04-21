import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pydub import AudioSegment
from dotenv import load_dotenv

from .config import CHANNELS, SAMPLE_RATE

# Set DEPLOY_MODE=hf to use browser audio input instead of sounddevice.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

_HF_MODE = os.environ.get("DEPLOY_MODE", "").lower() == "hf"

if not _HF_MODE:
    import sounddevice as sd


def capture_microphone(seconds: int, sample_rate: int = SAMPLE_RATE, preprocess: bool = True) -> np.ndarray:
    recording = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    audio = recording[:, 0]
    if preprocess:
        return preprocess_audio(audio, sample_rate)
    # For ASR: only normalize, do not trim (Whisper has its own VAD)
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def audio_input_to_numpy(uploaded_bytes, target_sr: int = SAMPLE_RATE, preprocess: bool = True) -> np.ndarray:
    """Convert st.audio_input() bytes to a float32 numpy array at target_sr."""
    raw = uploaded_bytes.read() if hasattr(uploaded_bytes, "read") else bytes(uploaded_bytes.getvalue())
    samples, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != target_sr:
        samples = resample_poly(samples, target_sr, sr)
    if preprocess:
        return preprocess_audio(samples, target_sr)
    samples = samples.astype(np.float32)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak
    return samples


def is_hf_mode() -> bool:
    return _HF_MODE


def load_audio_file(path: Path, target_sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    path = Path(path)
    
    # Handle .m4a and other audio formats that soundfile doesn't support
    if path.suffix.lower() in ['.m4a', '.mp3', '.aac']:
        try:
            # Try using pydub to load the file
            audio = AudioSegment.from_file(str(path), format=path.suffix[1:])
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)
            # Normalize to [-1, 1] range for 16-bit audio
            samples = samples / 32768.0
            sr = audio.frame_rate
        except Exception as e:
            print(f"Warning: Failed to load {path} with pydub: {e}")
            # Fallback: try to use soundfile anyway
            samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
    else:
        # Use soundfile for .wav and other supported formats
        samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
    
    if sr != target_sample_rate:
        samples = resample_poly(samples, target_sample_rate, sr)
    # Only normalize (no aggressive VAD trim) so short utterances stay intact
    samples = samples.astype(np.float32)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak
    return samples


def preprocess_audio(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if wav.size == 0:
        return wav

    wav = wav.astype(np.float32)
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak

    abs_wav = np.abs(wav)
    threshold = max(0.01, float(np.percentile(abs_wav, 75) * 0.2))
    idx = np.where(abs_wav >= threshold)[0]
    if idx.size > 0:
        start = max(0, idx[0] - int(0.1 * sample_rate))
        end = min(len(wav), idx[-1] + int(0.1 * sample_rate))
        wav = wav[start:end]

    return wav
