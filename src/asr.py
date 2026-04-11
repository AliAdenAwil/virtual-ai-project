"""Automatic Speech Recognition (ASR) module using Whisper."""

from dataclasses import dataclass

import numpy as np
import torch

try:
    import whisper
except ImportError:  # pragma: no cover - handled at runtime in UI
    whisper = None


@dataclass
class ASRResult:
    """ASR output payload."""

    text: str
    language: str
    translated_text: str | None = None


class WhisperASR:
    """Whisper wrapper for waveform transcription."""

    def __init__(self, model_name: str = "base"):
        if whisper is None:
            raise ImportError(
                "openai-whisper is not installed. Install dependencies with: pip install -r requirements.txt"
            )
        self.model_name = model_name
        self.model = whisper.load_model(model_name)

    def transcribe_waveform(
        self,
        wav: np.ndarray,
        multilingual: bool = False,
        include_english_translation: bool = False,
    ) -> ASRResult:
        """Transcribe a waveform and optionally return English translation.

        Args:
            wav: 1D waveform sampled at 16kHz.
            multilingual: If False, force English-only decoding.
            include_english_translation: If True, add English translation output.
        """
        if wav.size == 0:
            return ASRResult(text="", language="unknown", translated_text=None)

        audio = wav.astype(np.float32)

        # Whisper needs at least ~0.5 s; pad to 1 s with silence if shorter
        min_samples = 16000
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        transcribe_kwargs = {
            "fp16": torch.cuda.is_available(),
            "task": "transcribe",
            "condition_on_previous_text": False,  # avoid hallucination loops
            "no_speech_threshold": 0.6,           # suppress "no"/"." on silence
            "compression_ratio_threshold": 2.4,
        }
        if not multilingual:
            transcribe_kwargs["language"] = "en"

        base_result = self.model.transcribe(audio, **transcribe_kwargs)
        text = base_result.get("text", "").strip()
        language = base_result.get("language", "unknown")

        translated_text = None
        if include_english_translation and multilingual and language != "en":
            translation_result = self.model.transcribe(
                audio,
                fp16=torch.cuda.is_available(),
                task="translate",
            )
            translated_text = translation_result.get("text", "").strip()

        return ASRResult(text=text, language=language, translated_text=translated_text)
