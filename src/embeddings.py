from dataclasses import dataclass
import importlib

import numpy as np
from scipy.signal import stft

from .config import SAMPLE_RATE

torch = None
EncoderClassifier = None
SPEECHBRAIN_AVAILABLE = False

try:
    torch = importlib.import_module("torch")
    speaker_module = importlib.import_module("speechbrain.inference.speaker")
    EncoderClassifier = getattr(speaker_module, "EncoderClassifier")
    SPEECHBRAIN_AVAILABLE = True
except Exception:
    SPEECHBRAIN_AVAILABLE = False


@dataclass
class SpeakerEmbedder:
    use_speechbrain: bool = True

    def __post_init__(self) -> None:
        self.model = None
        if self.use_speechbrain and SPEECHBRAIN_AVAILABLE:
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
            )

    def embed_waveform(self, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if wav.size == 0:
            raise ValueError("Audio is empty after preprocessing.")

        if self.model is not None:
            return self._speechbrain_embedding(wav)

        return self._fallback_embedding(wav, sample_rate)

    def _speechbrain_embedding(self, wav: np.ndarray) -> np.ndarray:
        wav = wav.astype(np.float32)
        signal = torch.from_numpy(wav).unsqueeze(0)
        emb = self.model.encode_batch(signal)
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)

    def _fallback_embedding(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        freqs, _, zxx = stft(wav, fs=sample_rate, nperseg=400, noverlap=240)
        mag = np.abs(zxx) + 1e-8
        log_mag = np.log(mag)

        mel_scale = np.log(1 + mag / 10.0)
        mel_mean = mel_scale.mean(axis=1)
        mel_std = mel_scale.std(axis=1)

        d_mel = np.diff(mel_scale, axis=0, prepend=mel_scale[:1])
        d_mel_mean = d_mel.mean(axis=1)
        d_mel_std = d_mel.std(axis=1)

        dd_mel = np.diff(d_mel, axis=0, prepend=d_mel[:1])
        dd_mel_mean = dd_mel.mean(axis=1)
        dd_mel_std = dd_mel.std(axis=1)

        compressed_mel_mean = self._compress_vector(mel_mean, output_dim=32)
        compressed_mel_std = self._compress_vector(mel_std, output_dim=32)
        compressed_d_mean = self._compress_vector(d_mel_mean, output_dim=16)
        compressed_d_std = self._compress_vector(d_mel_std, output_dim=16)
        compressed_dd_mean = self._compress_vector(dd_mel_mean, output_dim=8)
        compressed_dd_std = self._compress_vector(dd_mel_std, output_dim=8)

        abs_wav = np.abs(wav)
        frame_energy = np.sqrt(np.sum(np.abs(zxx) ** 2, axis=0))
        zcr = float(((wav[:-1] * wav[1:]) < 0).mean()) if wav.shape[0] > 1 else 0.0
        rms = float(np.sqrt(np.mean(wav**2)))
        energy_stats = np.percentile(frame_energy, [10, 25, 50, 75, 90]).astype(np.float32)

        spectral_centroid = float((freqs[:, None] * mag).sum() / mag.sum()) if mag.sum() > 0 else 0.0
        spectral_spread = float(np.sqrt((((freqs[:, None] - spectral_centroid) ** 2) * mag).sum() / mag.sum())) if mag.sum() > 0 else 0.0

        emb = np.concatenate(
            [
                compressed_mel_mean,
                compressed_mel_std,
                compressed_d_mean,
                compressed_d_std,
                compressed_dd_mean,
                compressed_dd_std,
                np.array([zcr, rms, spectral_centroid, spectral_spread], dtype=np.float32),
                energy_stats,
            ]
        ).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @staticmethod
    def _compress_vector(values: np.ndarray, output_dim: int) -> np.ndarray:
        if values.shape[0] == output_dim:
            return values.astype(np.float32)
        old_idx = np.linspace(0.0, 1.0, num=values.shape[0])
        new_idx = np.linspace(0.0, 1.0, num=output_dim)
        return np.interp(new_idx, old_idx, values).astype(np.float32)
