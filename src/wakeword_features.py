"""Feature extraction for wake word detection."""

import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft


class MelSpectrogramExtractor:
    """Extract mel-spectrogram features from audio waveforms."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
        f_min: int = 50,
        f_max: int = 8000,
    ):
        """Initialize mel-spectrogram extractor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel-frequency bins
            f_min: Minimum frequency in Hz
            f_max: Maximum frequency in Hz
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        
        # Precompute mel filter bank
        self.mel_filterbank = self._create_mel_filterbank()
        
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel-scale filter bank."""
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        
        # Convert frequency limits to mel scale
        f_min_mel = self._hz_to_mel(self.f_min)
        f_max_mel = self._hz_to_mel(self.f_max)
        
        # Create mel-spaced frequencies
        mel_freqs = np.linspace(f_min_mel, f_max_mel, self.n_mels + 2)
        hz_freqs = self._mel_to_hz(mel_freqs)
        
        # Create triangular filters
        filterbank = np.zeros((self.n_mels, len(freqs)))
        for i in range(self.n_mels):
            left = hz_freqs[i]
            center = hz_freqs[i + 1]
            right = hz_freqs[i + 2]
            
            # Left slope
            left_slope = (freqs - left) / (center - left) if center != left else np.zeros_like(freqs)
            # Right slope
            right_slope = (right - freqs) / (right - center) if right != center else np.zeros_like(freqs)
            
            # Construct filter
            filterbank[i] = np.maximum(0, np.minimum(left_slope, right_slope))
        
        return filterbank
    
    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        """Convert frequency in Hz to mel scale."""
        return 2595 * np.log10(1 + hz / 700)
    
    @staticmethod
    def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
        """Convert mel scale to frequency in Hz."""
        return 700 * (10 ** (mel / 2595) - 1)
    
    def extract(self, wav: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Extract mel-spectrogram from waveform.
        
        Args:
            wav: Audio waveform (1D array)
            normalize: Whether to normalize the spectrogram
            
        Returns:
            Mel-spectrogram (n_mels, n_frames)
        """
        # Pad audio if too short
        if len(wav) < self.n_fft:
            wav = np.pad(wav, (0, self.n_fft - len(wav)), mode='constant')
        
        # Apply Hann window and compute STFT
        window = get_window('hann', self.n_fft)
        spec = np.zeros((len(self.mel_filterbank), 0))
        
        for start in range(0, len(wav) - self.n_fft + 1, self.hop_length):
            frame = wav[start : start + self.n_fft]
            
            # Apply window and FFT
            windowed = frame * window
            mag_spec = np.abs(rfft(windowed))
            
            # Apply mel filterbank
            mel_spec = self.mel_filterbank @ mag_spec
            spec = np.column_stack([spec, mel_spec])
        
        # Apply log scaling
        spec = np.log(np.maximum(spec, 1e-10))
        
        # Normalize
        if normalize:
            spec = (spec - spec.mean()) / (spec.std() + 1e-10)
        
        return spec.astype(np.float32)


def pad_or_truncate_spectrogram(spec: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad or truncate spectrogram to target frame count.
    
    Args:
        spec: Mel-spectrogram (n_mels, n_frames)
        target_frames: Target number of frames
        
    Returns:
        Spectrogram with target frame count (n_mels, target_frames)
    """
    n_frames = spec.shape[1]
    
    if n_frames < target_frames:
        # Pad with last frame
        pad_amount = target_frames - n_frames
        padding = np.tile(spec[:, -1:], (1, pad_amount))
        return np.column_stack([spec, padding]).astype(np.float32)
    elif n_frames > target_frames:
        # Truncate
        return spec[:, :target_frames].astype(np.float32)
    else:
        return spec.astype(np.float32)
