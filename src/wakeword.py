"""Wake word detection module for real-time inference."""

from pathlib import Path
from dataclasses import dataclass
import pickle
import numpy as np
import torch
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .wakeword_features import MelSpectrogramExtractor, pad_or_truncate_spectrogram
from .config import SAMPLE_RATE, LOG_PATH
from .logger import log_event

# Dynamic import for wakeword model
try:
    from .wakeword_model import WakeWordCNN
    WAKEWORD_MODEL_AVAILABLE = True
except ImportError:
    WAKEWORD_MODEL_AVAILABLE = False


@dataclass
class WakeWordDetectionResult:
    """Result of wake word detection."""
    detected: bool
    confidence: float
    frame_count: int


class WakeWordDetector:
    """Real-time wake word detector using trained CNN."""

    def __init__(self, model_path: Path | None = None, config_path: Path | None = None):
        """Initialize wake word detector.
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to model configuration
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.config = None
        self.extractor = None
        self.is_ready = False
        
        if model_path and config_path:
            self.load_model(model_path, config_path)
    
    def load_model(self, model_path: Path, config_path: Path) -> None:
        """Load trained model and configuration.
        
        Args:
            model_path: Path to model weights (.pt file)
            config_path: Path to config pickle file
        """
        if not WAKEWORD_MODEL_AVAILABLE:
            print("Warning: Neural network modules not available for wake word detection")
            return
        
        try:
            # Load configuration
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            
            # Initialize model
            self.model = WakeWordCNN(
                n_mels=self.config['n_mels'],
                n_frames=self.config['n_frames']
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize feature extractor
            self.extractor = MelSpectrogramExtractor(
                sample_rate=self.config['sample_rate'],
                n_mels=self.config['n_mels']
            )
            
            self.is_ready = True
            print(f"✓ Wake word detector loaded from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load wake word model: {e}")
            self.is_ready = False
    
    def detect_waveform(
        self, wav: np.ndarray, threshold: float = 0.6
    ) -> WakeWordDetectionResult:
        """Detect wake word in audio waveform.
        
        Args:
            wav: Audio waveform (1D array, 16-bit PCM)
            threshold: Detection threshold (0.0-1.0)
            
        Returns:
            WakeWordDetectionResult with detection status and confidence
        """
        if not self.is_ready or self.model is None:
            return WakeWordDetectionResult(detected=False, confidence=0.0, frame_count=0)
        
        try:
            # Extract mel-spectrogram
            spec = self.extractor.extract(wav, normalize=True)
            
            # Pad/truncate to model input size
            spec_padded = pad_or_truncate_spectrogram(spec, self.config['n_frames'])
            
            # Add batch dimension: (1, 1, n_mels, n_frames)
            spec_batch = np.expand_dims(spec_padded, axis=0)
            spec_batch = np.expand_dims(spec_batch, axis=0)
            
            # Run inference
            with torch.no_grad():
                spec_tensor = torch.from_numpy(spec_batch).float().to(self.device)
                logits = self.model(spec_tensor)
                confidence = torch.sigmoid(logits).item()
            
            # Make decision
            detected = confidence >= threshold
            
            result = WakeWordDetectionResult(
                detected=detected,
                confidence=confidence,
                frame_count=spec.shape[1]
            )
            
            # Log detection event
            log_event(
                LOG_PATH,
                {
                    "event": "wake_word_detection",
                    "detected": detected,
                    "confidence": confidence,
                    "threshold": threshold,
                    "wakeword": "Hey Atlas",
                }
            )
            
            return result
        except Exception as e:
            print(f"Error detecting wake word: {e}")
            return WakeWordDetectionResult(detected=False, confidence=0.0, frame_count=0)
    
    def detect_streaming(
        self,
        audio_buffer: np.ndarray,
        frame_size: int = 16000,
        threshold: float = 0.6,
    ) -> WakeWordDetectionResult:
        """Detect wake word in streaming audio (sliding window).
        
        Args:
            audio_buffer: Accumulated audio buffer
            frame_size: Size of frame to process (samples)
            threshold: Detection threshold
            
        Returns:
            WakeWordDetectionResult if audio buffer is large enough
        """
        if len(audio_buffer) < frame_size:
            return WakeWordDetectionResult(detected=False, confidence=0.0, frame_count=0)
        
        # Process last frame_size samples
        frame = audio_buffer[-frame_size:]
        return self.detect_waveform(frame, threshold=threshold)


class SimpleWakeWordDetector:
    """Fallback template-matching based wake word detector."""

    def __init__(self):
        """Initialize fallback detector."""
        self.extractor = MelSpectrogramExtractor(sample_rate=SAMPLE_RATE, n_mels=64)
        self.reference_specs = []
        self.is_ready = False
    
    def add_reference(self, wav: np.ndarray) -> None:
        """Add reference positive sample.
        
        Args:
            wav: Audio waveform of wake word
        """
        spec = self.extractor.extract(wav, normalize=True)
        self.reference_specs.append(spec)
    
    def detect_waveform(
        self, wav: np.ndarray, similarity_threshold: float = 0.7
    ) -> WakeWordDetectionResult:
        """Detect wake word using template matching.
        
        Args:
            wav: Audio waveform to test
            similarity_threshold: Similarity threshold for detection
            
        Returns:
            WakeWordDetectionResult
        """
        if not self.reference_specs:
            return WakeWordDetectionResult(detected=False, confidence=0.0, frame_count=0)
        
        try:
            spec = self.extractor.extract(wav, normalize=True)
            
            # Compare with all reference specs
            max_similarity = 0.0
            for ref_spec in self.reference_specs:
                # Compute average cosine similarity across frames
                sim = self._compute_similarity(spec, ref_spec)
                max_similarity = max(max_similarity, sim)
            
            detected = max_similarity >= similarity_threshold
            
            return WakeWordDetectionResult(
                detected=detected,
                confidence=max_similarity,
                frame_count=spec.shape[1]
            )
        except Exception as e:
            print(f"Error in simple detection: {e}")
            return WakeWordDetectionResult(detected=False, confidence=0.0, frame_count=0)
    
    @staticmethod
    def _compute_similarity(spec1: np.ndarray, spec2: np.ndarray) -> float:
        """Compute similarity between two spectrograms."""
        # Flatten spectrograms
        s1 = spec1.flatten()
        s2 = spec2.flatten()
        
        # Cosine similarity
        norm1 = np.linalg.norm(s1)
        norm2 = np.linalg.norm(s2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(s1, s2) / (norm1 * norm2))


def start_wakeword_listener(context: dict | None = None) -> None:
    """Start wake word listener (hook for state machine).
    
    Args:
        context: Optional context dictionary
    """
    log_event(
        LOG_PATH,
        {
            "event": "wakeword_listener_started",
            "wakeword": "Hey Atlas",
        }
    )
