"""Training script for wake word detection model."""

from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio import load_audio_file
from src.wakeword_features import MelSpectrogramExtractor
from src.wakeword_model import WakeWordCNN
from src.config import SAMPLE_RATE, WAKEWORD_DATASET_DIR


def train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    """Simple train/test split using numpy (avoids sklearn dependency)."""
    np.random.seed(random_state)
    n_samples = len(X)
    test_count = int(n_samples * test_size)
    
    # Create indices and shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # If stratify, ensure class balance
    if stratify is not None:
        train_indices = []
        test_indices = []
        for class_label in np.unique(y):
            class_idx = np.where(y == class_label)[0]
            np.random.shuffle(class_idx)
            class_test_count = int(len(class_idx) * test_size)
            test_indices.extend(class_idx[:class_test_count])
            train_indices.extend(class_idx[class_test_count:])
        indices_train = np.array(train_indices)
        indices_test = np.array(test_indices)
    else:
        indices_train = indices[test_count:]
        indices_test = indices[:test_count]
    
    return X[indices_train], X[indices_test], y[indices_train], y[indices_test]


def _peak_energy_window(spec: np.ndarray, target_frames: int) -> np.ndarray:
    """Select the highest-energy window of target_frames from a spectrogram.

    Instead of always taking the first N frames, this finds where the wake word
    actually occurs (the loudest region), reducing silence padding at inference.

    Args:
        spec: Mel-spectrogram (n_mels, n_frames)
        target_frames: Desired output width in frames

    Returns:
        Cropped spectrogram of shape (n_mels, target_frames)
    """
    n_frames = spec.shape[1]
    if n_frames <= target_frames:
        # Pad right with zeros
        pad = target_frames - n_frames
        return np.pad(spec, ((0, 0), (0, pad)), mode='constant')

    step = max(1, target_frames // 4)
    best_start = 0
    best_energy = -1.0
    for start in range(0, n_frames - target_frames + 1, step):
        energy = float(np.mean(spec[:, start:start + target_frames] ** 2))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return spec[:, best_start:best_start + target_frames]


def load_and_process_data(dataset_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    """Load audio files and extract features.

    Uses positive/ as wakeword positives and both near/ and other/ as negatives.

    Args:
        dataset_dir: Path to dataset with positive/, near/, and other/ folders

    Returns:
        Tuple of (spectrograms, labels) where label 1 = positive, 0 = negative
    """
    extractor = MelSpectrogramExtractor(sample_rate=SAMPLE_RATE, n_mels=64)
    spectrograms = []
    labels = []

    def iter_audio_files(folder: Path):
        patterns = ("*.wav", "*.m4a", "*.mp3", "*.aac")
        for pattern in patterns:
            for audio_file in sorted(folder.glob(pattern)):
                yield audio_file

    # Load positive samples (label = 1)
    positive_dir = dataset_dir / "positive"
    if positive_dir.exists():
        for audio_file in iter_audio_files(positive_dir):
            try:
                wav = load_audio_file(audio_file)
                spec = extractor.extract(wav, normalize=True)
                spectrograms.append(spec)
                labels.append(1)
                print(f"✓ Loaded positive: {audio_file.name} ({spec.shape})")
            except Exception as e:
                print(f"✗ Error loading {audio_file.name}: {e}")
    else:
        print(f"Warning: {positive_dir} not found")

    # Load near-miss negatives (label = 0)
    near_dir = dataset_dir / "near"
    if near_dir.exists():
        for audio_file in iter_audio_files(near_dir):
            try:
                wav = load_audio_file(audio_file)
                spec = extractor.extract(wav, normalize=True)
                spectrograms.append(spec)
                labels.append(0)
                print(f"✓ Loaded near: {audio_file.name} ({spec.shape})")
            except Exception as e:
                print(f"✗ Error loading {audio_file.name}: {e}")

    # Load negative samples from other/ folder (label = 0)
    other_dir = dataset_dir / "other"
    if other_dir.exists():
        for audio_file in iter_audio_files(other_dir):
            try:
                wav = load_audio_file(audio_file)
                spec = extractor.extract(wav, normalize=True)
                spectrograms.append(spec)
                labels.append(0)
                print(f"✓ Loaded other: {audio_file.name} ({spec.shape})")
            except Exception as e:
                print(f"✗ Error loading {audio_file.name}: {e}")

    if not spectrograms:
        raise ValueError(f"No audio files found in {dataset_dir}")

    print(f"\nLoaded {len(spectrograms)} samples:")
    print(f"  Positive: {sum(1 for l in labels if l == 1)}")
    print(f"  Negative: {sum(1 for l in labels if l == 0)}")

    return spectrograms, labels


def prepare_batch(spectrograms: list[np.ndarray], target_frames: int = 101) -> np.ndarray:
    """Prepare batch using peak-energy window selection.

    Args:
        spectrograms: List of mel-spectrograms
        target_frames: Number of frames to standardize to

    Returns:
        Batch array (n_samples, 1, n_mels, target_frames)
    """
    batch = []
    for spec in spectrograms:
        spec_windowed = _peak_energy_window(spec, target_frames)
        batch.append(spec_windowed)

    batch_array = np.stack(batch, axis=0)           # (n, n_mels, n_frames)
    batch_array = np.expand_dims(batch_array, axis=1)  # (n, 1, n_mels, n_frames)
    return batch_array


def train_model(
    model: WakeWordCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    device: str = "cpu",
    pos_weight: float = 1.0,
) -> dict:
    """Train wake word detection model."""
    import copy
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    # pos_weight compensates for class imbalance (neg_count / pos_count)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    patience = 8
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float().unsqueeze(1)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        # Save best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best weights before returning
    model.load_state_dict(best_model_weights)
    print(f"Restored best model (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
    return history


def main():
    """Main training pipeline."""
    # Configuration
    dataset_dir = WAKEWORD_DATASET_DIR
    model_dir = ROOT / "data" / "wakeword_models"
    model_path = model_dir / "wakeword_cnn.pt"
    config_path = model_dir / "wakeword_config.pkl"
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load and process data
    print("\n=== Loading and preprocessing audio ===")
    spectrograms, labels = load_and_process_data(dataset_dir)
    
    # Determine target frames: cap at 110 to avoid excessive silence padding
    frame_lengths = [s.shape[1] for s in spectrograms]
    target_frames = int(np.percentile(frame_lengths, 90))
    target_frames = min(max(target_frames, 100), 110)
    print(f"\nFrame length stats — min:{min(frame_lengths)} median:{int(np.median(frame_lengths))} 90pct:{target_frames} max:{max(frame_lengths)}")
    print(f"Standardizing spectrograms to {target_frames} frames (peak-energy window)")

    temporal_bins = 5

    # Prepare batch with peak-energy window selection
    X = prepare_batch(spectrograms, target_frames=target_frames)
    y = np.array(labels)

    print(f"\nBatch shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Split data (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Val set: {X_val.shape[0]} samples")

    print(f"Using original files only (no augmentation) — train: {len(X_train)}, val: {len(X_val)}")

    # Compute pos_weight to handle class imbalance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Train class balance — positive: {n_pos}, negative: {n_neg}, pos_weight: {pos_weight:.2f}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model with temporal bins
    print("\n=== Initializing model ===")
    model = WakeWordCNN(n_mels=64, n_frames=target_frames, temporal_bins=temporal_bins)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("\n=== Training model ===")
    history = train_model(model, train_loader, val_loader, num_epochs=30, device=device, pos_weight=pos_weight)
    
    # Save model
    print(f"\n=== Saving model ===")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Save configuration
    config = {
        'n_mels': 64,
        'n_frames': target_frames,
        'temporal_bins': temporal_bins,
        'sample_rate': SAMPLE_RATE,
        'device': device,
    }
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"Config saved to: {config_path}")
    
    # Print final metrics
    print(f"\n=== Final Results ===")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
