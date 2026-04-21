"""CNN model for wake word detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WakeWordCNN(nn.Module):
    """CNN for wake word detection with temporal bins (preserves phoneme sequence)."""

    def __init__(self, n_mels: int = 64, n_frames: int = 101, temporal_bins: int = 5):
        """Initialize wake word CNN.

        Args:
            n_mels: Number of mel-frequency bins
            n_frames: Number of frames in input spectrogram
            temporal_bins: Number of temporal buckets after pooling
        """
        super().__init__()

        self.n_mels = n_mels
        self.n_frames = n_frames
        self.temporal_bins = temporal_bins

        # Convolutional blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        # Temporal pooling: collapses freq dim fully, keeps temporal_bins time slots.
        # Preserves "hey" -> "atlas" ordering — unlike Global Average Pooling which
        # erases temporal structure, causing "hey google" to score same as "hey atlas".
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, temporal_bins))

        flat_dim = 128 * temporal_bins  # 128 * 5 = 640

        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input spectrogram (batch_size, 1, n_mels, n_frames)

        Returns:
            Logits for binary classification (batch_size, 1)
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = self.temporal_pool(x)          # (batch, 128, 1, temporal_bins)
        x = x.view(x.size(0), -1)          # (batch, 128 * temporal_bins)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


