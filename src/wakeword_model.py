"""CNN model for wake word detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WakeWordCNN(nn.Module):
    """Lightweight CNN for wake word detection (binary classification)."""

    def __init__(self, n_mels: int = 64, n_frames: int = 101):
        """Initialize wake word CNN.
        
        Args:
            n_mels: Number of mel-frequency bins
            n_frames: Number of frames in input spectrogram
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.n_frames = n_frames
        
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

        # Global Average Pooling: output is (batch, 128) regardless of input size.
        # This eliminates the flat_size dependency on n_frames, massively cuts params,
        # and generalises far better on small datasets than a large FC layer.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Small classifier head
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128, 64)
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

        x = self.gap(x)           # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1) # (batch, 128)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


