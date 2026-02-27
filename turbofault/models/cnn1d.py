"""
TurboFault v0.1.0

cnn1d.py — 1D Convolutional Neural Network for RUL prediction.

1D-CNNs extract local temporal patterns (e.g., sudden sensor spikes,
short-term degradation signatures) that complement LSTM/Transformer models.

Architecture:
    Input → Conv1D blocks × 3 (with BatchNorm + ReLU + MaxPool) →
    AdaptiveAvgPool → FC(channels → 64) → FC(64 → 1)

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("turbofault")


class ConvBlock(nn.Module):
    """Conv1D → BatchNorm → ReLU → MaxPool block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1DModel(nn.Module):
    """
    1D-CNN for RUL prediction from sensor windows.

    Input shape: (batch, seq_len, n_features)
    Internally transposes to (batch, n_features, seq_len) for Conv1D.

    Args:
        input_dim: Number of input features (sensor channels).
        channels: List of output channels per conv block.
        kernel_sizes: Kernel size per conv block.
        pool_sizes: Max pool size per block.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        channels: tuple[int, ...] = (64, 128, 256),
        kernel_sizes: tuple[int, ...] = (7, 5, 3),
        pool_sizes: tuple[int, ...] = (2, 2, 2),
        dropout: float = 0.3,
    ):
        super().__init__()

        assert (
            len(channels) == len(kernel_sizes) == len(pool_sizes)
        ), "channels, kernel_sizes, and pool_sizes must have the same length"

        # Build conv blocks
        blocks = []
        in_ch = input_dim
        for out_ch, ks, ps in zip(channels, kernel_sizes, pool_sizes, strict=True):
            blocks.append(ConvBlock(in_ch, out_ch, ks, ps))
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*blocks)

        # Adaptive pooling ensures fixed output regardless of input length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, n_features).

        Returns:
            RUL predictions (batch, 1).
        """
        # Transpose to (batch, n_features, seq_len) for Conv1D
        x = x.permute(0, 2, 1)

        x = self.conv_layers(x)  # (batch, channels[-1], reduced_len)
        x = self.adaptive_pool(x)  # (batch, channels[-1], 1)
        x = x.squeeze(-1)  # (batch, channels[-1])

        return self.fc(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# TurboFault v0.1.0
# Any usage is subject to this software's license.
