"""
TurboFault v0.1.0

transformer.py — Transformer encoder for RUL prediction.

Applies multi-head self-attention to sensor time-series windows.
Positional encoding captures cycle ordering without explicit recurrence.

Architecture:
    Input → Linear projection → Positional Encoding →
    TransformerEncoder (n_layers × [MultiHeadAttn + FFN]) →
    Mean Pool → FC(d_model → 64) → FC(64 → 1)

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger("turbofault")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerRUL(nn.Module):
    """
    Transformer encoder for Remaining Useful Life prediction.

    Args:
        input_dim: Number of input features per time step.
        d_model: Transformer model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of TransformerEncoder layers.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        max_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.2,
        max_len: int = 500,
    ):
        super().__init__()

        # Project input features to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Regression head
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            src_key_padding_mask: Optional mask for padded positions.

        Returns:
            RUL predictions of shape (batch, 1).
        """
        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling across sequence dimension
        if src_key_padding_mask is not None:
            # Mask out padded positions before averaging
            mask = ~src_key_padding_mask.unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)

        return self.fc(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# TurboFault v0.1.0
# Any usage is subject to this software's license.
