"""
TurboFault v0.1.0

lstm.py — LSTM sequence model for RUL prediction.

Long Short-Term Memory networks naturally capture temporal dependencies
in sensor degradation patterns. Bidirectional variant available.

Architecture:
    Input → LSTM(hidden) × n_layers → Dropout → FC(hidden → 64) → FC(64 → 1)

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("turbofault")


class LSTMModel(nn.Module):
    """
    LSTM-based RUL prediction model.

    Args:
        input_dim: Number of input features per time step.
        hidden_dim: LSTM hidden state size.
        n_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_input = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            RUL predictions of shape (batch, 1).
        """
        # LSTM output: (batch, seq_len, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(x)

        # Use the last time step's output
        last_output = lstm_out[:, -1, :]

        return self.fc(last_output)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRUModel(nn.Module):
    """
    GRU-based RUL prediction model (lighter alternative to LSTM).

    Args:
        input_dim: Number of input features per time step.
        hidden_dim: GRU hidden state size.
        n_layers: Number of stacked GRU layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# TurboFault v0.1.0
# Any usage is subject to this software's license.
