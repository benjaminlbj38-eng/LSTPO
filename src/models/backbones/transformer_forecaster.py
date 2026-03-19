from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Parameters
        ----------
        x:
            Tensor of shape [B, L, D].
        """
        return x + self.pe[:, :x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    """Channel-independent Transformer forecaster.

    The model shares the same parameters across channels, allowing one model to
    handle domains with different numbers of variables.

    Input
    -----
    x: [B, L, C]

    Output
    ------
    y_hat: [B, H, C]
    """

    def __init__(
        self,
        pred_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.input_proj = nn.Linear(1, d_model)
        self.positional = SinusoidalPositionalEncoding(d_model=d_model, max_len=4096)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            History tensor of shape [B, L, C].

        Returns
        -------
        torch.Tensor
            Forecast tensor of shape [B, H, C].
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input [B, L, C], got {tuple(x.shape)}")

        batch_size, seq_len, num_channels = x.shape

        # Channel-independent reshape:
        # [B, L, C] -> [B, C, L] -> [B*C, L, 1]
        tokens = x.permute(0, 2, 1).contiguous().view(batch_size * num_channels, seq_len, 1)

        # [B*C, L, 1] -> [B*C, L, D]
        tokens = self.input_proj(tokens)
        tokens = self.positional(tokens)

        # [B*C, L, D]
        encoded = self.encoder(tokens)

        # Pooled feature [B*C, D]
        pooled = 0.5 * (encoded.mean(dim=1) + encoded[:, -1, :])
        pooled = self.norm(pooled)

        # [B*C, H]
        out = self.head(pooled)

        # [B*C, H] -> [B, C, H] -> [B, H, C]
        out = out.view(batch_size, num_channels, self.pred_len).permute(0, 2, 1).contiguous()
        return out