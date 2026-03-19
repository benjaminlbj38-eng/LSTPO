from __future__ import annotations

import torch
import torch.nn as nn

from src.losses.preference_losses import mse_per_sample
from src.models.backbones.transformer_forecaster import TimeSeriesTransformer


class PreferenceGenerator(nn.Module):
    """Reduced-scale Transformer used to generate long-vs-short preference pairs.

    Input
    -----
    long_history:  [B, L_long, C]
    short_history: [B, L_short, C]
    target:        [B, H, C]

    Output
    ------
    dict with:
      - pred_long: [B, H, C]
      - pred_short: [B, H, C]
      - mse_long: [B]
      - mse_short: [B]
      - prefer_long: [B] bool
      - step_prefers_long: [B, H] bool
      - generator_loss: scalar
    """

    def __init__(
        self,
        pred_len: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = TimeSeriesTransformer(
            pred_len=pred_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    def forward(
        self,
        long_history: torch.Tensor,
        short_history: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate preference labels and auxiliary predictions."""
        pred_long = self.backbone(long_history)     # [B, H, C]
        pred_short = self.backbone(short_history)   # [B, H, C]

        mse_long = mse_per_sample(pred_long, target)    # [B]
        mse_short = mse_per_sample(pred_short, target)  # [B]

        # Sequence-level preference for training.
        prefer_long = (mse_long.detach() <= mse_short.detach())

        # Step-level preference for dynamic preference visualization.
        step_err_long = ((pred_long - target) ** 2).mean(dim=2)    # [B, H]
        step_err_short = ((pred_short - target) ** 2).mean(dim=2)  # [B, H]
        step_prefers_long = (step_err_long.detach() <= step_err_short.detach())

        generator_loss = 0.5 * (mse_long.mean() + mse_short.mean())

        return {
            "pred_long": pred_long,
            "pred_short": pred_short,
            "mse_long": mse_long,
            "mse_short": mse_short,
            "prefer_long": prefer_long,
            "step_prefers_long": step_prefers_long,
            "generator_loss": generator_loss,
        }