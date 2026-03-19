from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.losses.preference_losses import log_prob_from_prediction
from src.models.backbones.dlinear import DLinear
from src.models.backbones.transformer_forecaster import TimeSeriesTransformer
from src.models.modules.preference import PreferenceGenerator


class LSTPOSystem(nn.Module):
    """Top-level LSTPO system wrapper.

    It combines:
    - base forecasting backbone
    - reduced preference generator
    - helper functions for long/short scoring
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.pred_len = int(cfg["data"]["pred_len"])
        self.temperature = float(cfg["model"]["pref_temperature"])

        backbone_name = cfg["model"]["backbone"].lower()
        if backbone_name == "transformer":
            self.backbone = TimeSeriesTransformer(
                pred_len=self.pred_len,
                d_model=int(cfg["model"]["d_model"]),
                nhead=int(cfg["model"]["nhead"]),
                num_layers=int(cfg["model"]["num_layers"]),
                ff_dim=int(cfg["model"]["ff_dim"]),
                dropout=float(cfg["model"]["dropout"]),
            )
        elif backbone_name == "dlinear":
            self.backbone = DLinear(
                seq_len=int(cfg["data"]["history_len"]),
                pred_len=self.pred_len,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.preference_generator = PreferenceGenerator(
            pred_len=self.pred_len,
            d_model=int(cfg["model"]["pref_d_model"]),
            nhead=int(cfg["model"]["pref_nhead"]),
            num_layers=int(cfg["model"]["pref_layers"]),
            ff_dim=int(cfg["model"]["pref_ff_dim"]),
            dropout=float(cfg["model"]["pref_dropout"]),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Forecast future values from full history.

        Parameters
        ----------
        history:
            Tensor of shape [B, L, C].

        Returns
        -------
        torch.Tensor
            Tensor of shape [B, H, C].
        """
        return self.backbone(history)

    def split_views(
        self,
        history: torch.Tensor,
        window_spec: dict[str, int | float | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split full history into long and short views.

        Returns
        -------
        long_history:
            [B, L_long, C]
        short_history:
            [B, L_short, C]
        """
        long_len = int(window_spec["long_len"])
        short_len = int(window_spec["short_len"])

        long_history = history[:, -long_len:, :]
        short_history = history[:, -short_len:, :]
        return long_history, short_history

    def generate_preferences(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        window_spec: dict[str, int | float | None],
    ) -> dict[str, torch.Tensor]:
        """Run the preference generator on long/short views."""
        long_history, short_history = self.split_views(history, window_spec)
        return self.preference_generator(long_history, short_history, target)

    def score_views(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        window_spec: dict[str, int | float | None],
        backbone: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute long-view / short-view forecasts and their log-prob scores."""
        model = backbone if backbone is not None else self.backbone
        long_history, short_history = self.split_views(history, window_spec)

        pred_long = model(long_history)     # [B, H, C]
        pred_short = model(short_history)   # [B, H, C]

        logp_long = log_prob_from_prediction(target, pred_long, self.temperature)     # [B]
        logp_short = log_prob_from_prediction(target, pred_short, self.temperature)   # [B]

        return {
            "pred_long": pred_long,
            "pred_short": pred_short,
            "logp_long": logp_long,
            "logp_short": logp_short,
        }

    def selected_prediction(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        prefer_long: torch.Tensor,
        window_spec: dict[str, int | float | None],
        backbone: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the prediction from the preferred branch."""
        scores = self.score_views(history, target, window_spec, backbone=backbone)
        selected = torch.where(prefer_long[:, None, None], scores["pred_long"], scores["pred_short"])
        return selected, scores