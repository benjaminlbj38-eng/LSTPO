from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute MSE / MAE on one batch."""
    return {
        "mse": float(F.mse_loss(pred, target).item()),
        "mae": float(F.l1_loss(pred, target).item()),
    }


def aggregate_metric_list(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of metric dictionaries."""
    if not metrics_list:
        return {"mse": float("nan"), "mae": float("nan")}
    keys = metrics_list[0].keys()
    return {key: float(np.mean([m[key] for m in metrics_list])) for key in keys}


def aggregate_domain_metrics(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Aggregate metrics from domain-wise results."""
    metric_list = [domain_result["metrics"] for domain_result in results.values()]
    return aggregate_metric_list(metric_list)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)