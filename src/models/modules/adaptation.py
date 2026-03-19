from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from src.losses.preference_losses import adaptation_dpo_loss_from_logps


def adapt_model(
    system: nn.Module,
    reference_backbone: nn.Module,
    adapt_loader: Any,
    window_spec: dict[str, int | float | None],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, float]]:
    """Adapt the fine-tuned backbone to a target domain.

    Parameters
    ----------
    system:
        LSTPOSystem wrapper. Used for split_views / score_views / preference generation.
    reference_backbone:
        Frozen fine-tuned model used as DPO reference.
    adapt_loader:
        Validation / adaptation loader from the target domain.
    window_spec:
        Per-domain long/short window settings.
    cfg:
        Configuration dictionary.
    device:
        Target device.

    Returns
    -------
    adapted_backbone:
        A domain-adapted backbone copy.
    stats:
        Adaptation statistics.
    """
    adapted_backbone = deepcopy(system.backbone).to(device)
    reference = deepcopy(reference_backbone).to(device)

    for param in reference.parameters():
        param.requires_grad = False
    reference.eval()

    optimizer = torch.optim.Adam(
        adapted_backbone.parameters(),
        lr=float(cfg["train"]["adapt_lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    max_steps = int(cfg["train"]["adapt_steps"])
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    stats = {"adapt_loss": 0.0, "steps": 0.0}

    system.preference_generator.eval()
    adapted_backbone.train()

    for batch in adapt_loader:
        if stats["steps"] >= max_steps:
            break

        history = batch["history"].to(device)
        target = batch["target"].to(device)

        with torch.no_grad():
            pref_info = system.generate_preferences(history, target, window_spec)
            ref_scores = system.score_views(history, target, window_spec, backbone=reference)

        model_scores = system.score_views(history, target, window_spec, backbone=adapted_backbone)

        loss = adaptation_dpo_loss_from_logps(
            model_logp_long=model_scores["logp_long"],
            model_logp_short=model_scores["logp_short"],
            ref_logp_long=ref_scores["logp_long"],
            ref_logp_short=ref_scores["logp_short"],
            prefer_long=pref_info["prefer_long"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(adapted_backbone.parameters(), max_norm=grad_clip)
        optimizer.step()

        stats["adapt_loss"] += float(loss.item())
        stats["steps"] += 1.0

    if stats["steps"] > 0:
        stats["adapt_loss"] /= stats["steps"]

    adapted_backbone.eval()
    return adapted_backbone, stats