from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from src.evaluation.metrics import aggregate_metric_list, batch_metrics
from src.losses.preference_losses import selected_supervised_mse, tpo_loss_from_logps


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move batch tensors to device."""
    return batch["history"].to(device), batch["target"].to(device)


def next_batch(iterator: Any, loader: Any) -> tuple[dict[str, Any], Any]:
    """Cycle a DataLoader iterator."""
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def split_support_query(
    history: torch.Tensor,
    target: torch.Tensor,
    prefer_long: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Split a batch into support/query subsets for inner/outer meta-updates."""
    batch_size = history.size(0)
    if batch_size == 1:
        return {
            "support_history": history,
            "support_target": target,
            "support_prefers_long": prefer_long,
            "query_history": history,
            "query_target": target,
            "query_prefers_long": prefer_long,
        }

    permutation = torch.randperm(batch_size, device=history.device)
    split = max(1, batch_size // 2)
    if split >= batch_size:
        split = batch_size - 1

    support_idx = permutation[:split]
    query_idx = permutation[split:]

    return {
        "support_history": history[support_idx],
        "support_target": target[support_idx],
        "support_prefers_long": prefer_long[support_idx],
        "query_history": history[query_idx],
        "query_target": target[query_idx],
        "query_prefers_long": prefer_long[query_idx],
    }


def compute_backbone_objective(
    system: torch.nn.Module,
    backbone: torch.nn.Module,
    history: torch.Tensor,
    target: torch.Tensor,
    prefer_long: torch.Tensor,
    window_spec: dict[str, int | float | None],
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Compute backbone training objective for one batch.

    Modes:
    - full-history supervised baseline
    - TPO
    - selected-branch supervised ablation (w/o TPO)
    """
    if cfg["model"].get("train_on_full_history_only", False):
        pred = backbone(history)
        loss = F.mse_loss(pred, target)
        metrics = batch_metrics(pred, target)
        metrics["objective"] = float(loss.item())
        aux = {"pred": pred}
        return loss, metrics, aux

    scores = system.score_views(history, target, window_spec, backbone=backbone)
    if cfg["ablation"]["use_tpo"]:
        loss = tpo_loss_from_logps(
            logp_long=scores["logp_long"],
            logp_short=scores["logp_short"],
            prefer_long=prefer_long,
        )
    else:
        loss = selected_supervised_mse(
            pred_long=scores["pred_long"],
            pred_short=scores["pred_short"],
            target=target,
            prefer_long=prefer_long,
        )

    selected_pred = torch.where(prefer_long[:, None, None], scores["pred_long"], scores["pred_short"])
    metrics = batch_metrics(selected_pred, target)
    metrics["objective"] = float(loss.item())
    aux = scores
    return loss, metrics, aux


def reptile_meta_update(
    backbone: torch.nn.Module,
    fast_models: list[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    initial_state: dict[str, torch.Tensor],
    lambda_reg: float,
) -> None:
    """First-order meta update by nudging backbone toward domain-adapted fast models."""
    if not fast_models:
        return

    optimizer.zero_grad(set_to_none=True)
    state_dicts = [model.state_dict() for model in fast_models]

    for name, param in backbone.named_parameters():
        avg_fast = torch.stack([state[name].to(param.device) for state in state_dicts], dim=0).mean(dim=0)
        grad = (param.detach() - avg_fast).clone()
        if lambda_reg > 0.0:
            grad = grad + lambda_reg * (param.detach() - initial_state[name].to(param.device))
        param.grad = grad

    optimizer.step()


def train_meta_epoch(
    system: torch.nn.Module,
    train_loaders: dict[str, Any],
    optimizer_backbone: torch.optim.Optimizer,
    optimizer_pref: torch.optim.Optimizer,
    segmentation_cache: dict[str, dict[str, int | float | None]],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Train one epoch.

    This function supports:
    - full LSTPO with first-order meta-learning
    - sequential non-meta ablation
    - no-TPO ablation
    - baseline supervised training
    """
    system.train()
    initial_state = {name: param.detach().cpu().clone() for name, param in system.backbone.named_parameters()}
    iterators = {domain: iter(loader) for domain, loader in train_loaders.items()}

    running = defaultdict(float)
    metric_records: list[dict[str, float]] = []
    meta_batches = int(cfg["train"]["meta_batches_per_epoch"])
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    lr_inner = float(cfg["train"]["lr_inner"])
    inner_steps = int(cfg["train"]["inner_steps"])
    lambda_reg = float(cfg["train"]["lambda_reg"])

    for _ in range(meta_batches):
        fast_models: list[torch.nn.Module] = []

        for domain, loader in train_loaders.items():
            batch, iterators[domain] = next_batch(iterators[domain], loader)
            history, target = move_batch_to_device(batch, device)
            window_spec = segmentation_cache[domain]

            # 1) preference generator update
            pref_info = system.generate_preferences(history, target, window_spec)
            generator_loss = pref_info["generator_loss"]

            optimizer_pref.zero_grad(set_to_none=True)
            generator_loss.backward()
            clip_grad_norm_(system.preference_generator.parameters(), grad_clip)
            optimizer_pref.step()

            running["generator_loss"] += float(generator_loss.item())
            running["prefer_long_rate"] += float(pref_info["prefer_long"].float().mean().item())

            # 2) backbone update
            if cfg["ablation"]["use_mlb"]:
                split = split_support_query(history, target, pref_info["prefer_long"])

                fast_model = deepcopy(system.backbone).to(device)
                fast_model.train()
                inner_optimizer = torch.optim.SGD(fast_model.parameters(), lr=lr_inner)

                inner_loss_value = 0.0
                for _inner in range(inner_steps):
                    inner_loss, _, _ = compute_backbone_objective(
                        system=system,
                        backbone=fast_model,
                        history=split["support_history"],
                        target=split["support_target"],
                        prefer_long=split["support_prefers_long"],
                        window_spec=window_spec,
                        cfg=cfg,
                    )
                    inner_optimizer.zero_grad(set_to_none=True)
                    inner_loss.backward()
                    clip_grad_norm_(fast_model.parameters(), grad_clip)
                    inner_optimizer.step()
                    inner_loss_value = float(inner_loss.item())

                query_loss, query_metrics, _ = compute_backbone_objective(
                    system=system,
                    backbone=fast_model,
                    history=split["query_history"],
                    target=split["query_target"],
                    prefer_long=split["query_prefers_long"],
                    window_spec=window_spec,
                    cfg=cfg,
                )
                fast_models.append(fast_model)

                running["inner_loss"] += inner_loss_value
                running["query_loss"] += float(query_loss.item())
                metric_records.append(query_metrics)
            else:
                loss, metrics, _ = compute_backbone_objective(
                    system=system,
                    backbone=system.backbone,
                    history=history,
                    target=target,
                    prefer_long=pref_info["prefer_long"],
                    window_spec=window_spec,
                    cfg=cfg,
                )
                optimizer_backbone.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(system.backbone.parameters(), grad_clip)
                optimizer_backbone.step()

                running["query_loss"] += float(loss.item())
                metric_records.append(metrics)

        if cfg["ablation"]["use_mlb"]:
            reptile_meta_update(
                backbone=system.backbone,
                fast_models=fast_models,
                optimizer=optimizer_backbone,
                initial_state=initial_state,
                lambda_reg=lambda_reg,
            )

    averaged = aggregate_metric_list(metric_records)
    averaged["generator_loss"] = running["generator_loss"] / max(1, meta_batches * len(train_loaders))
    averaged["query_loss"] = running["query_loss"] / max(1, meta_batches * len(train_loaders))
    averaged["prefer_long_rate"] = running["prefer_long_rate"] / max(1, meta_batches * len(train_loaders))
    if cfg["ablation"]["use_mlb"]:
        averaged["inner_loss"] = running["inner_loss"] / max(1, meta_batches * len(train_loaders))
    return averaged


@torch.no_grad()
def evaluate_backbone(
    backbone: torch.nn.Module,
    loaders: dict[str, Any],
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Evaluate a backbone on full-history forecasting."""
    backbone.eval()
    per_domain: dict[str, dict[str, float]] = {}
    all_metric_records: list[dict[str, float]] = []

    for domain, loader in loaders.items():
        domain_metrics: list[dict[str, float]] = []
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            history, target = move_batch_to_device(batch, device)
            pred = backbone(history)
            metrics = batch_metrics(pred, target)
            domain_metrics.append(metrics)
            all_metric_records.append(metrics)
        per_domain[domain] = aggregate_metric_list(domain_metrics)

    return per_domain, aggregate_metric_list(all_metric_records)