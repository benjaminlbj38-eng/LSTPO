from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.datasets import DomainDataBundle, SlidingWindowDataset
from src.evaluation.metrics import aggregate_domain_metrics, batch_metrics
from src.evaluation.visualize import (
    plot_history_sweep,
    plot_long_short_predictions,
    plot_noise_flip_rates,
    plot_preference_dynamics,
)
from src.losses.preference_losses import compute_flip_rate
from src.models.modules.adaptation import adapt_model
from src.utils.io import ensure_dir, save_json


@torch.no_grad()
def evaluate_forecaster_on_loader(
    backbone: torch.nn.Module,
    loader: Any,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate a forecasting backbone on one DataLoader."""
    backbone.eval()
    metrics_list: list[dict[str, float]] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        pred = backbone(history)
        metrics_list.append(batch_metrics(pred, target))

    if not metrics_list:
        return {"mse": float("nan"), "mae": float("nan")}

    mse = sum(m["mse"] for m in metrics_list) / len(metrics_list)
    mae = sum(m["mae"] for m in metrics_list) / len(metrics_list)
    return {"mse": float(mse), "mae": float(mae)}


@torch.no_grad()
def evaluate_history_sweep(
    backbone: torch.nn.Module,
    bundle: DomainDataBundle,
    history_lengths: list[int],
    pred_len: int,
    device: torch.device,
) -> list[float]:
    """Evaluate MSE under different history lengths on the test split."""
    results = []
    for history_len in history_lengths:
        ds = SlidingWindowDataset(bundle.test, history_len=history_len, pred_len=pred_len, domain=bundle.domain)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        metrics = evaluate_forecaster_on_loader(backbone, loader, device)
        results.append(metrics["mse"])
    return results


@torch.no_grad()
def evaluate_noise_flip_rates(
    system: torch.nn.Module,
    bundle: DomainDataBundle,
    window_spec: dict[str, int | float | None],
    sigmas: list[float],
    history_len: int,
    pred_len: int,
    device: torch.device,
    max_batches: int = 16,
) -> list[float]:
    """Evaluate preference flip rate under Gaussian noise."""
    ds = SlidingWindowDataset(bundle.test, history_len=history_len, pred_len=pred_len, domain=bundle.domain)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    system.preference_generator.eval()
    flip_rates: list[float] = []

    for sigma in sigmas:
        per_batch_rates: list[float] = []
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            history = batch["history"].to(device)
            target = batch["target"].to(device)

            long_hist, short_hist = system.split_views(history, window_spec)
            clean_pref = system.preference_generator(long_hist, short_hist, target)

            noisy_history = history + sigma * torch.randn_like(history)
            noisy_long, noisy_short = system.split_views(noisy_history, window_spec)
            noisy_pref = system.preference_generator(noisy_long, noisy_short, target)

            per_batch_rates.append(compute_flip_rate(clean_pref["prefer_long"], noisy_pref["prefer_long"]))

        flip_rates.append(float(sum(per_batch_rates) / max(1, len(per_batch_rates))))
    return flip_rates


def _get_first_batch(loader: Any) -> dict[str, Any]:
    return next(iter(loader))


def run_full_evaluation(
    system: torch.nn.Module,
    bundles: dict[str, DomainDataBundle],
    val_loaders: dict[str, Any],
    test_loaders: dict[str, Any],
    segmentation_cache: dict[str, dict[str, int | float | None]],
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
    adapt: bool = True,
) -> dict[str, Any]:
    """Run evaluation for all domains, optionally including target-domain adaptation."""
    output_dir = Path(output_dir)
    figure_dir = ensure_dir(output_dir / "figures")

    results: dict[str, dict[str, Any]] = {}
    plot_domain = cfg["evaluation"]["plot_domain"]
    max_eval_batches = cfg["evaluation"].get("max_eval_batches")

    for domain, bundle in bundles.items():
        window_spec = segmentation_cache[domain]
        backbone_to_eval = system.backbone
        adapt_stats = None

        if adapt and cfg["ablation"]["use_adaptation"]:
            backbone_to_eval, adapt_stats = adapt_model(
                system=system,
                reference_backbone=system.backbone,
                adapt_loader=val_loaders[domain],
                window_spec=window_spec,
                cfg=cfg,
                device=device,
            )

        metrics = evaluate_forecaster_on_loader(
            backbone=backbone_to_eval,
            loader=test_loaders[domain],
            device=device,
            max_batches=max_eval_batches,
        )
        results[domain] = {
            "metrics": metrics,
            "adaptation": adapt_stats,
            "window_spec": window_spec,
        }

        if cfg["evaluation"]["save_visualizations"] and domain == plot_domain:
            batch = _get_first_batch(test_loaders[domain])
            history = batch["history"].to(device)
            target = batch["target"].to(device)

            pref_info = system.generate_preferences(history, target, window_spec)
            scores = system.score_views(history, target, window_spec, backbone=backbone_to_eval)

            target_np = target[0].detach().cpu().numpy()
            pred_long_np = scores["pred_long"][0].detach().cpu().numpy()
            pred_short_np = scores["pred_short"][0].detach().cpu().numpy()
            step_pref_np = pref_info["step_prefers_long"][0].detach().cpu().numpy().astype(int)

            plot_long_short_predictions(
                target=target_np,
                pred_long=pred_long_np,
                pred_short=pred_short_np,
                path=figure_dir / "long_short_predictions.png",
                title=f"{domain}: Long vs Short Forecasting",
            )
            plot_preference_dynamics(
                target=target_np,
                pred_long=pred_long_np,
                pred_short=pred_short_np,
                step_prefers_long=step_pref_np,
                path=figure_dir / "preference_dynamics.png",
                title=f"{domain}: Preference Dynamics",
            )

            sigmas = list(cfg["evaluation"]["noise_sigmas"])
            flip_rates = evaluate_noise_flip_rates(
                system=system,
                bundle=bundle,
                window_spec=window_spec,
                sigmas=sigmas,
                history_len=int(cfg["data"]["history_len"]),
                pred_len=int(cfg["data"]["pred_len"]),
                device=device,
            )
            plot_noise_flip_rates(
                sigmas=sigmas,
                flip_rates=flip_rates,
                path=figure_dir / "noise_flip_rate.png",
                title=f"{domain}: Preference Flip Rate under Noise",
            )

            history_sweep = list(cfg["evaluation"]["history_sweep"])
            history_mse = evaluate_history_sweep(
                backbone=backbone_to_eval,
                bundle=bundle,
                history_lengths=history_sweep,
                pred_len=int(cfg["data"]["pred_len"]),
                device=device,
            )
            plot_history_sweep(
                history_lengths=history_sweep,
                mse_values=history_mse,
                path=figure_dir / "history_sweep.png",
                title=f"{domain}: History Horizon Sweep",
            )

            results[domain]["noise_flip_rates"] = dict(zip(sigmas, flip_rates))
            results[domain]["history_sweep_mse"] = dict(zip(history_sweep, history_mse))

    aggregate = aggregate_domain_metrics(results)
    full_results = {
        "per_domain": results,
        "aggregate": aggregate,
    }
    save_json(output_dir / "metrics.json", full_results)
    return full_results