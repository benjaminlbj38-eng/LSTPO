from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sample-wise MSE over [H, C], output shape [B]."""
    return ((pred - target) ** 2).mean(dim=(1, 2))


def mae_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sample-wise MAE over [H, C], output shape [B]."""
    return (pred - target).abs().mean(dim=(1, 2))


def huber_per_sample(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Sample-wise Huber loss over [H, C], output shape [B]."""
    loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    return loss.mean(dim=(1, 2))


def log_prob_from_prediction(target: torch.Tensor, pred: torch.Tensor, temperature: float) -> torch.Tensor:
    """Convert prediction error into a log-probability proxy.

    Under a Gaussian noise assumption and up to an additive constant:
    log p(y | x) ∝ -MSE(y, y_hat) / tau
    """
    temperature = max(float(temperature), 1e-6)
    return -mse_per_sample(pred, target) / temperature


def choose_preferred_and_nonpreferred(
    long_score: torch.Tensor,
    short_score: torch.Tensor,
    prefer_long: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select preferred / non-preferred scores according to a boolean mask."""
    prefer_long = prefer_long.bool()
    preferred = torch.where(prefer_long, long_score, short_score)
    nonpreferred = torch.where(prefer_long, short_score, long_score)
    return preferred, nonpreferred


def tpo_loss_from_logps(
    logp_long: torch.Tensor,
    logp_short: torch.Tensor,
    prefer_long: torch.Tensor,
) -> torch.Tensor:
    """Temporal Preference Optimization loss.

    Conservative engineering completion:
    use DPO-consistent `-logsigmoid(pref - nonpref)` rather than `-sigmoid(...)`.
    """
    logp_pref, logp_nonpref = choose_preferred_and_nonpreferred(logp_long, logp_short, prefer_long)
    return (-F.logsigmoid(logp_pref - logp_nonpref)).mean()


def adaptation_dpo_loss_from_logps(
    model_logp_long: torch.Tensor,
    model_logp_short: torch.Tensor,
    ref_logp_long: torch.Tensor,
    ref_logp_short: torch.Tensor,
    prefer_long: torch.Tensor,
) -> torch.Tensor:
    """Reference-guided adaptation loss."""
    model_pref, model_nonpref = choose_preferred_and_nonpreferred(model_logp_long, model_logp_short, prefer_long)
    ref_pref, ref_nonpref = choose_preferred_and_nonpreferred(ref_logp_long, ref_logp_short, prefer_long)
    logits = (model_pref - ref_pref) - (model_nonpref - ref_nonpref)
    return (-F.logsigmoid(logits)).mean()


def selected_supervised_mse(
    pred_long: torch.Tensor,
    pred_short: torch.Tensor,
    target: torch.Tensor,
    prefer_long: torch.Tensor,
) -> torch.Tensor:
    """Supervised MSE on the selected preferred branch."""
    selected = torch.where(prefer_long[:, None, None], pred_long, pred_short)
    return F.mse_loss(selected, target)


def compute_flip_rate(reference_mask: torch.Tensor, noisy_mask: torch.Tensor) -> float:
    """Compute percentage of preference-label flips."""
    reference_mask = reference_mask.detach().cpu().to(torch.bool)
    noisy_mask = noisy_mask.detach().cpu().to(torch.bool)
    return float((reference_mask != noisy_mask).float().mean().item())