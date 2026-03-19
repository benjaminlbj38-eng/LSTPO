from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import ensure_dir


def plot_long_short_predictions(
    target: np.ndarray,
    pred_long: np.ndarray,
    pred_short: np.ndarray,
    path: str | Path,
    title: str,
    channel_idx: int = 0,
) -> None:
    """Plot target vs long-view vs short-view predictions."""
    path = Path(path)
    ensure_dir(path.parent)

    plt.figure(figsize=(10, 4))
    plt.plot(target[:, channel_idx], label="target")
    plt.plot(pred_long[:, channel_idx], label="long-term prediction")
    plt.plot(pred_short[:, channel_idx], label="short-term prediction")
    plt.title(title)
    plt.xlabel("forecast step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_preference_dynamics(
    target: np.ndarray,
    pred_long: np.ndarray,
    pred_short: np.ndarray,
    step_prefers_long: np.ndarray,
    path: str | Path,
    title: str,
    channel_idx: int = 0,
) -> None:
    """Plot prediction dynamics and step-wise long/short preference mask."""
    path = Path(path)
    ensure_dir(path.parent)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(target[:, channel_idx], label="target")
    axes[0].plot(pred_long[:, channel_idx], label="long-term")
    axes[0].plot(pred_short[:, channel_idx], label="short-term")
    axes[0].set_title(title)
    axes[0].set_ylabel("forecast value")
    axes[0].legend()

    mask = step_prefers_long.astype(np.float32)
    axes[1].step(np.arange(len(mask)), mask, where="mid", label="1=prefer long, 0=prefer short")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlabel("forecast step")
    axes[1].set_ylabel("preference")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_noise_flip_rates(
    sigmas: list[float],
    flip_rates: list[float],
    path: str | Path,
    title: str,
) -> None:
    """Plot preference-label flip rate vs Gaussian noise level."""
    path = Path(path)
    ensure_dir(path.parent)

    plt.figure(figsize=(7, 4))
    plt.plot(sigmas, flip_rates, marker="o")
    plt.title(title)
    plt.xlabel("noise sigma")
    plt.ylabel("flip rate")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_history_sweep(
    history_lengths: list[int],
    mse_values: list[float],
    path: str | Path,
    title: str,
) -> None:
    """Plot MSE vs history horizon."""
    path = Path(path)
    ensure_dir(path.parent)

    plt.figure(figsize=(7, 4))
    plt.plot(history_lengths, mse_values, marker="o")
    plt.title(title)
    plt.xlabel("history length")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()