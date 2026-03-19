from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, save_checkpoint


class EarlyStopping:
    """Simple early stopping on a scalar metric."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min") -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: float | None = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        """Update state and return True if training should stop."""
        if self.best is None:
            self.best = value
            return False

        improved = False
        if self.mode == "min":
            improved = value < (self.best - self.min_delta)
        else:
            improved = value > (self.best + self.min_delta)

        if improved:
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience


class CheckpointManager:
    """Save best / last checkpoint according to a monitored metric."""

    def __init__(self, output_dir: str | Path, mode: str = "min") -> None:
        self.ckpt_dir = ensure_dir(Path(output_dir) / "checkpoints")
        self.mode = mode
        self.best_metric: float | None = None
        self.best_path = self.ckpt_dir / "best.pt"
        self.last_path = self.ckpt_dir / "last.pt"

    def save_last(self, state: dict[str, Any]) -> None:
        """Save the latest checkpoint."""
        save_checkpoint(self.last_path, state)

    def save_best_if_needed(self, metric: float, state: dict[str, Any]) -> bool:
        """Save checkpoint if the monitored metric improves."""
        if self.best_metric is None:
            self.best_metric = metric
            save_checkpoint(self.best_path, state)
            return True

        improved = metric < self.best_metric if self.mode == "min" else metric > self.best_metric
        if improved:
            self.best_metric = metric
            save_checkpoint(self.best_path, state)
            return True
        return False