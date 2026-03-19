from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Standardizer:
    """Per-channel z-score standardizer fitted on training split only."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, array: np.ndarray) -> "Standardizer":
        """Fit on an array of shape [T, C]."""
        if array.ndim != 2:
            raise ValueError(f"Expected 2D array [T, C], got shape {array.shape}")
        self.mean_ = array.mean(axis=0, keepdims=True)
        self.std_ = array.std(axis=0, keepdims=True)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        """Apply standardization."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fitted before transform().")
        return (array - self.mean_) / self.std_

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        """Invert standardization."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fitted before inverse_transform().")
        return array * self.std_ + self.mean_


def load_series_csv(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load a CSV file and keep only numeric columns.

    Returns
    -------
    array:
        Shape [T, C].
    columns:
        Numeric column names.
    """
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=["number"])
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {path}")
    array = numeric.to_numpy(dtype=np.float32)
    return array, list(numeric.columns)


def save_series_csv(path: str | Path, array: np.ndarray, columns: list[str] | None = None) -> None:
    """Save a [T, C] array to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = [f"v{i}" for i in range(array.shape[1])]
    pd.DataFrame(array, columns=columns).to_csv(path, index=False)