from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import Standardizer, load_series_csv


@dataclass
class DomainDataBundle:
    """Container for one domain's standardized splits and metadata."""

    domain: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    standardizer: Standardizer
    columns: list[str]


class SlidingWindowDataset(Dataset):
    """Sliding-window dataset for multivariate time series.

    Each sample:
    - history: [L, C]
    - target: [H, C]
    """

    def __init__(self, array: np.ndarray, history_len: int, pred_len: int, domain: str) -> None:
        self.array = array.astype(np.float32)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.domain = domain
        self.total_window = self.history_len + self.pred_len
        if len(self.array) < self.total_window:
            raise ValueError(
                f"Series for domain={domain} is too short: len={len(self.array)}, "
                f"required >= {self.total_window}"
            )

    def __len__(self) -> int:
        return len(self.array) - self.total_window + 1

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start = idx
        mid = idx + self.history_len
        end = mid + self.pred_len

        history = self.array[start:mid]  # [L, C]
        target = self.array[mid:end]  # [H, C]

        return {
            "history": torch.tensor(history, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "domain": self.domain,
            "index": torch.tensor(idx, dtype=torch.long),
        }


def _discover_domains(root: Path, configured_domains: list[str] | None) -> list[str]:
    if configured_domains:
        return configured_domains
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def build_domain_data_bundles(cfg: dict[str, Any]) -> dict[str, DomainDataBundle]:
    """Load all domains from ``data.root`` and standardize with train stats."""
    root = Path(cfg["data"]["root"])
    domains = _discover_domains(root, cfg["data"].get("domains"))
    bundles: dict[str, DomainDataBundle] = {}

    for domain in domains:
        domain_dir = root / domain
        train_array, columns = load_series_csv(domain_dir / "train.csv")
        val_array, _ = load_series_csv(domain_dir / "val.csv")
        test_array, _ = load_series_csv(domain_dir / "test.csv")

        standardizer = Standardizer().fit(train_array)
        if cfg["data"].get("standardize", True):
            train_array = standardizer.transform(train_array)
            val_array = standardizer.transform(val_array)
            test_array = standardizer.transform(test_array)

        bundles[domain] = DomainDataBundle(
            domain=domain,
            train=train_array,
            val=val_array,
            test=test_array,
            standardizer=standardizer,
            columns=columns,
        )
    return bundles


def build_domain_loaders(
    bundles: dict[str, DomainDataBundle],
    cfg: dict[str, Any],
) -> tuple[dict[str, DataLoader], dict[str, DataLoader], dict[str, DataLoader]]:
    """Create per-domain DataLoaders for train / val / test."""
    history_len = int(cfg["data"]["history_len"])
    pred_len = int(cfg["data"]["pred_len"])
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 0))

    train_loaders: dict[str, DataLoader] = {}
    val_loaders: dict[str, DataLoader] = {}
    test_loaders: dict[str, DataLoader] = {}

    for domain, bundle in bundles.items():
        train_ds = SlidingWindowDataset(bundle.train, history_len, pred_len, domain)
        val_ds = SlidingWindowDataset(bundle.val, history_len, pred_len, domain)
        test_ds = SlidingWindowDataset(bundle.test, history_len, pred_len, domain)

        train_loaders[domain] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        val_loaders[domain] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        test_loaders[domain] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    return train_loaders, val_loaders, test_loaders