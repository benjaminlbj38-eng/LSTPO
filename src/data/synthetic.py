### quick test!
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocessing import save_series_csv


def _weather_like(steps: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(steps, dtype=np.float32)
    series = []
    for c in range(channels):
        annual = 1.5 * np.sin(2 * np.pi * t / 96 + c * 0.2)
        seasonal = 0.7 * np.sin(2 * np.pi * t / 24 + c * 0.3)
        trend = 0.003 * t + 0.3 * np.sin(2 * np.pi * t / 240)
        noise = 0.15 * rng.standard_normal(steps)
        series.append(annual + seasonal + trend + noise)
    return np.stack(series, axis=1).astype(np.float32)


def _traffic_like(steps: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(steps, dtype=np.float32)
    series = []
    for c in range(channels):
        daily = 1.2 * np.sin(2 * np.pi * t / 24 + c * 0.1)
        weekly = 0.4 * np.sin(2 * np.pi * t / 168 + c * 0.2)
        local = np.zeros(steps, dtype=np.float32)
        eps = 0.25 * rng.standard_normal(steps)
        for i in range(1, steps):
            local[i] = 0.7 * local[i - 1] + eps[i]
        spikes = np.zeros(steps, dtype=np.float32)
        spike_positions = rng.choice(np.arange(12, steps - 12), size=max(4, steps // 80), replace=False)
        spikes[spike_positions] = rng.uniform(1.0, 2.5, size=len(spike_positions))
        series.append(daily + weekly + local + spikes)
    return np.stack(series, axis=1).astype(np.float32)


def _exchange_like(steps: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(steps, dtype=np.float32)
    series = []
    for c in range(channels):
        base = np.cumsum(0.05 * rng.standard_normal(steps)).astype(np.float32)
        slow = 0.5 * np.sin(2 * np.pi * t / 128 + c * 0.4)
        fast = 0.2 * np.sin(2 * np.pi * t / 12 + c * 0.8)
        drift = 0.001 * t * ((-1) ** c)
        series.append(base + slow + fast + drift)
    return np.stack(series, axis=1).astype(np.float32)


def _split_series(series: np.ndarray, split_ratios: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(series)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    n_test = n - n_train - n_val
    train = series[:n_train]
    val = series[n_train:n_train + n_val]
    test = series[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def generate_synthetic_repository(
    output_root: str | Path,
    domain_specs: dict[str, dict[str, Any]] | None = None,
    split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> None:
    """Generate a small cross-domain repository that follows the expected CSV format."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if domain_specs is None:
        domain_specs = {
            "weather_like": {"kind": "weather", "steps": 1200, "channels": 5},
            "traffic_like": {"kind": "traffic", "steps": 1200, "channels": 7},
            "exchange_like": {"kind": "exchange", "steps": 1200, "channels": 3},
        }

    for domain, spec in domain_specs.items():
        kind = spec["kind"]
        steps = int(spec["steps"])
        channels = int(spec["channels"])

        if kind == "weather":
            series = _weather_like(steps, channels, rng)
        elif kind == "traffic":
            series = _traffic_like(steps, channels, rng)
        elif kind == "exchange":
            series = _exchange_like(steps, channels, rng)
        else:
            raise ValueError(f"Unknown synthetic kind: {kind}")

        train, val, test = _split_series(series, split_ratios)
        columns = [f"v{i}" for i in range(channels)]
        save_series_csv(output_root / domain / "train.csv", train, columns=columns)
        save_series_csv(output_root / domain / "val.csv", val, columns=columns)
        save_series_csv(output_root / domain / "test.csv", test, columns=columns)