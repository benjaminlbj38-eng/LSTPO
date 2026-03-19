from __future__ import annotations

from typing import Any

import numpy as np

from src.data.datasets import DomainDataBundle


def _to_1d_signal(array: np.ndarray) -> np.ndarray:
    """Reduce [T, C] to [T] by averaging channels for window estimation."""
    if array.ndim == 1:
        return array.astype(np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {array.shape}")
    return array.mean(axis=1).astype(np.float64)


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute normalized autocorrelation for lags [0, max_lag]."""
    x = x - np.mean(x)
    denom = np.sum(x * x) + 1e-8
    acf = np.zeros(max_lag + 1, dtype=np.float64)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            break
        acf[lag] = np.sum(x[:-lag] * x[lag:]) / denom
    return acf


def first_confidence_lag(acf_values: np.ndarray, length: int) -> int | None:
    """Find the first lag after which ACF stays inside the 95% confidence interval."""
    conf = 1.96 / np.sqrt(max(length, 1))
    for lag in range(1, len(acf_values)):
        if np.all(np.abs(acf_values[lag:]) <= conf):
            return lag
    return None


def welch_dominant_period(x: np.ndarray, segment_length: int | None = None) -> int | None:
    """Estimate dominant period with a lightweight Welch-style PSD average."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 16:
        return None

    seg = min(segment_length or 128, n)
    if seg < 8:
        return None

    step = max(seg // 2, 1)
    window = np.hanning(seg)
    spectra = []

    for start in range(0, n - seg + 1, step):
        chunk = x[start:start + seg]
        chunk = chunk - chunk.mean()
        spec = np.abs(np.fft.rfft(chunk * window)) ** 2
        spectra.append(spec)

    if not spectra:
        return None

    psd = np.mean(np.stack(spectra, axis=0), axis=0)
    freqs = np.fft.rfftfreq(seg, d=1.0)
    psd[0] = 0.0
    idx = int(np.argmax(psd))
    freq = float(freqs[idx])

    if freq <= 0.0:
        return None

    period = int(round(1.0 / freq))
    if period <= 1 or period > n:
        return None
    return period


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average trend estimate."""
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    trend = np.convolve(x_pad, kernel, mode="valid")
    return trend


def detect_change_points(trend: np.ndarray, min_gap: int = 8) -> list[int]:
    """Greedy change-point detector on absolute first differences."""
    if len(trend) < 4:
        return []

    diffs = np.abs(np.diff(trend))
    threshold = diffs.mean() + diffs.std()
    change_points: list[int] = []
    last = -min_gap

    for i, value in enumerate(diffs, start=1):
        if value > threshold and (i - last) >= min_gap:
            change_points.append(i)
            last = i
    return change_points


def average_change_interval(change_points: list[int], fallback: int | None = None) -> int | None:
    """Average spacing between change points."""
    if len(change_points) < 2:
        return fallback
    intervals = np.diff(np.asarray(change_points, dtype=np.int64))
    return int(round(float(intervals.mean())))


def estimate_windows_from_sequence(
    sequence: np.ndarray,
    history_len: int,
    t_sample: int | None = None,
    default_short_len: int | None = None,
    fixed_short: int | None = None,
    fixed_long: int | None = None,
) -> dict[str, int | float | None]:
    """Estimate long/short windows using ACF + Welch + trend change analysis.

    Notes
    -----
    This is a lightweight STL/PELT-inspired implementation.
    """
    if fixed_short is not None and fixed_long is not None:
        short_len = max(4, min(int(fixed_short), history_len - 1))
        long_len = max(short_len + 1, min(int(fixed_long), history_len))
        return {
            "short_len": short_len,
            "long_len": long_len,
            "k_acf": None,
            "period": None,
            "avg_change_interval": None,
        }

    x = _to_1d_signal(sequence)
    if t_sample is not None and len(x) > t_sample:
        start = max(0, (len(x) - t_sample) // 2)
        x = x[start:start + t_sample]

    max_lag = min(history_len - 1, max(8, len(x) // 2))
    acf_values = autocorrelation(x, max_lag=max_lag)
    k_acf = first_confidence_lag(acf_values, len(x))
    period = welch_dominant_period(x, segment_length=min(128, len(x)))

    if default_short_len is None:
        default_short_len = max(8, history_len // 2)

    candidates = []
    if k_acf is not None:
        candidates.append(int(k_acf))
    if period is not None:
        candidates.append(max(2, int(period // 2)))
    candidates.append(int(default_short_len))

    short_len = max(4, min(candidates))
    short_len = min(short_len, history_len - 1)

    trend_window = min(max(5, period or short_len), len(x) - 1 if len(x) > 5 else len(x))
    trend = moving_average(x, trend_window)
    cps = detect_change_points(trend, min_gap=max(4, short_len // 2))
    avg_interval = average_change_interval(cps, fallback=period)

    long_candidates = [short_len + 1]
    if period is not None:
        long_candidates.append(int(period))
    if avg_interval is not None:
        long_candidates.append(int(avg_interval))

    long_len = max(long_candidates)
    long_len = max(short_len + 1, long_len)
    long_len = min(history_len, long_len)

    short_len = min(short_len, long_len - 1)
    short_len = max(4, short_len)

    return {
        "short_len": int(short_len),
        "long_len": int(long_len),
        "k_acf": None if k_acf is None else int(k_acf),
        "period": None if period is None else int(period),
        "avg_change_interval": None if avg_interval is None else int(avg_interval),
    }


def estimate_all_domain_windows(
    bundles: dict[str, DomainDataBundle],
    cfg: dict[str, Any],
) -> dict[str, dict[str, int | float | None]]:
    """Estimate per-domain long/short windows from a representative mini-batch (M=1)."""
    history_len = int(cfg["data"]["history_len"])
    fixed_short = cfg["model"].get("fixed_short_len")
    fixed_long = cfg["model"].get("fixed_long_len")

    t_sample_map = cfg["data"].get("t_sample_by_domain", {})
    default_short_map = cfg["data"].get("default_short_len_by_domain", {})

    windows: dict[str, dict[str, int | float | None]] = {}
    for domain, bundle in bundles.items():
        windows[domain] = estimate_windows_from_sequence(
            sequence=bundle.train,
            history_len=history_len,
            t_sample=t_sample_map.get(domain),
            default_short_len=default_short_map.get(domain),
            fixed_short=fixed_short,
            fixed_long=fixed_long,
        )
    return windows