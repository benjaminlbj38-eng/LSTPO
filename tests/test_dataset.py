from __future__ import annotations

from pathlib import Path

from src.data.datasets import build_domain_data_bundles, build_domain_loaders
from src.data.synthetic import generate_synthetic_repository


def _cfg(root: str) -> dict:
    return {
        "seed": 42,
        "output_dir": "outputs/test_dataset",
        "data": {
            "root": root,
            "domains": ["weather_like", "traffic_like"],
            "history_len": 48,
            "pred_len": 12,
            "batch_size": 8,
            "num_workers": 0,
            "standardize": True,
            "split_ratios": [0.7, 0.1, 0.2],
            "t_sample_by_domain": {"weather_like": 96, "traffic_like": 96},
            "default_short_len_by_domain": {"weather_like": 48, "traffic_like": 24},
        },
        "model": {
            "backbone": "transformer",
            "d_model": 64,
            "nhead": 2,
            "num_layers": 2,
            "ff_dim": 128,
            "dropout": 0.1,
            "train_on_full_history_only": False,
            "pref_d_model": 32,
            "pref_nhead": 2,
            "pref_layers": 2,
            "pref_ff_dim": 64,
            "pref_dropout": 0.1,
            "pref_temperature": 1.0,
            "fixed_short_len": None,
            "fixed_long_len": None,
        },
        "train": {
            "device": "cpu",
            "epochs": 1,
            "pref_lr": 1e-3,
            "lr_outer": 1e-4,
            "lr_inner": 1e-2,
            "weight_decay": 1e-5,
            "lambda_reg": 1e-3,
            "inner_steps": 1,
            "meta_batches_per_epoch": 2,
            "grad_clip": 1.0,
            "patience": 2,
            "amp": False,
            "adapt_steps": 2,
            "adapt_lr": 1e-4,
        },
        "ablation": {
            "use_tss": True,
            "use_tpo": True,
            "use_mlb": True,
            "use_adaptation": True,
        },
        "evaluation": {
            "noise_sigmas": [0.01],
            "history_sweep": [24, 48],
            "save_visualizations": False,
            "plot_domain": "weather_like",
            "max_eval_batches": 2,
        },
    }


def test_dataset_shapes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    generate_synthetic_repository(
        output_root=data_root,
        domain_specs={
            "weather_like": {"kind": "weather", "steps": 400, "channels": 4},
            "traffic_like": {"kind": "traffic", "steps": 400, "channels": 6},
        },
        split_ratios=(0.7, 0.1, 0.2),
        seed=42,
    )
    cfg = _cfg(str(data_root))
    bundles = build_domain_data_bundles(cfg)
    train_loaders, val_loaders, test_loaders = build_domain_loaders(bundles, cfg)

    batch = next(iter(train_loaders["weather_like"]))
    assert batch["history"].shape == (8, 48, 4)
    assert batch["target"].shape == (8, 12, 4)

    batch2 = next(iter(train_loaders["traffic_like"]))
    assert batch2["history"].shape == (8, 48, 6)
    assert batch2["target"].shape == (8, 12, 6)

    assert len(val_loaders["weather_like"]) > 0
    assert len(test_loaders["traffic_like"]) > 0