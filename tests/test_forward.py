from __future__ import annotations

import torch

from src.models.lstpo import LSTPOSystem


def _cfg() -> dict:
    return {
        "seed": 42,
        "output_dir": "outputs/test_forward",
        "data": {
            "root": "unused",
            "domains": ["dummy"],
            "history_len": 96,
            "pred_len": 24,
            "batch_size": 4,
            "num_workers": 0,
            "standardize": True,
            "split_ratios": [0.7, 0.1, 0.2],
            "t_sample_by_domain": {"dummy": 96},
            "default_short_len_by_domain": {"dummy": 48},
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
            "use_tss": True,}
    }
