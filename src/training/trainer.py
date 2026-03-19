from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.data.datasets import build_domain_data_bundles, build_domain_loaders
from src.data.segmentation import estimate_all_domain_windows
from src.evaluation.evaluate import run_full_evaluation
from src.models.lstpo import LSTPOSystem
from src.training.callbacks import CheckpointManager, EarlyStopping
from src.training.loops import evaluate_backbone, train_meta_epoch
from src.utils.io import load_checkpoint, save_json
from src.utils.logging import ExperimentLogger
from src.utils.seed import set_seed


class Trainer:
    """High-level trainer for LSTPO and baselines."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(int(cfg["seed"]))
        self.device = self._select_device(cfg["train"].get("device", "auto"))

        self.system = LSTPOSystem(cfg).to(self.device)
        self.optimizer_backbone = torch.optim.Adam(
            self.system.backbone.parameters(),
            lr=float(cfg["train"]["lr_outer"]),
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
        self.optimizer_pref = torch.optim.Adam(
            self.system.preference_generator.parameters(),
            lr=float(cfg["train"]["pref_lr"]),
            weight_decay=float(cfg["train"]["weight_decay"]),
        )

        self.logger = ExperimentLogger(self.output_dir, enable_tensorboard=True)
        self.early_stopper = EarlyStopping(patience=int(cfg["train"]["patience"]), mode="min")
        self.checkpoints = CheckpointManager(self.output_dir, mode="min")

        self.bundles: dict[str, Any] | None = None
        self.train_loaders: dict[str, Any] | None = None
        self.val_loaders: dict[str, Any] | None = None
        self.test_loaders: dict[str, Any] | None = None
        self.segmentation_cache: dict[str, Any] | None = None

        save_json(self.output_dir / "merged_config.json", self.cfg)

    @staticmethod
    def _select_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def prepare_data(self) -> None:
        """Load datasets, create loaders, and estimate long/short windows."""
        self.bundles = build_domain_data_bundles(self.cfg)
        self.train_loaders, self.val_loaders, self.test_loaders = build_domain_loaders(self.bundles, self.cfg)
        self.segmentation_cache = estimate_all_domain_windows(self.bundles, self.cfg)
        save_json(self.output_dir / "segmentation_windows.json", self.segmentation_cache)

    def fit(self) -> Path:
        """Run training and return the best checkpoint path."""
        if self.bundles is None:
            self.prepare_data()

        assert self.train_loaders is not None
        assert self.val_loaders is not None
        assert self.segmentation_cache is not None

        epochs = int(self.cfg["train"]["epochs"])
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            train_metrics = train_meta_epoch(
                system=self.system,
                train_loaders=self.train_loaders,
                optimizer_backbone=self.optimizer_backbone,
                optimizer_pref=self.optimizer_pref,
                segmentation_cache=self.segmentation_cache,
                cfg=self.cfg,
                device=self.device,
            )

            _, val_metrics = evaluate_backbone(
                backbone=self.system.backbone,
                loaders=self.val_loaders,
                device=self.device,
                max_batches=self.cfg["evaluation"].get("max_eval_batches"),
            )

            self.logger.log_metrics("train", epoch, train_metrics)
            self.logger.log_metrics("val", epoch, val_metrics)

            checkpoint_state = {
                "epoch": epoch,
                "cfg": self.cfg,
                "model_state": self.system.state_dict(),
                "optimizer_backbone": self.optimizer_backbone.state_dict(),
                "optimizer_pref": self.optimizer_pref.state_dict(),
                "val_metrics": val_metrics,
                "segmentation_cache": self.segmentation_cache,
            }
            self.checkpoints.save_last(checkpoint_state)
            improved = self.checkpoints.save_best_if_needed(val_metrics["mse"], checkpoint_state)

            if improved:
                best_val = val_metrics["mse"]
                self.logger.log_message(f"New best checkpoint at epoch={epoch}, val_mse={best_val:.6f}")

            if self.early_stopper.step(val_metrics["mse"]):
                self.logger.log_message(f"Early stopping at epoch={epoch}")
                break

        self.load(self.checkpoints.best_path)
        self.logger.log_message(f"Best checkpoint loaded from {self.checkpoints.best_path}")
        return self.checkpoints.best_path

    def load(self, checkpoint_path: str | Path) -> None:
        """Load model and optimizers from checkpoint."""
        state = load_checkpoint(checkpoint_path, map_location=self.device)
        self.system.load_state_dict(state["model_state"])
        if "optimizer_backbone" in state:
            self.optimizer_backbone.load_state_dict(state["optimizer_backbone"])
        if "optimizer_pref" in state:
            self.optimizer_pref.load_state_dict(state["optimizer_pref"])
        if "segmentation_cache" in state:
            self.segmentation_cache = state["segmentation_cache"]

    def evaluate(self, adapt: bool = True) -> dict[str, Any]:
        """Run full evaluation including adaptation and visualizations."""
        if self.bundles is None:
            self.prepare_data()

        assert self.bundles is not None
        assert self.val_loaders is not None
        assert self.test_loaders is not None
        assert self.segmentation_cache is not None

        results = run_full_evaluation(
            system=self.system,
            bundles=self.bundles,
            val_loaders=self.val_loaders,
            test_loaders=self.test_loaders,
            segmentation_cache=self.segmentation_cache,
            cfg=self.cfg,
            device=self.device,
            output_dir=self.output_dir,
            adapt=adapt,
        )
        return results

    def close(self) -> None:
        """Close logger resources."""
        self.logger.close()