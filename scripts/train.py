from __future__ import annotations

import argparse
from pathlib import Path

from src.training.trainer import Trainer
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTPO or baselines.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/train.yaml"],
        help="One or more YAML config files merged from left to right.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    trainer = Trainer(cfg)
    try:
        best_path = trainer.fit()
        print(f"Best checkpoint saved to: {Path(best_path).resolve()}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()