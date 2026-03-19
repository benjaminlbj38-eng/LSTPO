from __future__ import annotations

import argparse
from pathlib import Path

from src.training.trainer import Trainer
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate preference visualizations.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/eval.yaml"],
        help="Config files merged from left to right.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Optional domain override for the visualization target.",
    )
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="Enable target-domain adaptation before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.domain is not None:
        cfg["evaluation"]["plot_domain"] = args.domain

    trainer = Trainer(cfg)
    try:
        trainer.prepare_data()
        trainer.load(args.checkpoint)
        results = trainer.evaluate(adapt=args.adapt)
        figure_dir = Path(cfg["output_dir"]) / "figures"
        print(f"Visualizations saved in: {figure_dir.resolve()}")
        print(f"Aggregate metrics: {results['aggregate']}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()