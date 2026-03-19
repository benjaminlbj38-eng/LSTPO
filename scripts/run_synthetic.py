from __future__ import annotations

import argparse
from pathlib import Path

from src.data.synthetic import generate_synthetic_repository
from src.training.trainer import Trainer
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic cross-domain data and run LSTPO.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="demo_data",
        help="Directory where synthetic CSVs will be generated.",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/train.yaml"],
        help="Config files merged from left to right.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["data"]["root"] = args.output_root

    domain_specs = {
        "weather_like": {"kind": "weather", "steps": 1200, "channels": 5},
        "traffic_like": {"kind": "traffic", "steps": 1200, "channels": 7},
        "exchange_like": {"kind": "exchange", "steps": 1200, "channels": 3},
    }
    generate_synthetic_repository(
        output_root=args.output_root,
        domain_specs=domain_specs,
        split_ratios=tuple(cfg["data"]["split_ratios"]),
        seed=int(cfg["seed"]),
    )

    trainer = Trainer(cfg)
    try:
        best_path = trainer.fit()
        results = trainer.evaluate(adapt=True)
        print(f"Synthetic demo finished.\nBest checkpoint: {Path(best_path).resolve()}")
        print(f"Aggregate metrics: {results['aggregate']}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()