from __future__ import annotations

import argparse
from pathlib import Path

from src.training.trainer import Trainer
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSTPO / baselines.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/eval.yaml"],
        help="One or more YAML config files merged from left to right.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to outputs/.../checkpoints/best.pt from the loaded config.",
    )
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="Enable target-domain adaptation before testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    trainer = Trainer(cfg)
    try:
        trainer.prepare_data()
        checkpoint = args.checkpoint or str(Path(cfg["output_dir"]) / "checkpoints" / "best.pt")
        trainer.load(checkpoint)
        results = trainer.evaluate(adapt=args.adapt)
        print(f"Evaluation complete. Aggregate metrics: {results['aggregate']}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()