from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


class ExperimentLogger:
    """Lightweight experiment logger with console, file, JSONL, and TensorBoard support."""

    def __init__(self, output_dir: str | Path, enable_tensorboard: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.log_dir = ensure_dir(self.output_dir / "logs")
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.text_log_path = self.log_dir / "train.log"

        self.logger = logging.getLogger(str(self.output_dir))
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(self.text_log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        self.tb_writer = None
        if enable_tensorboard and SummaryWriter is not None:
            self.tb_writer = SummaryWriter(str(self.log_dir / "tensorboard"))

    def log_message(self, message: str) -> None:
        """Write a plain text message to console and log file."""
        self.logger.info(message)

    def log_metrics(self, split: str, step: int, metrics: dict[str, float]) -> None:
        """Log a metrics dictionary."""
        record: dict[str, Any] = {"split": split, "step": int(step), "metrics": metrics}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        pretty = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        self.logger.info("[%s][step=%d] %s", split, step, pretty)

        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"{split}/{key}", value, global_step=step)

    def close(self) -> None:
        """Close TensorBoard writer if available."""
        if self.tb_writer is not None:
            self.tb_writer.close()