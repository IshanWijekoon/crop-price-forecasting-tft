from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from src.features.build_dataset import build_dataset
from src.features.prepare_model_data import run_prepare
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
setup_logger("src")

STAGES: dict[str, Callable[[Path], object]] = {
    "build": build_dataset,
    "prepare": run_prepare,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop price forecasting research pipeline. "
            "build: observed dataset; prepare: train-only Min-Max scaling and forecast windows."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the experiment configuration YAML file.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES),
        default=list(STAGES),
        help="Pipeline stages to run. They always run in pipeline order.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger.info("Config path: %s", args.config)

    for name, stage in STAGES.items():
        if name in args.stages:
            logger.info("Stage: %s", name)
            stage(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
