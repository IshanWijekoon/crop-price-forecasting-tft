from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.features.build_dataset import build_dataset
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
setup_logger("src")

STAGES = ("build",)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop price forecasting research pipeline "
            "(dataset build implemented; models are not implemented yet)."
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
        choices=STAGES,
        default=list(STAGES),
        help="Pipeline stages to run, in order.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger.info("Config path: %s", args.config)

    if "build" in args.stages:
        build_dataset(args.config)

    # TODO: Add train and evaluate stages once SARIMAX and TFT are implemented.
    return 0


if __name__ == "__main__":
    sys.exit(main())
