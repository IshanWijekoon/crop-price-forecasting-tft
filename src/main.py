from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop price forecasting research pipeline "
            "(stub — models are not implemented yet)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the experiment configuration YAML file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger.info("Pipeline entrypoint started (stub).")
    logger.info("Config path: %s", args.config)
    logger.warning(
        "Forecasting pipeline is not implemented yet. "
        "Planned stages: preprocess -> features -> models -> evaluate.",
    )
    # TODO: Load config, run preprocessing, feature engineering, models, evaluation.
    return 0


if __name__ == "__main__":
    sys.exit(main())
