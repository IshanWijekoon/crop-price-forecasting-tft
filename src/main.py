from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from src.evaluation.evaluate import run_evaluation
from src.evaluation.walk_forward_validation import run_walk_forward
from src.features.build_dataset import build_dataset
from src.features.prepare_model_data import run_prepare
from src.models.train import run_sarimax_stage, run_tft_stage
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
setup_logger("src")

STAGES: dict[str, Callable[[Path], object]] = {
    "build": build_dataset,
    "prepare": run_prepare,
    "sarimax": run_sarimax_stage,
    "tft": run_tft_stage,
    "walkforward": run_walk_forward,
    "evaluate": run_evaluation,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop price forecasting research pipeline. "
            "build: observed dataset; prepare: train-only Min-Max scaling and forecast windows; "
            "sarimax / tft: fit the model and write 60-day quantile forecasts to artifacts/forecasts/; "
            "walkforward: expanding-window refits of both models over the test split; "
            "evaluate: metrics, Diebold-Mariano tests, and plots in artifacts/evaluation/."
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
