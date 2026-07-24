from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

import pandas as pd

logger = logging.getLogger(__name__)


def walk_forward_validation(
    df: pd.DataFrame,
    horizon: int,
    initial_train_size: int,
    step_size: int = 1,
    model_fn: Callable[..., Any] | None = None,
) -> Iterator[dict[str, Any]]:
    # TODO: Implement expanding-window origins without temporal leakage.
    logger.debug(
        "walk_forward_validation called (stub) horizon=%s step_size=%s",
        horizon,
        step_size,
    )
    raise NotImplementedError("walk_forward_validation is not implemented yet.")
