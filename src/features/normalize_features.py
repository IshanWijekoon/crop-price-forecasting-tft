from __future__ import annotations

import logging
from typing import Any, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    method: str = "standard",
    fitted_scaler: Any | None = None,
) -> tuple[pd.DataFrame, Any]:
    # TODO: Fit scalers on training data only; transform validation/test separately.
    logger.debug(
        "normalize_features called (stub) method=%s columns=%s",
        method,
        columns,
    )
    raise NotImplementedError("normalize_features is not implemented yet.")
