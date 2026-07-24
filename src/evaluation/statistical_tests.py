from __future__ import annotations

import logging
from typing import Any

from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def diebold_mariano_test(
    loss_a: ArrayLike,
    loss_b: ArrayLike,
    h: int = 1,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    # TODO: Implement DM statistic with appropriate variance estimator.
    logger.debug(
        "diebold_mariano_test called (stub) h=%s alternative=%s",
        h,
        alternative,
    )
    raise NotImplementedError("diebold_mariano_test is not implemented yet.")
