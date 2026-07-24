from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    # TODO: Read YAML via PyYAML and validate required keys.
    logger.debug("load_config called (stub) path=%s", path)
    raise NotImplementedError("load_config is not implemented yet.")


def ensure_dir(path: str | Path) -> Path:
    # TODO: Create directories with Path.mkdir(parents=True, exist_ok=True).
    logger.debug("ensure_dir called (stub) path=%s", path)
    raise NotImplementedError("ensure_dir is not implemented yet.")
