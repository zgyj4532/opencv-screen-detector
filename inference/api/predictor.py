"""Predictor lifecycle management for the API.

Provides lazy singleton with explicit startup/shutdown and failure logging.
"""

import contextlib
from collections.abc import Generator

from ..log import logger
from ..predictor import ScreenDetectorPredictor

_predictor: ScreenDetectorPredictor | None = None
_load_attempted: bool = False
_load_error: str | None = None


@contextlib.contextmanager
def ensure_predictor() -> Generator[None]:
    global _predictor, _load_attempted, _load_error
    _load_attempted = True
    try:
        _predictor = ScreenDetectorPredictor()
        logger.info("Predictor loaded successfully")
    except Exception as exc:
        _load_error = str(exc)
        logger.exception("Failed to load predictor")

    try:
        yield
    finally:
        if _predictor is not None:
            _predictor = None
            logger.info("Predictor released")


def get_predictor() -> ScreenDetectorPredictor | None:
    """Get the predictor singleton.

    Returns None if the model failed to load. Call startup() first during
    FastAPI lifespan to eagerly load and log any failures.
    """
    return _predictor


def load_error() -> str | None:
    """Return the error message if loading failed, else None."""
    return _load_error
