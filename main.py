"""Screen Detector API - Main entry point."""

import uvicorn

from inference.api import app
from inference.config import settings
from inference.log import DEFAULT_LOG_LEVEL, LOGGING_CONFIG, logger

if __name__ == "__main__":
    logger.info(
        "Starting uvicorn host={} port={} log_level={} access_log=True",
        settings.api_host,
        settings.api_port,
        DEFAULT_LOG_LEVEL,
    )
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_config=LOGGING_CONFIG,
        log_level=DEFAULT_LOG_LEVEL.lower(),
        access_log=True,
    )
