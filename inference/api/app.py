import contextlib
import time
from collections.abc import AsyncGenerator, Awaitable, Callable

import anyio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from ..image_index import image_index
from ..log import logger
from ..scheduler import run_cleanup_loop
from .predictor import ensure_predictor
from .router import router as api_router


@contextlib.asynccontextmanager
async def lifespan(_: object) -> AsyncGenerator[None]:
    logger.info("API startup: migrating image index and loading predictor")
    await image_index.migrate_from_index_file()
    with ensure_predictor():
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_cleanup_loop)
            try:
                logger.info("API startup complete")
                yield
            finally:
                logger.info("API shutdown requested")
                tg.cancel_scope.cancel()


app = FastAPI(
    title="Screen Detector API",
    description="Screen detector with single-stage CNN + FFT Branch (3-class)",
    version="3.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_request(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    started = time.perf_counter()
    client = request.client.host if request.client else "-"
    query = f"?{request.url.query}" if request.url.query else ""
    logger.info(
        "HTTP start method={} path={}{} client={}",
        request.method,
        request.url.path,
        query,
        client,
    )
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "HTTP failed method={} path={} duration_ms={:.1f} client={}",
            request.method,
            request.url.path,
            elapsed_ms,
            client,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "HTTP done method={} path={} status={} duration_ms={:.1f} client={}",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        client,
    )
    return response


app.include_router(api_router)
