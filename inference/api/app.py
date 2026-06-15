import contextlib
from collections.abc import AsyncGenerator

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..image_index import image_index
from ..scheduler import run_cleanup_loop
from .predictor import ensure_predictor
from .router import router as api_router


@contextlib.asynccontextmanager
async def lifespan(_: object) -> AsyncGenerator[None]:
    await image_index.migrate_from_index_file()
    with ensure_predictor():
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_cleanup_loop)
            try:
                yield
            finally:
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

app.include_router(api_router)
