import contextlib
from datetime import UTC, datetime
from typing import Annotated

import anyio.to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from ..image_index import image_index
from ..log import logger
from .predictor import get_predictor, load_error
from .schema import (
    ClassifyRequest,
    ClassifyResponse,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    PackageRequest,
)
from .utils import (
    package_entries_to_stream,
    run_detect,
    stream_file_to_upload,
    stream_url_to_upload,
)

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    predictor = get_predictor()
    error = load_error()
    return HealthResponse(
        status="healthy" if predictor else ("degraded" if error else "starting"),
        model_loaded=predictor.model_available if predictor else False,
        load_error=error,
    )


@router.post("/detect", response_model=DetectResponse)
async def detect_url(request: DetectRequest) -> DetectResponse:
    """Detect screen photo from image URL."""
    logger.info(f"Received detection request for URL: {request.url}")

    try:
        entry = await stream_url_to_upload(request.url)
        is_screen = await anyio.to_thread.run_sync(run_detect, entry.path)
        await image_index.classify(entry.file_hash, is_screen)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Detection failed for URL: {request.url!r}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc!r}",
        ) from exc
    else:
        return DetectResponse(image_id=entry.file_hash, is_screen=is_screen)


@router.post("/detect/upload", response_model=DetectResponse)
async def detect_upload(
    file: Annotated[UploadFile, File()],
) -> DetectResponse:
    """Detect screen photo from uploaded file."""
    logger.info(f"Received detection request for uploaded file: {file.filename!r}")

    try:
        entry = await stream_file_to_upload(file)
        is_screen = await anyio.to_thread.run_sync(run_detect, entry.path)
        await image_index.classify(entry.file_hash, is_screen)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Detection failed for uploaded file: {file.filename!r}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc!r}",
        ) from exc
    else:
        return DetectResponse(image_id=entry.file_hash, is_screen=is_screen)


@router.post("/classify", response_model=ClassifyResponse)
async def classify_image(request: ClassifyRequest) -> ClassifyResponse:
    logger.info(f"Received classification request: {request!r}")

    try:
        entry = await image_index.classify(request.image_id, request.is_screen)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image not found: {request.image_id}",
        ) from exc

    return ClassifyResponse(
        image_id=request.image_id,
        is_screen=request.is_screen,
        class_name=entry.class_name or "unclassified",
    )


@router.post("/package")
async def package_images(request: PackageRequest) -> StreamingResponse:
    """Package images uploaded after the given timestamp into a zip file.

    Returns a zip file containing:
    - /screen_photo/*: screen capture images
    - /normal_photo/*: non-screen images
    """
    # Filter images after timestamp
    after_time = (
        after_time.astimezone(UTC)
        if (after_time := request.after_timestamp).tzinfo
        else after_time.replace(tzinfo=UTC)
    )

    logger.info(f"Received package request for images after {after_time.isoformat()}")

    async with contextlib.AsyncExitStack() as stack:
        matching_entries = await stack.enter_async_context(
            image_index.list_entries_after(after_time)
        )
        if not matching_entries:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No images found after the specified timestamp",
            )

        zip_filename = f"images_{datetime.now(UTC):%Y%m%d_%H%M%S}.zip"
        return StreamingResponse(
            package_entries_to_stream(matching_entries),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
            background=BackgroundTask(stack.pop_all().aclose),
        )
