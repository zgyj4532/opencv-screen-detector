import hashlib
import tempfile
import uuid
import zipfile
from collections.abc import AsyncIterable, Generator
from pathlib import Path

import anyio
import fleep
import httpx
from fastapi import HTTPException, UploadFile, status

from ..config import settings
from ..image_index import ImageEntry, image_index
from .predictor import get_predictor

# Package export limits
MAX_FILES = 10000
MAX_EXPORT_SIZE = 20 * 1024**3  # 20GB
CHUNK_SIZE = 1024 * 1024  # 1MB
PACKAGE_TEMP_DIR = settings.data_dir / "temp_packages"


async def _stream_to_temp(
    stream: AsyncIterable[bytes],
    first_chunk: bytes,
    suffix: str,
) -> tuple[str, Path]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)

    h = hashlib.sha256(first_chunk)
    try:
        async with await anyio.Path(tmp_path).open("wb") as file:
            await file.write(first_chunk)
            async for chunk in stream:
                h.update(chunk)
                await file.write(chunk)
            await file.flush()
    except Exception:
        await anyio.Path(tmp_path).unlink(missing_ok=True)
        raise
    else:
        return h.hexdigest(), tmp_path


CONTENT_TYPE_SUFFIX_MAP: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


def _get_image_ext(header: bytes) -> str | None:
    if len(header) < FLEEP_HEADER_SIZE:
        return None
    info = fleep.get(header[:FLEEP_HEADER_SIZE])
    return CONTENT_TYPE_SUFFIX_MAP.get(info.mime[0]) if info.mime else None


REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}
FLEEP_HEADER_SIZE = 128
STREAM_CHUNK_SIZE = 64 * 1024  # 64 KB


async def stream_url_to_upload(url: str) -> ImageEntry:
    try:
        async with (
            httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers=REQUEST_HEADERS,
            ) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            chunk_iter = resp.aiter_bytes(STREAM_CHUNK_SIZE)
            first_chunk = b""
            async for chunk in chunk_iter:
                first_chunk += chunk
                if len(first_chunk) >= FLEEP_HEADER_SIZE:
                    break
            if not (suffix := _get_image_ext(first_chunk)):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Not an image",
                )
            file_hash, tmp_path = await _stream_to_temp(chunk_iter, first_chunk, suffix)
            return await image_index.add(file_hash, tmp_path)
    except httpx.HTTPStatusError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"HTTP {err.response.status_code}",
        ) from err
    except httpx.HTTPError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download: {err}",
        ) from err


async def _stream_file(file: UploadFile) -> AsyncIterable[bytes]:
    while chunk := await file.read(STREAM_CHUNK_SIZE):
        yield chunk


async def stream_file_to_upload(file: UploadFile) -> ImageEntry:
    first_chunk = await file.read(FLEEP_HEADER_SIZE)
    if not (suffix := _get_image_ext(first_chunk)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is not a valid image",
        )
    file_hash, tmp_path = await _stream_to_temp(_stream_file(file), first_chunk, suffix)
    return await image_index.add(file_hash, tmp_path)


def run_detect(file_path: Path) -> bool:
    """Run single-stage 3-class detection.

    Flow:
    1. CNN+FFT 3-class inference (with TTA)
    2. OOD detection: max_prob < 0.45 → unknown
    3. Threshold-based screen_photo classification (prob >= 0.35)
    4. Confidence tiering: accept/review/ignore
    """
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not available")

    result = predictor.predict(file_path)
    return result["class"] == "screen_photo"


def package_entries_to_temp_file(
    entries: list[ImageEntry],
    compress_level: int = 1,
) -> Path:
    """Package entries into a temporary ZIP file on disk.

    Uses ZIP_STORED for already-compressed images (jpg/png/webp)
    or low compression level to minimize CPU usage.

    Args:
        entries: List of ImageEntry objects to package.
        compress_level: ZIP compression level (0-9). 0=ZIP_STORED, 1=fastest.

    Returns:
        Path to the temporary ZIP file.

    Raises:
        HTTPException: If export exceeds size or file limits.
    """
    # Check limits
    if len(entries) > MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Export exceeds maximum file limit ({MAX_FILES} files)",
        )

    # Calculate total size
    total_size = sum(
        entry.path.stat().st_size for entry in entries if entry.path.exists()
    )
    if total_size > MAX_EXPORT_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Export size exceeds limit ({MAX_EXPORT_SIZE // (1024**3)}GB)",
        )

    # Choose compression
    if compress_level == 0:
        compression = zipfile.ZIP_STORED
        actual_level = 0
    else:
        compression = zipfile.ZIP_DEFLATED
        actual_level = compress_level

    # Write ZIP to disk
    tmp_path = PACKAGE_TEMP_DIR / f"{uuid.uuid4().hex}.zip"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            tmp_path,
            "w",
            compression=compression,
            compresslevel=actual_level,
        ) as zf:
            for entry in entries:
                if entry.path.exists():
                    zf.write(entry.path, entry.path.relative_to(settings.upload_dir))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return tmp_path


def iter_file(path: Path, chunk_size: int = CHUNK_SIZE) -> Generator[bytes, None, None]:
    """Yield file contents in chunks for streaming.

    Args:
        path: Path to the file to stream.
        chunk_size: Size of each chunk in bytes.

    Yields:
        Chunks of file data.
    """
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk


def cleanup_temp_file(path: Path) -> None:
    """Delete a temporary file if it exists.

    Args:
        path: Path to the file to delete.
    """
    path.unlink(missing_ok=True)
