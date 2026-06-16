"""Image index manager for tracking image IDs and training status."""

import contextlib
import shutil
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import anyio
import anyio.to_thread
from pydantic import BaseModel, Field, TypeAdapter

from .config import settings

EXPIRATION_SECONDS = 60 * 10  # 10 mins
TABLE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS images (
    file_hash TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    class_name TEXT,
    created_at INTEGER NOT NULL
)
"""


class ImageEntry(BaseModel):
    file_name: str
    file_hash: str
    class_name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def classified(self) -> bool:
        return self.class_name is not None

    @property
    def path(self) -> Path:
        upload_dir = settings.upload_dir
        if self.class_name is None:  # unclassified
            return upload_dir / self.file_name
        return upload_dir / self.class_name / self.file_name

    async def classify(self, class_name: str) -> None:
        src = anyio.Path(self.path)
        if not await src.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")

        class_dir = anyio.Path(settings.upload_dir / class_name)
        await class_dir.mkdir(parents=True, exist_ok=True)
        await src.rename(class_dir / self.file_name)

        self.class_name = class_name


class ImageIndex:
    def __init__(self) -> None:
        self._lock = anyio.Lock()

    @contextlib.asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[aiosqlite.Connection]:
        async with self._lock, aiosqlite.connect(settings.index_db) as conn:
            await conn.execute(TABLE_SCHEMA)
            await conn.commit()
            yield conn

    async def _find_by_hash(
        self, conn: aiosqlite.Connection, file_hash: str
    ) -> ImageEntry | None:
        cursor = await conn.execute(
            "SELECT file_name, class_name, created_at FROM images WHERE file_hash = ?",
            (file_hash,),
        )
        row = await cursor.fetchone()
        if row:
            file_name, class_name, created_at = row
            return ImageEntry(
                file_name=file_name,
                file_hash=file_hash,
                class_name=class_name,
                created_at=datetime.fromtimestamp(created_at, tz=UTC),
            )
        return None

    async def add(self, file_hash: str, path: Path) -> ImageEntry:
        async with self._get_connection() as conn:
            if entry := await self._find_by_hash(conn, file_hash):
                await anyio.Path(entry.path).touch()
                return entry

            new_path = settings.upload_dir / f"{file_hash}{path.suffix}"
            await anyio.Path(new_path.parent).mkdir(parents=True, exist_ok=True)
            await anyio.to_thread.run_sync(shutil.move, path, new_path)

            entry = ImageEntry(file_name=new_path.name, file_hash=file_hash)
            await conn.execute(
                "INSERT INTO images (file_hash, file_name, class_name, created_at) VALUES (?, ?, ?, ?)",
                (
                    entry.file_hash,
                    entry.file_name,
                    entry.class_name,
                    entry.created_at.astimezone(UTC).timestamp(),
                ),
            )
            await conn.commit()
            return entry

    async def classify(self, file_hash: str, is_screen: bool) -> ImageEntry:
        class_name = "screen_photo" if is_screen else "normal_photo"
        async with self._get_connection() as conn:
            entry = await self._find_by_hash(conn, file_hash)
            if not entry:
                raise ValueError(f"Image not found: {file_hash}")

            await entry.classify(class_name)
            await conn.execute(
                "UPDATE images SET class_name = ? WHERE file_hash = ?",
                (entry.class_name, entry.file_hash),
            )
            await conn.commit()
            return entry

    async def clean_expired(self) -> None:
        now = datetime.now(UTC)
        deadline = (now - timedelta(seconds=EXPIRATION_SECONDS)).timestamp()
        async with self._get_connection() as conn:
            await conn.execute(
                "DELETE FROM images WHERE created_at < ? AND class_name IS NULL",
                (deadline,),
            )
            await conn.commit()

    @contextlib.asynccontextmanager
    async def list_entries_after(
        self, ts: datetime
    ) -> AsyncGenerator[list[ImageEntry]]:
        timestamp = ts.astimezone(UTC).timestamp()
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT file_hash, file_name, class_name, created_at FROM images WHERE created_at > ?",
                (timestamp,),
            )
            rows = await cursor.fetchall()
            entries = [
                ImageEntry(
                    file_hash=row[0],
                    file_name=row[1],
                    class_name=row[2],
                    created_at=datetime.fromtimestamp(row[3], tz=UTC),
                )
                for row in rows
            ]
            yield entries

    async def migrate_from_index_file(self) -> None:
        index_file = settings.upload_dir / "index.json"
        if not await anyio.Path(index_file).exists():
            return

        raw = await anyio.Path(index_file).read_bytes()
        entries = TypeAdapter(dict[str, ImageEntry]).validate_json(raw)
        async with self._get_connection() as conn:
            for entry in entries.values():
                await conn.execute(
                    "INSERT OR IGNORE INTO images (file_hash, file_name, class_name, created_at) VALUES (?, ?, ?, ?)",
                    (
                        entry.file_hash,
                        entry.file_name,
                        entry.class_name,
                        entry.created_at.astimezone(UTC).timestamp(),
                    ),
                )
            await conn.commit()
        await anyio.Path(index_file).rename(index_file.with_suffix(".json.bak"))


image_index = ImageIndex()
