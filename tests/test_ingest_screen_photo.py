"""Tests for the shipped daily-package screen_photo ingest path."""

import zipfile
from pathlib import Path

from trainer.ingest_screen_photo import ingest_screen_photo_zips, is_screen_photo_image_member


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_is_screen_photo_image_member_requires_folder_and_image_suffix() -> None:
    assert is_screen_photo_image_member("screen_photo/keep.jpg")
    assert is_screen_photo_image_member("nested/screen_photo/keep.PNG")
    assert not is_screen_photo_image_member("normal_photo/skip.jpg")
    assert not is_screen_photo_image_member("screen_photo/")
    assert not is_screen_photo_image_member("screen_photo/notes.txt")
    assert not is_screen_photo_image_member("normal_photo/screen_photo.jpg")


def test_ingest_screen_photo_zips_copies_only_screen_photo(tmp_path: Path) -> None:
    zip_path = _write_zip(
        tmp_path / "daily-package-fixture.zip",
        {
            "screen_photo/keep.jpg": b"\xff\xd8keep-bytes-screen",
            "normal_photo/skip.jpg": b"\xff\xd8skip-bytes-normal",
            "screenshot/also-skip.png": b"\x89PNGskip",
        },
    )
    dest = tmp_path / "screen_photo"

    result = ingest_screen_photo_zips([zip_path], dest)

    names = sorted(path.name for path in dest.iterdir())
    assert names == ["keep.jpg"]
    assert (dest / "keep.jpg").read_bytes() == b"\xff\xd8keep-bytes-screen"
    assert result["extracted_count"] == 1
    assert result["skipped_other_count"] == 2
    assert not (dest / "skip.jpg").exists()
    assert not (dest / "also-skip.png").exists()
    assert (dest / "normal_photo").exists() is False


def test_ingest_screen_photo_zips_skips_duplicate_content(tmp_path: Path) -> None:
    dest = tmp_path / "screen_photo"
    dest.mkdir()
    existing = dest / "already.jpg"
    existing.write_bytes(b"\xff\xd8same-bytes")
    zip_path = _write_zip(
        tmp_path / "daily-package-dup.zip",
        {
            "screen_photo/new-name.jpg": b"\xff\xd8same-bytes",
            "screen_photo/fresh.jpg": b"\xff\xd8fresh-bytes",
        },
    )

    result = ingest_screen_photo_zips([zip_path], dest)

    assert result["extracted_count"] == 1
    assert len(result["skipped_duplicate"]) == 1
    assert (dest / "fresh.jpg").read_bytes() == b"\xff\xd8fresh-bytes"
    assert {path.name for path in dest.iterdir()} == {"already.jpg", "fresh.jpg"}
