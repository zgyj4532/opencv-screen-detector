"""Stream only ``screen_photo`` image members from daily-package zip archives.

Other zip folders (including ``normal_photo``) are never written. Members are
read in memory and skipped when the destination already holds the same SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCREEN_PHOTO_DIR_NAME = "screen_photo"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "input"


def is_screen_photo_image_member(member_name: str) -> bool:
    """True when a zip member is an image file under a ``screen_photo`` folder."""
    path = Path(member_name.replace("\\", "/"))
    if path.suffix.lower() not in IMAGE_EXTS:
        return False
    parents = {part.lower() for part in path.parts[:-1]}
    return SCREEN_PHOTO_DIR_NAME in parents


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _index_existing_hashes(dest_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not dest_dir.exists():
        return index
    for path in dest_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            index[_sha256_bytes(path.read_bytes())] = path
    return index


def ingest_screen_photo_zips(zip_paths: list[Path], dest_dir: Path) -> dict:
    """Copy only ``screen_photo`` image members into ``dest_dir``.

    Duplicate content (by SHA-256) is skipped. Non-target members are counted
    but never written.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = _index_existing_hashes(dest_dir)
    extracted: list[str] = []
    skipped_duplicate: list[str] = []
    skipped_other = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                name = info.filename
                if info.is_dir() or not is_screen_photo_image_member(name):
                    if not info.is_dir():
                        skipped_other += 1
                    continue
                with archive.open(info) as handle:
                    payload = handle.read()
                digest = _sha256_bytes(payload)
                if digest in existing:
                    skipped_duplicate.append(f"{zip_path.name}:{name}")
                    continue
                source_name = Path(name.replace("\\", "/")).name
                target = dest_dir / source_name
                if target.exists():
                    target = dest_dir / f"{digest}{Path(source_name).suffix.lower()}"
                target.write_bytes(payload)
                existing[digest] = target
                extracted.append(target.as_posix())
    return {
        "extracted": extracted,
        "extracted_count": len(extracted),
        "skipped_duplicate": skipped_duplicate,
        "skipped_other_count": skipped_other,
        "dest_dir": dest_dir.as_posix(),
    }


def default_daily_package_zips(data_dir: Path = DEFAULT_DATA_DIR) -> list[Path]:
    return sorted(data_dir.glob("daily-package-*.zip"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("zips", nargs="*", type=Path)
    args = parser.parse_args(argv)
    dest = args.dest or (args.data_dir / SCREEN_PHOTO_DIR_NAME)
    zips = [path.resolve() for path in args.zips] if args.zips else default_daily_package_zips(args.data_dir)
    missing = [str(path) for path in zips if not path.is_file()]
    if missing:
        raise SystemExit(f"Zip archive not found: {', '.join(missing)}")
    if not zips:
        raise SystemExit("No daily-package-*.zip archives found")
    result = ingest_screen_photo_zips(zips, dest)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
