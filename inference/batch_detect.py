"""Batch detection for screen detector V3."""

import json
from pathlib import Path

from .config import settings
from .predictor import ScreenDetectorPredictor


def detect_batch(
    input_dir: str,
    output_path: str | None = None,
) -> dict:
    """Run batch detection on all images in directory.

    Args:
        input_dir: Directory containing images
        output_path: Path to save JSON results

    Returns:
        dict with detection results
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Create predictor
    predictor = ScreenDetectorPredictor()

    # Collect all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in input_path.rglob("*") if f.suffix.lower() in image_extensions
    ]

    # Run detection
    results = []
    for image_path in image_files:
        try:
            result = predictor.predict(image_path)
            result["filename"] = image_path.name
            result["filepath"] = str(image_path)
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "filename": image_path.name,
                    "filepath": str(image_path),
                    "error": str(e),
                }
            )

    # Summary
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    # Count by class
    class_counts: dict[str, int] = {}
    for r in successful:
        class_name = r["class"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    summary = {
        "total": len(image_files),
        "successful": len(successful),
        "failed": len(failed),
        "class_distribution": class_counts,
    }

    output = {
        "summary": summary,
        "results": results,
    }

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        output_file.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return output


def main() -> None:
    """Main entry point for batch detection."""
    import sys

    # Default paths
    input_dir = str(settings.data_dir / "input")
    output_path = str(settings.output_dir / "result_v2.json")

    # Parse arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    # Create output directory
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    # Run batch detection
    detect_batch(
        input_dir=input_dir,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
