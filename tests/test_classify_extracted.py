"""Test script to classify extracted images into correct categories.

This script uses the inference predictor to classify images from the extracted zip file
into the correct training categories (natural_photo, screenshot, screen_photo).
"""

import shutil
from pathlib import Path

import pytest

from inference.predictor import ScreenDetectorPredictor


@pytest.fixture(scope="module")
def predictor():
    """Create predictor instance."""
    return ScreenDetectorPredictor()


@pytest.fixture(scope="module")
def extracted_dir():
    """Path to extracted images."""
    return Path("data/temp_extract")


@pytest.fixture(scope="module")
def output_dir():
    """Path to output classified images."""
    return Path("data/input")


def test_classify_non_screen_photo(predictor, extracted_dir, output_dir):
    """Classify non_screen_photo images into natural_photo or screenshot."""
    non_screen_dir = extracted_dir / "non_screen_photo"
    if not non_screen_dir.exists():
        pytest.skip("non_screen_photo directory not found")

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in non_screen_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    assert len(image_files) > 0, "No images found in non_screen_photo"

    # Create output directories if they don't exist
    natural_dir = output_dir / "natural_photo"
    screenshot_dir = output_dir / "screenshot"
    natural_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    results = {"natural": 0, "screenshot": 0, "unknown": 0, "error": 0}

    for img_path in image_files:
        try:
            result = predictor.predict(img_path)
            predicted_class = result["class"]

            if predicted_class == "natural":
                # Copy to natural_photo
                dest = natural_dir / img_path.name
                shutil.copy2(img_path, dest)
                results["natural"] += 1
            elif predicted_class in ("screenshot", "screen_photo"):
                # For non_screen_photo, screenshot means it's a screenshot
                dest = screenshot_dir / img_path.name
                shutil.copy2(img_path, dest)
                results["screenshot"] += 1
            else:
                # unknown - put in screenshot as default
                dest = screenshot_dir / img_path.name
                shutil.copy2(img_path, dest)
                results["unknown"] += 1
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            results["error"] += 1

    print("\nClassification results for non_screen_photo:")
    print(f"  Natural: {results['natural']}")
    print(f"  Screenshot: {results['screenshot']}")
    print(f"  Unknown: {results['unknown']}")
    print(f"  Errors: {results['error']}")

    # Verify we processed some images
    assert results["natural"] + results["screenshot"] + results["unknown"] > 0


def test_copy_screen_photo(predictor, extracted_dir, output_dir):
    """Copy screen_photo images directly to output."""
    screen_dir = extracted_dir / "screen_photo"
    if not screen_dir.exists():
        pytest.skip("screen_photo directory not found")

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in screen_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    assert len(image_files) > 0, "No images found in screen_photo"

    # Create output directory
    output_screen_dir = output_dir / "screen_photo"
    output_screen_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in image_files:
        try:
            dest = output_screen_dir / img_path.name
            shutil.copy2(img_path, dest)
            copied += 1
        except Exception as e:
            print(f"Error copying {img_path.name}: {e}")

    print(f"\nCopied {copied} screen_photo images")
    assert copied > 0


def test_verify_classification(output_dir):
    """Verify that images were classified correctly."""
    # Count images in each directory
    categories = ["natural_photo", "screenshot", "screen_photo"]
    counts = {}

    for category in categories:
        category_dir = output_dir / category
        if category_dir.exists():
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            count = sum(
                1
                for f in category_dir.rglob("*")
                if f.suffix.lower() in image_extensions
            )
            counts[category] = count
        else:
            counts[category] = 0

    print("\nFinal image counts in data/input:")
    for category, count in counts.items():
        print(f"  {category}: {count}")

    total = sum(counts.values())
    print(f"  Total: {total}")

    # Verify we have images in all required categories
    assert counts["natural_photo"] > 0, "No natural_photo images"
    assert counts["screenshot"] > 0, "No screenshot images"
    assert counts["screen_photo"] > 0, "No screen_photo images"
