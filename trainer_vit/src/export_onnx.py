"""ONNX export for screen detector models.

Export DeiT/FFTDeiT/DWTFFTDeiT models to ONNX format for deployment.
"""

from pathlib import Path

import onnx
import torch
from loguru import logger
from onnx import checker

from .model import (
    DeiTScreenDetector,
    DWTFFTDeiT,
    FFTDeiT,
    load_deit_model,
)


def export_deit_to_onnx(
    model: DeiTScreenDetector,
    output_path: str,
    image_size: int = 224,
    opset_version: int = 17,
) -> None:
    """Export DeiT model to ONNX format.

    Args:
        model: DeiT model to export
        output_path: Path to save ONNX model
        image_size: Input image size
        opset_version: ONNX opset version
    """
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)

    logger.info(f"Exporting DeiT model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    checker.check_model(onnx_model)

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info("ONNX model exported successfully!")
    logger.info(f"File size: {file_size:.2f} MB")


def export_fft_deit_to_onnx(
    model: FFTDeiT,
    output_path: str,
    image_size: int = 224,
    opset_version: int = 17,
) -> None:
    """Export FFT+DeiT model to ONNX format.

    Args:
        model: FFT+DeiT model to export
        output_path: Path to save ONNX model
        image_size: Input image size
        opset_version: ONNX opset version
    """
    model.eval()

    dummy_rgb = torch.randn(1, 3, image_size, image_size)
    dummy_fft = torch.randn(1, 1, image_size, image_size)

    logger.info(f"Exporting FFT+DeiT model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_fft),
        output_path,
        opset_version=opset_version,
        input_names=["rgb_input", "fft_input"],
        output_names=["output"],
        dynamic_axes={
            "rgb_input": {0: "batch_size"},
            "fft_input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    checker.check_model(onnx_model)

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info("ONNX model exported successfully!")
    logger.info(f"File size: {file_size:.2f} MB")


def export_dwt_fft_deit_to_onnx(
    model: DWTFFTDeiT,
    output_path: str,
    image_size: int = 224,
    opset_version: int = 17,
) -> None:
    """Export DWT+FFT+DeiT model to ONNX format.

    Args:
        model: DWT+FFT+DeiT model to export
        output_path: Path to save ONNX model
        image_size: Input image size
        opset_version: ONNX opset version
    """
    model.eval()

    dummy_rgb = torch.randn(1, 3, image_size, image_size)
    dummy_fft = torch.randn(1, 1, image_size, image_size)

    logger.info(f"Exporting DWT+FFT+DeiT model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_fft),
        output_path,
        opset_version=opset_version,
        input_names=["rgb_input", "fft_input"],
        output_names=["output"],
        dynamic_axes={
            "rgb_input": {0: "batch_size"},
            "fft_input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    checker.check_model(onnx_model)

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info("ONNX model exported successfully!")
    logger.info(f"File size: {file_size:.2f} MB")


def export_from_checkpoint(
    checkpoint_path: str,
    output_path: str = "outputs/model.onnx",
    image_size: int = 224,
    opset_version: int = 17,
) -> None:
    """Export model from checkpoint to ONNX.

    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Path to save ONNX model
        image_size: Input image size
        opset_version: ONNX opset version
    """
    logger.info(f"Loading model from {checkpoint_path}...")
    model = load_deit_model(checkpoint_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, DWTFFTDeiT):
        export_dwt_fft_deit_to_onnx(model, output_path, image_size, opset_version)
    elif isinstance(model, FFTDeiT):
        export_fft_deit_to_onnx(model, output_path, image_size, opset_version)
    elif isinstance(model, DeiTScreenDetector):
        export_deit_to_onnx(model, output_path, image_size, opset_version)
    else:
        raise TypeError(f"Unknown model type: {type(model)}")


def main() -> None:
    """Main export entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Export screen detector model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/model.onnx",
        help="Path to save ONNX model",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version",
    )

    args = parser.parse_args()

    export_from_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        image_size=args.image_size,
        opset_version=args.opset_version,
    )


if __name__ == "__main__":
    main()
