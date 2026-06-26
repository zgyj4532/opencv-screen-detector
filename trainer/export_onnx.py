"""Export trained 3-class model to ONNX format.

Exports single model with 3-class output (natural/screenshot/screen_photo).
Model has triple inputs (RGB + FFT + DWT) with dynamic batch axes.
"""

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from . import config
from .model import load_model


def export_to_onnx(
    checkpoint_path: str,
    onnx_path: str,
    opset_version: int = 11,
    verify: bool = True,
):
    """Export PyTorch 3-class model to ONNX format with triple inputs.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        onnx_path: Path to save ONNX model
        opset_version: ONNX opset version
        verify: Whether to verify exported model
    """
    device = "cpu"

    # Load model (with DWT support)
    model = load_model(checkpoint_path, device=device, use_dwt=True)
    model.eval()

    # Create dummy inputs (RGB + FFT + DWT)
    # DWT is resized to IMAGE_SIZE to match FFT/RGB dimensions
    dummy_rgb = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    dummy_fft = torch.randn(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    dummy_dwt = torch.randn(1, 4, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)

    # Export to ONNX with triple inputs and dynamic axes
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_fft, dummy_dwt),
        onnx_path,
        opset_version=opset_version,
        input_names=["rgb_input", "fft_input", "dwt_input"],
        output_names=["output"],
        dynamic_axes={
            "rgb_input": {0: "batch_size"},
            "fft_input": {0: "batch_size"},
            "dwt_input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    if verify:
        verify_onnx_model(onnx_path, model, dummy_rgb, dummy_fft, dummy_dwt)

    return onnx_path


def verify_onnx_model(
    onnx_path, pytorch_model, dummy_rgb, dummy_fft, dummy_dwt
) -> None:
    """Verify ONNX model produces same output as PyTorch model."""
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(onnx_path)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_rgb, dummy_fft, dummy_dwt).numpy()

    # ONNX inference
    ort_inputs = {
        "rgb_input": dummy_rgb.numpy(),
        "fft_input": dummy_fft.numpy(),
        "dwt_input": dummy_dwt.numpy(),
    }
    onnx_output = ort_session.run(None, ort_inputs)[0]

    np.testing.assert_allclose(  # pyright: ignore[reportCallIssue]
        pytorch_output,
        onnx_output,  # pyright: ignore[reportArgumentType]
        rtol=1e-03,
        atol=1e-05,
    )

    print(f"ONNX model verified: {onnx_path}")


def export_to_torchscript(
    checkpoint_path: str,
    torchscript_path: str,
):
    """Export PyTorch model to TorchScript format."""
    device = "cpu"

    model = load_model(checkpoint_path, device=device, use_dwt=True)
    model.eval()

    dummy_rgb = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    dummy_fft = torch.randn(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    dummy_dwt = torch.randn(
        1, 4, config.IMAGE_SIZE // 2, config.IMAGE_SIZE // 2, device=device
    )

    scripted_model: Any = torch.jit.trace(model, (dummy_rgb, dummy_fft, dummy_dwt))
    scripted_model.save(torchscript_path)

    return torchscript_path


def main() -> None:
    """Main entry point for exporting 3-class model."""
    models_dir = config.PROJECT_ROOT / "inference" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Export 3-class model: natural/screenshot/screen_photo
    checkpoint_path = str(config.CHECKPOINT_DIR / "three_class_best.pth")
    onnx_path = str(models_dir / "three_class.onnx")

    if Path(checkpoint_path).exists():
        print("Exporting 3-class model...")
        export_to_onnx(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            opset_version=11,
            verify=True,
        )
        print(f"3-class model exported to: {onnx_path}")

        # Also export TorchScript
        torchscript_path = str(models_dir / "three_class.torchscript")
        print("Exporting TorchScript model...")
        export_to_torchscript(checkpoint_path, torchscript_path)
        print(f"TorchScript model exported to: {torchscript_path}")
    else:
        print(f"3-class checkpoint not found: {checkpoint_path}")
        print("Please train the 3-class model first: uv run python -m src.train")
