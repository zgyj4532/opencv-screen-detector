"""Export a harness checkpoint to ONNX and verify PyTorch<->ONNX parity.

Usage: uv run python experiment/cnn_fft_dwt_ablation/finalize_export.py <checkpoint.pth> <out.onnx>
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from trainer.model import load_model  # noqa: E402

IMAGE_SIZE = 224


def export(ckpt: str, onnx_path: str) -> None:
    model = load_model(ckpt, device="cpu", use_dwt=True)
    model.eval()
    rgb = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    fft = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    dwt = torch.randn(1, 4, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model, (rgb, fft, dwt), onnx_path, opset_version=11,
        input_names=["rgb_input", "fft_input", "dwt_input"], output_names=["output"],
        dynamic_axes={"rgb_input": {0: "b"}, "fft_input": {0: "b"}, "dwt_input": {0: "b"}, "output": {0: "b"}},
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    sess = ort.InferenceSession(onnx_path)
    with torch.no_grad():
        pt = model(rgb, fft, dwt)
        pt = (pt[0] if isinstance(pt, tuple) else pt).numpy()
    ox = sess.run(None, {"rgb_input": rgb.numpy(), "fft_input": fft.numpy(), "dwt_input": dwt.numpy()})[0]
    np.testing.assert_allclose(pt, ox, rtol=1e-3, atol=1e-5)
    print(f"OK: ONNX parity verified -> {onnx_path}")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
