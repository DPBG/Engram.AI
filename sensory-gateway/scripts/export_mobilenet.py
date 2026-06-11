"""
Export MobileNetV3-Small feature extractor to ONNX format.

Removes the classification head, keeping only the feature backbone.
Output: 576-dimensional feature vector per input image.

Usage:
    python -m scripts.export_mobilenet
    python scripts/export_mobilenet.py --output models/mobilenetv3_small_features.onnx

Requires: torch, torchvision (install temporarily, not needed at runtime).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export(output_path: str | Path) -> None:
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print(
            "PyTorch + torchvision required for export only.\n"
            "  pip install torch torchvision\n"
            "After export, you can uninstall them — runtime uses onnxruntime."
        )
        sys.exit(1)

    print("Loading MobileNetV3-Small (pretrained)...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Remove classifier — keep features + adaptive pool
    # MobileNetV3-Small.features outputs (B, 576, 7, 7)
    # avgpool reduces to (B, 576, 1, 1)
    class FeatureExtractor(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            return x.flatten(1)  # (B, 576)

    feat_model = FeatureExtractor(model)
    feat_model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, 224, 224)

    # Verify output shape
    with torch.no_grad():
        out = feat_model(dummy)
    print(f"Feature vector shape: {out.shape}")  # should be (1, 576)

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        feat_model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done: {output_path} ({size_mb:.1f} MB)")
    print(f"Feature dimension: {out.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description="Export MobileNetV3-Small features to ONNX")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "models" / "mobilenetv3_small_features.onnx"),
        help="Output ONNX file path",
    )
    args = parser.parse_args()
    export(args.output)


if __name__ == "__main__":
    main()
