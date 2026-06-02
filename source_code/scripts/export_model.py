"""
scripts/export_model.py
=======================
Exports the YOLO model to ONNX and TensorRT formats for optimized inference.

Usage:
    python scripts/export_model.py --model models/detection/conveyor_best.pt
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX/TensorRT")
    parser.add_argument("--model", required=True, help="Path to the YOLO .pt model")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"[Export] Loading model from {model_path}")
    model = YOLO(str(model_path))

    print("[Export] Exporting to ONNX...")
    model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
    
    print("[Export] Exporting to TensorRT...")
    # Requires TensorRT to be installed, might fail if not available
    try:
        model.export(format="engine", imgsz=args.imgsz, workspace=4, half=True)
    except Exception as e:
        print(f"[Export] TensorRT export failed (this is normal if TensorRT is not installed): {e}")

    print("[Export] Finished!")

if __name__ == "__main__":
    main()
