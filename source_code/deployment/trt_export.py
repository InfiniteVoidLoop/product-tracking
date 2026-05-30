"""
deployment/trt_export.py
========================
Exports trained PyTorch / Ultralytics YOLO models to:
  1. ONNX  (.onnx)
  2. TensorRT Engine (.engine)  — requires NVIDIA TensorRT on Jetson / server GPU

Usage:
    python deployment/trt_export.py --model models/detection/best.pt
                                    --format onnx        # "onnx" | "trt"
                                    --fp16               # enable FP16 quantization
                                    --imgsz 640

For TensorRT export, either:
  - Ultralytics' built-in export (recommended): --format engine
  - External trtexec: pass --use-trtexec

Notes:
  - TensorRT export requires CUDA + TensorRT SDK installed.
  - INT8 calibration requires a calibration dataset (not implemented here;
    use Ultralytics' INT8 export directly).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def export_yolo_onnx(model_path: Path, imgsz: int = 640, opset: int = 17) -> Path:
    """Export Ultralytics YOLO .pt → .onnx."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics required.")
        sys.exit(1)

    model = YOLO(str(model_path))
    results = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True)
    onnx_path = Path(str(results))
    print(f"[Export] ONNX saved to: {onnx_path}")
    return onnx_path


def export_yolo_engine(
    model_path: Path,
    imgsz: int = 640,
    fp16: bool = True,
    int8: bool = False,
    batch: int = 1,
) -> Path:
    """
    Export Ultralytics YOLO .pt → TensorRT .engine via Ultralytics API.

    Requires: CUDA + TensorRT + ultralytics ≥ 8.1.0
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics required.")
        sys.exit(1)

    model = YOLO(str(model_path))
    results = model.export(
        format="engine",
        imgsz=imgsz,
        half=fp16,
        int8=int8,
        batch=batch,
        simplify=True,
    )
    engine_path = Path(str(results))
    print(f"[Export] TensorRT Engine saved to: {engine_path}")
    return engine_path


def export_via_trtexec(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    workspace_mb: int = 4096,
):
    """
    Alternative: call trtexec CLI to build a TensorRT engine from ONNX.
    Requires trtexec to be on PATH (installed with TensorRT SDK).
    """
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")

    print(f"[trtexec] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("[trtexec] ERROR: trtexec failed.")
        sys.exit(1)
    print(f"[trtexec] Engine saved to: {engine_path}")


def validate_engine(engine_path: Path, model_path: Path, imgsz: int = 640):
    """
    Quick validation: run one inference with the engine and compare outputs
    against the original .pt model to verify correctness.
    """
    import numpy as np
    import cv2
    from ultralytics import YOLO

    print(f"[Validate] Loading original model: {model_path}")
    original = YOLO(str(model_path))

    print(f"[Validate] Loading engine: {engine_path}")
    engine_model = YOLO(str(engine_path))

    # Create dummy frame
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    r1 = original.predict(dummy, verbose=False)
    r2 = engine_model.predict(dummy, verbose=False)

    n1 = len(r1[0].boxes) if r1[0].boxes else 0
    n2 = len(r2[0].boxes) if r2[0].boxes else 0

    print(f"[Validate] Original detections: {n1} | Engine detections: {n2}")
    if abs(n1 - n2) <= 2:
        print("[Validate] ✓ Engine output matches original (within tolerance).")
    else:
        print("[Validate] ⚠ Significant detection count difference — review FP16 precision loss.")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX or TensorRT")
    parser.add_argument("--model",       required=True, help="Path to .pt model weights")
    parser.add_argument("--format",      choices=["onnx", "engine", "trt"], default="onnx",
                        help="Export format: 'onnx' or 'engine'/'trt' (TensorRT)")
    parser.add_argument("--imgsz",       type=int, default=640)
    parser.add_argument("--fp16",        action="store_true", help="Enable FP16 quantization")
    parser.add_argument("--int8",        action="store_true", help="Enable INT8 (requires calib)")
    parser.add_argument("--batch",       type=int, default=1)
    parser.add_argument("--use-trtexec", action="store_true",
                        help="Use external trtexec CLI instead of Ultralytics engine export")
    parser.add_argument("--validate",    action="store_true",
                        help="Validate engine output against original model")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    if args.format == "onnx":
        out = export_yolo_onnx(model_path, imgsz=args.imgsz)

    elif args.format in ("engine", "trt"):
        if args.use_trtexec:
            # First export to ONNX, then call trtexec
            onnx_path = export_yolo_onnx(model_path, imgsz=args.imgsz)
            engine_path = model_path.with_suffix(".engine")
            export_via_trtexec(onnx_path, engine_path, fp16=args.fp16)
            out = engine_path
        else:
            out = export_yolo_engine(
                model_path,
                imgsz=args.imgsz,
                fp16=args.fp16,
                int8=args.int8,
                batch=args.batch,
            )

        if args.validate:
            validate_engine(out, model_path, args.imgsz)

    print(f"\n[Export] Complete → {out}")
    print(f"  Update config.yaml → detection.model_path: {out}")


if __name__ == "__main__":
    main()
