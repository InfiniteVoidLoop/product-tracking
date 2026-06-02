"""
scripts/run_inference.py
========================
Runs YOLO inference on a video file and saves an annotated output video.

Usage:
    python scripts/run_inference.py \
        --model runs/detect/product_v1/weights/best.pt \
        --source demo/demo-conveyor.mp4 \
        --output demo/demo-conveyor-detected.mp4 \
        --conf 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def run_inference(
    model_path: str,
    source: str,
    output: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "",
):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        sys.exit(1)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        sys.exit(1)

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (W, H))

    print(f"[Inference] Source : {source}  ({W}×{H} @ {fps:.1f} FPS, {total} frames)")
    print(f"[Inference] Model  : {model_path}")
    print(f"[Inference] Output : {output}")
    print(f"[Inference] Conf   : {conf}  |  IOU: {iou}")
    print()

    frame_idx = 0
    total_detections = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device if device else None,
            verbose=False,
        )

        # Annotate frame
        annotated = results[0].plot(
            line_width=2,
            font_size=0.6,
        )

        # Overlay frame counter
        n_det = len(results[0].boxes) if results[0].boxes is not None else 0
        total_detections += n_det
        cv2.putText(
            annotated,
            f"Frame {frame_idx}/{total} | Detections: {n_det}",
            (8, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        writer.write(annotated)

        if frame_idx % 30 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  [{pct:5.1f}%] Frame {frame_idx}/{total} — dets this frame: {n_det}")

    cap.release()
    writer.release()
    print(f"\n[Inference] Done! Total detections: {total_detections}")
    print(f"[Inference] Annotated video saved → {output}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a video")
    parser.add_argument("--model",  required=True, help="Path to .pt weights")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", default=None,  help="Output annotated video path")
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--iou",    type=float, default=0.45)
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--device", default="",   help="cuda device or cpu")
    args = parser.parse_args()

    output = args.output
    if output is None:
        src = Path(args.source)
        output = str(src.parent / f"{src.stem}-detected{src.suffix}")

    run_inference(
        model_path=args.model,
        source=args.source,
        output=output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
