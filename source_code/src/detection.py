"""
src/detection.py
================
YOLO detection wrapper for the Conveyor Belt CV System.

Provides a unified ``Detector`` interface that supports:
  - Ultralytics YOLO (.pt weights)
  - ONNX runtime (.onnx)  — auto-selected when path ends in .onnx
  - TensorRT engine (.engine) — loaded via Ultralytics backend

Detection results are returned as a list of ``Detection`` dataclasses so the
rest of the pipeline never needs to import ultralytics directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Detection result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single object detection result."""

    bbox: List[float]          # [x1, y1, x2, y2] in pixel coordinates
    confidence: float          # Detection confidence 0–1
    class_id: int              # Integer class index
    class_name: str            # Human-readable class name

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """
    Unified YOLO detector wrapper.

    Usage::

        det = Detector()
        det.load("models/detection/yolov8s.pt", device="cuda:0", conf=0.35)
        detections = det.infer(frame)   # frame: H×W×3 BGR numpy array

    Attributes:
        model_path: Path to the loaded model weights.
        conf_threshold: Minimum confidence to report a detection.
        iou_threshold: NMS IoU threshold.
        imgsz: Inference image size (square).
        device: Torch device string.
        class_names: Mapping from class_id → class_name.
        latency_ms: Inference latency of the last call (milliseconds).
    """

    def __init__(self):
        self._model = None
        self.model_path: Optional[Path] = None
        self.conf_threshold: float = 0.35
        self.iou_threshold: float = 0.45
        self.imgsz: int = 640
        self.device: str = ""
        self.class_names: Dict[int, str] = {}
        self.latency_ms: float = 0.0
        self._backend: str = "ultralytics"   # "ultralytics" | "onnx"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: str | Path,
        device: str = "",
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: Optional[List[int]] = None,
    ) -> "Detector":
        """
        Load model weights.

        Args:
            model_path: Path to .pt / .onnx / .engine weights.
            device:     Torch device ("", "cpu", "cuda:0", …).
            conf:       Confidence threshold.
            iou:        NMS IoU threshold.
            imgsz:      Inference image size.
            classes:    Optional class ID filter list.

        Returns:
            self (fluent API)
        """
        model_path = Path(model_path)
        self.model_path = model_path
        self.conf_threshold = conf
        self.iou_threshold = iou
        self.imgsz = imgsz
        self.device = device
        self._classes_filter = classes

        if not model_path.exists():
            print(
                f"[Detector] Warning: model file not found at '{model_path}'. "
                "Will auto-download yolov8s.pt when first inference is called."
            )
            model_path = Path("yolov8s.pt")

        suffix = model_path.suffix.lower()

        if suffix == ".onnx":
            self._backend = "onnx"
            self._load_onnx(model_path)
        else:
            self._backend = "ultralytics"
            self._load_ultralytics(model_path)

        return self

    def _load_ultralytics(self, model_path: Path):
        """Load model via Ultralytics YOLO API."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package required: pip install ultralytics")

        self._model = YOLO(str(model_path))
        # Populate class names
        if hasattr(self._model, "names") and self._model.names:
            self.class_names = dict(self._model.names)
        print(f"[Detector] Loaded Ultralytics model: {model_path}")

    def _load_onnx(self, model_path: Path):
        """Load model via ONNX Runtime."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required: pip install onnxruntime-gpu")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._ort_session = ort.InferenceSession(str(model_path), providers=providers)
        self._ort_input_name = self._ort_session.get_inputs()[0].name
        print(f"[Detector] Loaded ONNX model: {model_path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.

        Args:
            frame: H×W×3 uint8 numpy array in BGR colour space.

        Returns:
            List of ``Detection`` objects (may be empty).
        """
        if self._backend == "ultralytics":
            return self._infer_ultralytics(frame)
        elif self._backend == "onnx":
            return self._infer_onnx(frame)
        else:
            return []

    def _infer_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        if self._model is None:
            self.load(self.model_path or "yolov8s.pt")

        t0 = time.perf_counter()

        kwargs: Dict[str, Any] = dict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        if self._classes_filter is not None:
            kwargs["classes"] = self._classes_filter

        results = self._model.predict(**kwargs)
        self.latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            # Update class name map if needed
            if result.names:
                self.class_names.update(result.names)

            for box, conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
                detections.append(
                    Detection(
                        bbox=box.tolist(),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=self.class_names.get(int(cls_id), str(cls_id)),
                    )
                )

        return detections

    def _infer_onnx(self, frame: np.ndarray) -> List[Detection]:
        """Basic ONNX Runtime inference path (YOLOv8 ONNX format)."""
        import cv2

        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # Preprocess
        blob = cv2.resize(frame, (self.imgsz, self.imgsz))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis]  # (1, 3, H, W)

        outputs = self._ort_session.run(None, {self._ort_input_name: blob})
        self.latency_ms = (time.perf_counter() - t0) * 1000.0

        # YOLOv8 ONNX output: (1, 84, 8400) → transpose → (8400, 84)
        preds = np.squeeze(outputs[0]).T  # (8400, nc+4)
        nc = preds.shape[1] - 4
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        cls_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(cls_ids)), cls_ids]

        mask = confs >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        detections: List[Detection] = []
        scale_x, scale_y = w / self.imgsz, h / self.imgsz
        for bx, conf, cls_id in zip(boxes_xywh, confs, cls_ids):
            cx, cy, bw, bh = bx
            x1 = (cx - bw / 2) * scale_x
            y1 = (cy - bh / 2) * scale_y
            x2 = (cx + bw / 2) * scale_x
            y2 = (cy + bh / 2) * scale_y
            detections.append(
                Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self.class_names.get(int(cls_id), str(cls_id)),
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def warm_up(self, frame_shape: tuple = (640, 640, 3)):
        """Run a dummy inference to warm up the model on GPU."""
        dummy = np.zeros(frame_shape, dtype=np.uint8)
        self.infer(dummy)
        print(f"[Detector] Warm-up done ({self.latency_ms:.1f} ms)")

    def __repr__(self) -> str:
        return (
            f"Detector(backend={self._backend!r}, "
            f"model={self.model_path}, "
            f"conf={self.conf_threshold}, "
            f"iou={self.iou_threshold})"
        )
