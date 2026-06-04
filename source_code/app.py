"""
app.py
======
Main multi-threaded application entry point for the Conveyor Belt CV System.

Thread architecture:
    1. Capture Thread   — reads frames from camera/file → frame_queue
    2. Processing Thread — YOLO + ByteTrack + ZoneCounter → result_queue, crop_queue
    3. Classifier Thread — DefectClassifier on crops → db_queue
    4. DB Thread        — batched SQLite inserts from db_queue
    Main Thread         — visualization loop (cv2.imshow or headless)

Usage:
    python app.py [--config config.yaml] [--source <path>] [--headless] [--no-record]
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Set

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Resolve project root (allow running from any directory)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.counting import ProductState, ZoneCounter, ZoneCounterConfig
from src.database import AlertRecord, Database, ProductRecord
from src.defect_classifier import DefectClassifier
from src.detection import Detector
from src.tracking import ByteTracker
from src.utils.video_utils import ThreadedCapture, VideoSource, VideoWriter
from src.utils.visualization import draw_hud, draw_tracks, draw_zones


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_project_path(path: str | Path | None) -> Path | None:
    """Resolve local config paths from source_code/, independent of shell cwd."""
    if path in (None, ""):
        return None

    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


class FPSCounter:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30):
        self._ts = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._ts.append(now)
        if len(self._ts) > self._window:
            self._ts.pop(0)
        if len(self._ts) < 2:
            return 0.0
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class ConveyorApp:
    """
    Full pipeline application with four background threads and a main
    visualisation / control loop.

    Args:
        config: Parsed configuration dictionary (from config.yaml).
    """

    def __init__(self, config: dict):
        self.cfg = config
        self._stop_event = threading.Event()

        # Queues
        qcfg = config.get("app", {})
        self._frame_q: queue.Queue = queue.Queue(maxsize=qcfg.get("frame_queue_size", 8))
        self._result_q: queue.Queue = queue.Queue(maxsize=32)
        self._crop_q: queue.Queue = queue.Queue(maxsize=qcfg.get("crop_queue_size", 32))
        self._db_q: queue.Queue = queue.Queue(maxsize=qcfg.get("db_queue_size", 128))

        # Shared state (read by main thread for display)
        self._display_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_tracks = []
        self._latest_state_map: Dict[int, ProductState] = {}
        self._defect_ids: Set[int] = set()
        self._fps = FPSCounter()
        self._frame_idx = 0

        # Counts
        self._total_count = 0
        self._normal_count = 0
        self._defective_count = 0

        # Component init
        self._detector = Detector()
        self._tracker = ByteTracker(**config.get("tracking", {}))
        zone_cfg_raw = config.get("counting", {})
        self._zone_counter = ZoneCounter(ZoneCounterConfig(**zone_cfg_raw))
        self._classifier = DefectClassifier(
            weights_path=resolve_project_path(config.get("classifier", {}).get("model_path")),
            input_size=config.get("classifier", {}).get("input_size", 224),
            defect_threshold=config.get("classifier", {}).get("defect_threshold", 0.55),
            simulate=config.get("classifier", {}).get("simulate", True),
        )
        self._db = Database(resolve_project_path(config.get("database", {}).get("path", "data/conveyor.db")))

    # ------------------------------------------------------------------
    # Thread: Capture
    # ------------------------------------------------------------------

    def _capture_thread(self):
        """Read frames from VideoSource → frame_queue."""
        src_cfg = self.cfg.get("source", {})
        src_type = src_cfg.get("type", "file")
        W = src_cfg.get("width", 1280)
        H = src_cfg.get("height", 720)

        if src_type == "webcam":
            source = src_cfg.get("webcam_index", 0)
        elif src_type == "rtsp":
            source = src_cfg.get("rtsp_url", "")
        else:
            source = resolve_project_path(src_cfg.get("path", "demo/demo_conveyor.mp4"))

        fps_cap = src_cfg.get("fps_cap", 60)
        vs = VideoSource(source, width=W, height=H, fps_cap=fps_cap)

        try:
            vs.open()
        except RuntimeError as e:
            print(f"[CaptureThread] ERROR: {e}")
            self._stop_event.set()
            return

        print("[CaptureThread] Started.")
        while not self._stop_event.is_set():
            ok, frame = vs.read()
            if not ok:
                print("[CaptureThread] Stream ended.")
                self._stop_event.set()
                break
            try:
                self._frame_q.put((frame,), timeout=1.0)
            except queue.Full:
                pass  # Drop frame if processing is too slow

        vs.release()
        print("[CaptureThread] Stopped.")

    # ------------------------------------------------------------------
    # Thread: Processing (Detection + Tracking + Counting)
    # ------------------------------------------------------------------

    def _processing_thread(self):
        """Detect + track + count → result_queue + crop_queue."""
        det_cfg = self.cfg.get("detection", {})
        model_path = resolve_project_path(det_cfg.get("model_path", "yolov8s.pt"))
        self._detector.load(
            model_path=model_path or "yolov8s.pt",
            device=det_cfg.get("device", ""),
            conf=det_cfg.get("conf_threshold", 0.35),
            iou=det_cfg.get("iou_threshold", 0.45),
            imgsz=det_cfg.get("imgsz", 640),
        )

        W = self.cfg.get("source", {}).get("width", 1280)
        H = self.cfg.get("source", {}).get("height", 720)

        print("[ProcessingThread] Started.")
        while not self._stop_event.is_set():
            try:
                (frame,) = self._frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            frame_idx = self._frame_idx
            self._frame_idx += 1

            # Detection
            detections = self._detector.infer(frame)

            # Tracking
            tracks = self._tracker.update(detections)

            # Counting + crop triggering
            delta, crop_requests = self._zone_counter.update(tracks, frame, (W, H))
            state_map = self._zone_counter.get_state_map()

            # Update shared display state
            with self._display_lock:
                self._latest_frame = frame.copy()
                self._latest_tracks = tracks
                self._latest_state_map = state_map
                self._total_count = self._zone_counter.total_count
                self._normal_count = self._zone_counter.normal_count
                self._defective_count = self._zone_counter.defective_count

            # Push result for main-thread display
            try:
                self._result_q.put_nowait(
                    (frame, tracks, state_map, frame_idx)
                )
            except queue.Full:
                pass

            # Push crops for classification
            for tid, crop in crop_requests:
                try:
                    self._crop_q.put_nowait((tid, crop, frame_idx))
                except queue.Full:
                    pass

            # Record newly-counted products to DB queue
            if delta > 0:
                for tid, rec in self._zone_counter._records.items():
                    if rec.state == ProductState.COUNTED and rec.counted_at is not None:
                        try:
                            self._db_q.put_nowait(
                                ("product", ProductRecord(
                                    track_id=tid,
                                    status="Pending",
                                    frame_idx=frame_idx,
                                    timestamp=rec.counted_at or time.time(),
                                ))
                            )
                        except queue.Full:
                            pass

        print("[ProcessingThread] Stopped.")

    # ------------------------------------------------------------------
    # Thread: Defect Classifier
    # ------------------------------------------------------------------

    def _classifier_thread(self):
        """Classify crops → db_queue + update defect_ids."""
        print("[ClassifierThread] Started.")
        while not self._stop_event.is_set():
            try:
                tid, crop, frame_idx = self._crop_q.get(timeout=0.5)
            except queue.Empty:
                continue

            result = self._classifier.classify(crop)
            self._zone_counter.register_defect_result(tid, result.is_defective)

            if result.is_defective:
                with self._display_lock:
                    self._defect_ids.add(tid)

            # Queue DB update
            try:
                self._db_q.put_nowait(
                    ("classify", tid, result.label, result.confidence)
                )
                if result.is_defective:
                    self._db_q.put_nowait(
                        ("alert", tid, "defect_detected",
                         f"Track {tid} classified as Defective ({result.confidence:.2%})")
                    )
            except queue.Full:
                pass

        print("[ClassifierThread] Stopped.")

    # ------------------------------------------------------------------
    # Thread: Database Writer
    # ------------------------------------------------------------------

    def _db_thread(self):
        """Drain db_queue and write to SQLite."""
        print("[DBThread] Started.")
        # Keep a track_id → product_id mapping for classification updates
        tid_to_db_id: Dict[int, int] = {}

        while not self._stop_event.is_set() or not self._db_q.empty():
            try:
                item = self._db_q.get(timeout=0.5)
            except queue.Empty:
                continue

            kind = item[0]

            if kind == "product":
                _, record = item
                # Deduplicate: only insert if not already in DB
                if record.track_id not in tid_to_db_id:
                    db_id = self._db.insert_product(record)
                    tid_to_db_id[record.track_id] = db_id

            elif kind == "classify":
                _, tid, status, confidence = item
                db_id = tid_to_db_id.get(tid)
                if db_id is not None:
                    self._db.update_product_status(db_id, status, confidence)

            elif kind == "alert":
                _, tid, alert_type, message = item
                db_id = tid_to_db_id.get(tid)
                self._db.insert_alert(AlertRecord(
                    product_id=db_id,
                    alert_type=alert_type,
                    message=message,
                    timestamp=time.time(),
                ))

        print("[DBThread] Stopped.")

    # ------------------------------------------------------------------
    # Main visualisation loop
    # ------------------------------------------------------------------

    def run(self, headless: bool = False, record: bool = False):
        """
        Start all background threads and enter the display loop.

        Args:
            headless: If True, skip cv2.imshow (server/edge mode).
            record:   If True, record annotated output video.
        """
        # Start threads
        threads = [
            threading.Thread(target=self._capture_thread,    daemon=True, name="Capture"),
            threading.Thread(target=self._processing_thread, daemon=True, name="Processing"),
            threading.Thread(target=self._classifier_thread, daemon=True, name="Classifier"),
            threading.Thread(target=self._db_thread,         daemon=True, name="Database"),
        ]
        for t in threads:
            t.start()

        print("[App] All threads started. Press 'q' to quit.")

        # Video writer (optional)
        writer: Optional[VideoWriter] = None
        if record:
            app_cfg = self.cfg.get("app", {})
            out_path = resolve_project_path(app_cfg.get("output_path", "data/output_annotated.mp4"))
            W = self.cfg.get("source", {}).get("width", 1280)
            H = self.cfg.get("source", {}).get("height", 720)
            writer = VideoWriter(out_path, fps=30.0, frame_size=(W, H)).open()

        # Zone boundary pixel coordinates
        W = self.cfg.get("source", {}).get("width", 1280)
        H = self.cfg.get("source", {}).get("height", 720)
        axis = self.cfg.get("counting", {}).get("axis", "x")
        line1, line2 = self._zone_counter.get_zone_boundaries((W, H))

        try:
            while not self._stop_event.is_set():
                # Get the latest rendered result
                try:
                    frame, tracks, state_map, fidx = self._result_q.get(timeout=0.5)
                except queue.Empty:
                    # Check for stop signal
                    if not headless:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    continue

                self._fps.tick()
                fps_val = self._fps.tick()

                # --- Annotate ---
                with self._display_lock:
                    defect_ids = set(self._defect_ids)
                    total    = self._total_count
                    normal   = self._normal_count
                    defective = self._defective_count

                # Draw zones
                draw_zones(frame, line1, line2, axis=axis, alpha=0.15)
                # Draw tracks
                draw_tracks(frame, tracks, state_map, defect_ids, draw_trajectory=True)
                # Draw HUD
                draw_hud(
                    frame, fps_val, total, normal, defective,
                    frame_idx=fidx,
                    simulate_mode=self._classifier.is_simulation,
                )

                if writer:
                    writer.write(frame)

                if not headless:
                    cv2.imshow("Conveyor Belt Monitor", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        # Reset counters
                        self._zone_counter.reset()
                        self._tracker.reset()
                        self._db.clear_all()
                        print("[App] Counters reset.")

        except KeyboardInterrupt:
            print("\n[App] Keyboard interrupt.")
        finally:
            self._stop_event.set()
            for t in threads:
                t.join(timeout=5.0)
            if writer:
                writer.release()
            if not headless:
                cv2.destroyAllWindows()
            print(f"\n[App] Session complete. Total counted: {self._total_count}")
            print(f"      Normal: {self._normal_count} | Defective: {self._defective_count}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Conveyor Belt Product Detection & Tracking System"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--source", default=None,
        help="Override video source path (file, webcam index, or RTSP URL)"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without display window"
    )
    parser.add_argument(
        "--no-record", action="store_true", help="Disable output video recording"
    )
    parser.add_argument(
        "--conf", type=float, default=None, help="Override detection confidence threshold"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = resolve_project_path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Apply CLI overrides
    if args.source is not None:
        # Auto-detect type
        try:
            config["source"]["type"] = "webcam"
            config["source"]["webcam_index"] = int(args.source)
        except ValueError:
            if args.source.startswith("rtsp://") or args.source.startswith("http://"):
                config["source"]["type"] = "rtsp"
                config["source"]["rtsp_url"] = args.source
            else:
                config["source"]["type"] = "file"
                config["source"]["path"] = str(resolve_project_path(args.source))

    if args.conf is not None:
        config.setdefault("detection", {})["conf_threshold"] = args.conf

    app_record = not args.no_record and config.get("app", {}).get("record_output", True)

    app = ConveyorApp(config)
    app.run(headless=args.headless, record=app_record)


if __name__ == "__main__":
    main()
