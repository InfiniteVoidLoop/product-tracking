"""
src/utils/video_utils.py
========================
Camera / video source utilities for the Conveyor Belt CV System.

Classes:
  - VideoSource        — wraps cv2.VideoCapture (file / webcam / RTSP)
  - ThreadedCapture    — background thread that continuously reads frames
  - VideoWriter        — thin cv2.VideoWriter wrapper with auto-codec selection
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# VideoSource
# ---------------------------------------------------------------------------

class VideoSource:
    """
    Unified video source that supports:
      - Video files (mp4, avi, …)
      - Webcam (by device index)
      - RTSP / HTTP streams

    Args:
        source:   File path, integer webcam index, or RTSP URL string.
        width:    Desired capture width (best-effort, hardware-dependent).
        height:   Desired capture height.
        fps_cap:  Maximum frames per second cap (0 = no cap).
    """

    def __init__(
        self,
        source,
        width: int = 1280,
        height: int = 720,
        fps_cap: int = 0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps_cap = fps_cap
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_interval = (1.0 / fps_cap) if fps_cap > 0 else 0.0
        self._last_read_ts = 0.0

    def open(self) -> "VideoSource":
        """Open the capture device."""
        if isinstance(self.source, int):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(str(self.source))

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")

        # Request resolution (ignored for files)
        if isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print(
            f"[VideoSource] Opened '{self.source}' — "
            f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
            f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
            f"{self._cap.get(cv2.CAP_PROP_FPS):.1f} FPS"
        )
        return self

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read one frame from the capture device.

        Returns:
            (ok, frame)  where ok=False means end-of-stream or error.
        """
        if self._cap is None:
            return False, None

        # FPS cap
        if self._frame_interval > 0:
            elapsed = time.perf_counter() - self._last_read_ts
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        ok, frame = self._cap.read()
        self._last_read_ts = time.perf_counter()

        if not ok:
            return False, None

        # Resize if frame dimensions differ from requested
        fh, fw = frame.shape[:2]
        if (fw, fh) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))

        return True, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def frame_count(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def native_fps(self) -> float:
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.release()

    def __repr__(self) -> str:
        return f"VideoSource(source={self.source!r}, {self.width}×{self.height})"


# ---------------------------------------------------------------------------
# ThreadedCapture
# ---------------------------------------------------------------------------

class ThreadedCapture:
    """
    Wraps ``VideoSource`` and reads frames in a background daemon thread.

    Frames are pushed into a bounded ``queue.Queue``.  The processing
    thread can call ``get()`` without blocking the camera reader.

    Args:
        source:       VideoSource instance (already opened).
        maxsize:      Max frames buffered in the queue.
        loop:         If True, loop the video when it ends (file sources).
    """

    def __init__(
        self,
        source: VideoSource,
        maxsize: int = 8,
        loop: bool = False,
    ):
        self.source = source
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._loop = loop
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._frame_count = 0

    def start(self) -> "ThreadedCapture":
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=3.0)

    def get(self, timeout: float = 2.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieve the next frame from the buffer.

        Returns:
            (ok, frame) — ok=False signals end of stream.
        """
        try:
            item = self._q.get(timeout=timeout)
            return item
        except queue.Empty:
            return False, None

    @property
    def qsize(self) -> int:
        return self._q.qsize()

    def _reader(self):
        while not self._stop_event.is_set():
            ok, frame = self.source.read()
            if not ok:
                if self._loop and isinstance(self.source.source, (str, Path)):
                    # Re-open the source to loop
                    self.source.release()
                    self.source.open()
                    continue
                else:
                    # Signal end of stream
                    self._q.put((False, None))
                    break
            self._frame_count += 1
            # Drop oldest frame if queue is full (non-blocking put)
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put((True, frame))

    def __repr__(self) -> str:
        return f"ThreadedCapture(qsize={self._q.qsize()}/{self._q.maxsize})"


# ---------------------------------------------------------------------------
# VideoWriter
# ---------------------------------------------------------------------------

class VideoWriter:
    """
    Thin wrapper around ``cv2.VideoWriter`` with automatic codec selection.

    Args:
        output_path: Output file path (mp4 / avi).
        fps:         Output frames per second.
        frame_size:  (width, height) of output frames.
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 30.0,
        frame_size: Tuple[int, int] = (1280, 720),
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self) -> "VideoWriter":
        suffix = self.output_path.suffix.lower()
        if suffix == ".mp4":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        elif suffix == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.frame_size,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter at {self.output_path}")
        print(f"[VideoWriter] Recording to '{self.output_path}'")
        return self

    def write(self, frame: np.ndarray):
        if self._writer is not None:
            # Ensure correct size
            fh, fw = frame.shape[:2]
            if (fw, fh) != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            self._writer.write(frame)

    def release(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"[VideoWriter] Saved '{self.output_path}'")

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.release()
