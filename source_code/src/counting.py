"""
src/counting.py
===============
Virtual Counting Zones — State Machine for product counting.

Three-zone architecture:
    ENTRY ZONE → TRACKING ZONE → EXIT ZONE

A product is officially counted only once it traverses all three zones
in the correct direction and sequence.  This eliminates:
  - Double-counting from ID switches near the exit boundary.
  - False counts from spurious detections that only appear in one zone.

The module also triggers optimal-viewpoint crop extraction for defect
classification: a crop is captured when the product centre is closest
to the midpoint of the TRACKING zone.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from src.tracking import Track, TrackState


# ---------------------------------------------------------------------------
# Product State Machine
# ---------------------------------------------------------------------------

class ProductState(Enum):
    DISCOVERED = auto()   # Track appeared; zone not yet evaluated
    ENTRY      = auto()   # Track is in the Entry zone
    TRACKING   = auto()   # Track has passed through Entry into Tracking
    EXIT       = auto()   # Track has reached the Exit zone
    COUNTED    = auto()   # Successfully counted — logged to DB
    TERMINATED = auto()   # Lost without completing the sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ZoneCounterConfig:
    """
    Runtime configuration for the Virtual Counting Zones.

    Coordinate fractions are along the *conveyor axis* (0.0 → 1.0):
      axis="x"  → fraction of frame width
      axis="y"  → fraction of frame height

    Attributes:
        axis:             "x" (horizontal conveyor) or "y" (vertical).
        zone_start:       Boundary between Entry and Tracking zones.
        zone_end:         Boundary between Tracking and Exit zones.
        direction:        "positive" (left→right / top→bottom) or "negative".
        count_cache_size: Size of recently-counted ID deque (anti-double-count).
        crop_margin:      Pixel padding around the bounding box for classifier crops.
    """
    axis: str = "x"
    zone_start: float = 0.20
    zone_end: float = 0.80
    direction: str = "positive"
    count_cache_size: int = 200
    crop_margin: int = 10


# ---------------------------------------------------------------------------
# Internal per-track record
# ---------------------------------------------------------------------------

@dataclass
class _TrackRecord:
    track_id: int
    state: ProductState = ProductState.DISCOVERED
    first_seen_coord: Optional[float] = None
    max_coord_seen: float = -np.inf          # for positive direction
    min_coord_seen: float = np.inf           # for negative direction
    crop_triggered: bool = False
    created_at: float = field(default_factory=time.time)
    counted_at: Optional[float] = None


# ---------------------------------------------------------------------------
# ZoneCounter
# ---------------------------------------------------------------------------

class ZoneCounter:
    """
    Manages per-track state machines and product counting logic.

    Usage::

        cfg = ZoneCounterConfig(axis="x", zone_start=0.20, zone_end=0.80)
        counter = ZoneCounter(cfg)

        # In the main loop:
        counts_delta, crops = counter.update(tracks, frame, frame_wh=(W, H))
        total_counted = counter.total_count
    """

    def __init__(self, config: ZoneCounterConfig):
        self.config = config
        self._records: Dict[int, _TrackRecord] = {}
        self._recently_counted: Deque[int] = deque(maxlen=config.count_cache_size)
        self.total_count: int = 0
        self.normal_count: int = 0
        self.defective_count: int = 0
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        tracks: List[Track],
        frame: Optional[np.ndarray],
        frame_wh: Tuple[int, int] = (1280, 720),
    ) -> Tuple[int, List[Tuple[int, np.ndarray]]]:
        """
        Process one frame's worth of track updates.

        Args:
            tracks:    Active ``Track`` objects from ByteTracker.
            frame:     Current BGR frame (used for cropping). May be None.
            frame_wh:  (width, height) of the frame.

        Returns:
            counts_delta:   Number of new products counted this frame.
            crop_requests:  List of (track_id, crop_image) pairs for defect
                            classification.  Crops are uint8 BGR arrays.
        """
        self._frame_count += 1
        W, H = frame_wh
        axis_size = W if self.config.axis == "x" else H

        active_ids: Set[int] = set()
        counts_delta = 0
        crop_requests: List[Tuple[int, np.ndarray]] = []

        for track in tracks:
            tid = track.track_id
            active_ids.add(tid)

            # Create a record for new tracks
            if tid not in self._records:
                self._records[tid] = _TrackRecord(track_id=tid)

            rec = self._records[tid]

            # Skip tracks that are already in terminal states
            if rec.state in (ProductState.COUNTED, ProductState.TERMINATED):
                continue

            # Determine normalised position along conveyor axis
            coord = (track.cx / W) if self.config.axis == "x" else (track.cy / H)

            # Initialise first-seen coordinate
            if rec.first_seen_coord is None:
                rec.first_seen_coord = coord

            # Update extremes (for direction validation)
            rec.max_coord_seen = max(rec.max_coord_seen, coord)
            rec.min_coord_seen = min(rec.min_coord_seen, coord)

            # Zone classification
            zone = self._classify_zone(coord)

            # State machine transitions
            new_state = self._transition(rec, zone, coord)

            if new_state != rec.state:
                rec.state = new_state

            # Trigger crop at optimal viewpoint (centre of Tracking zone)
            if (
                rec.state == ProductState.TRACKING
                and not rec.crop_triggered
                and frame is not None
            ):
                mid = (self.config.zone_start + self.config.zone_end) / 2.0
                if abs(coord - mid) < 0.05:  # within 5% of zone centre
                    crop = self._extract_crop(frame, track, W, H)
                    if crop is not None:
                        rec.crop_triggered = True
                        crop_requests.append((tid, crop))

            # Count when a product reaches COUNTED state
            if rec.state == ProductState.COUNTED and rec.counted_at is None:
                if tid not in self._recently_counted:
                    rec.counted_at = time.time()
                    self._recently_counted.append(tid)
                    self.total_count += 1
                    counts_delta += 1
                else:
                    # Duplicate — already counted
                    rec.state = ProductState.TERMINATED

        # Terminate records for tracks that have disappeared
        for tid, rec in list(self._records.items()):
            if tid not in active_ids and rec.state not in (
                ProductState.COUNTED, ProductState.TERMINATED
            ):
                rec.state = ProductState.TERMINATED

        return counts_delta, crop_requests

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_zone(self, coord: float) -> str:
        """Return "entry", "tracking", or "exit" based on coordinate."""
        if coord < self.config.zone_start:
            return "entry"
        elif coord <= self.config.zone_end:
            return "tracking"
        else:
            return "exit"

    def _transition(
        self, rec: _TrackRecord, zone: str, coord: float
    ) -> ProductState:
        """Apply zone-based transitions to a _TrackRecord."""
        state = rec.state

        # Validate direction of travel
        if not self._direction_ok(rec, coord):
            return state

        if state == ProductState.DISCOVERED:
            if zone == "entry":
                return ProductState.ENTRY
            # Appeared already in tracking or exit — suspicious, might be spurious
            return state

        if state == ProductState.ENTRY:
            if zone == "tracking":
                return ProductState.TRACKING
            if zone == "exit":
                # Skipped tracking zone — likely spurious, do not count
                return ProductState.TERMINATED
            return state  # still in entry

        if state == ProductState.TRACKING:
            if zone == "exit":
                return ProductState.EXIT
            return state  # still tracking

        if state == ProductState.EXIT:
            # Confirm count on first arrival in exit zone
            return ProductState.COUNTED

        return state  # COUNTED / TERMINATED don't transition further

    def _direction_ok(self, rec: _TrackRecord, coord: float) -> bool:
        """Return True if the track is moving in the expected direction."""
        if rec.first_seen_coord is None:
            return True
        delta = coord - rec.first_seen_coord
        if self.config.direction == "positive":
            # Should be moving toward higher coordinate values
            return delta >= -0.25  # allow larger backward jitter due to rotation
        else:
            return delta <= 0.25

    def _extract_crop(
        self, frame: np.ndarray, track: Track, W: int, H: int
    ) -> Optional[np.ndarray]:
        """Extract a padded bounding-box crop from the frame."""
        x1, y1, x2, y2 = track.last_bbox
        m = self.config.crop_margin
        x1c = max(0, int(x1) - m)
        y1c = max(0, int(y1) - m)
        x2c = min(W, int(x2) + m)
        y2c = min(H, int(y2) + m)
        if x2c <= x1c or y2c <= y1c:
            return None
        return frame[y1c:y2c, x1c:x2c].copy()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state_map(self) -> Dict[int, ProductState]:
        """Return {track_id: ProductState} for all known tracks."""
        return {tid: rec.state for tid, rec in self._records.items()}

    def register_defect_result(self, track_id: int, is_defective: bool):
        """Update normal/defective counters from classifier results."""
        if is_defective:
            self.defective_count += 1
        else:
            self.normal_count += 1

    def reset(self):
        """Clear all state (e.g. between sessions)."""
        self._records.clear()
        self._recently_counted.clear()
        self.total_count = 0
        self.normal_count = 0
        self.defective_count = 0
        self._frame_count = 0

    @property
    def pending_defect_count(self) -> int:
        """Products counted but not yet classified."""
        return self.total_count - self.normal_count - self.defective_count

    def get_zone_boundaries(
        self, frame_wh: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Return pixel coordinates of the two zone boundary lines.

        Returns:
            (line1_px, line2_px) as integers along the conveyor axis.
        """
        W, H = frame_wh
        size = W if self.config.axis == "x" else H
        return (
            int(self.config.zone_start * size),
            int(self.config.zone_end * size),
        )
