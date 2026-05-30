"""
src/tracking.py
===============
Pure-Python ByteTrack-style Multi-Object Tracker for the Conveyor Belt CV System.

Design goals:
  - No hard dependency on any external tracker library (only numpy + scipy).
  - Implements the two-stage high/low confidence matching strategy of ByteTrack.
  - Kalman Filter state: [cx, cy, aspect_ratio, height, vx, vy, va, vh].
  - Returns ``Track`` objects consumed directly by counting.py.

References:
  Zhang, Y. et al., "ByteTrack: Multi-Object Tracking by Associating Every
  Detection Box", ECCV 2022.  https://arxiv.org/abs/2110.06864
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection import Detection


# ---------------------------------------------------------------------------
# Track state enum
# ---------------------------------------------------------------------------

class TrackState(IntEnum):
    Tentative = 0    # Recently appeared, not yet confirmed
    Confirmed = 1    # Confirmed active track
    Lost      = 2    # Missing for ≤ max_age frames
    Removed   = 3    # Expired; should be pruned


# ---------------------------------------------------------------------------
# Kalman Filter (constant velocity model)
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    Kalman Filter modelling object motion as constant velocity in 2-D space.

    State vector:  x = [cx, cy, a, h, vx, vy, va, vh]^T
    Measurement:   z = [cx, cy, a, h]^T
    """

    ndim = 4           # measurement dims
    dt   = 1.0         # time step (1 frame)

    def __init__(self):
        # Transition matrix F
        self.F = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self.F[i, self.ndim + i] = self.dt

        # Measurement matrix H
        self.H = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)

        # Noise covariances (tunable)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def _Q(self, height: float) -> np.ndarray:
        """Process noise covariance."""
        p = self._std_weight_position * height
        v = self._std_weight_velocity * height
        stds = [p, p, 1e-2, p, v, v, 1e-5, v]
        return np.diag(np.square(stds, dtype=np.float32))

    def _R(self, height: float) -> np.ndarray:
        """Measurement noise covariance."""
        p = self._std_weight_position * height
        stds = [p, p, 1e-1, p]
        return np.diag(np.square(stds, dtype=np.float32))

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialise a new track from a measurement [cx, cy, a, h]."""
        mean = np.r_[measurement, np.zeros(self.ndim, dtype=np.float32)]
        h = measurement[3]
        p = self._std_weight_position * h
        v = self._std_weight_velocity * h
        stds = [2 * p, 2 * p, 1e-2, 2 * p, 10 * v, 10 * v, 1e-5, 10 * v]
        covariance = np.diag(np.square(stds, dtype=np.float32))
        return mean.astype(np.float32), covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate state one step forward."""
        Q = self._Q(mean[3])
        mean_new = self.F @ mean
        covariance_new = self.F @ covariance @ self.F.T + Q
        return mean_new, covariance_new

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct state with a new measurement."""
        R = self._R(mean[3])
        S = self.H @ covariance @ self.H.T + R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ mean
        mean_new = mean + K @ innovation
        covariance_new = (np.eye(2 * self.ndim, dtype=np.float32) - K @ self.H) @ covariance
        return mean_new, covariance_new

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project state into measurement space (for Mahalanobis distance)."""
        R = self._R(mean[3])
        proj_mean = self.H @ mean
        proj_cov = self.H @ covariance @ self.H.T + R
        return proj_mean, proj_cov


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

_GLOBAL_ID_COUNTER: int = 0


def _next_track_id() -> int:
    global _GLOBAL_ID_COUNTER
    _GLOBAL_ID_COUNTER += 1
    return _GLOBAL_ID_COUNTER


@dataclass
class Track:
    """
    Single tracked object across frames.

    Attributes:
        track_id:    Unique integer identifier.
        state:       Current ``TrackState``.
        hits:        Number of frames with associated detection.
        age:         Total frames this track has existed.
        time_since_update: Frames since last successful association.
        mean:        Kalman state mean vector.
        covariance:  Kalman state covariance matrix.
        last_bbox:   Most recent [x1, y1, x2, y2] in pixel space.
        class_id:    Detected object class.
        class_name:  Human-readable class name.
        score:       Detection confidence of last association.
        history:     List of centre-point (cx, cy) positions.
    """

    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    state: TrackState = TrackState.Tentative
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    class_id: int = 0
    class_name: str = "product"
    score: float = 1.0
    history: List[Tuple[float, float]] = field(default_factory=list)
    _kf: Optional[KalmanFilter] = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        detection: "Detection",
        kf: KalmanFilter,
        min_hits: int = 2,
    ) -> "Track":
        meas = _xyxy_to_xywh(detection.bbox)
        mean, cov = kf.initiate(meas)
        obj = cls(
            track_id=_next_track_id(),
            mean=mean,
            covariance=cov,
            class_id=detection.class_id,
            class_name=detection.class_name,
            score=detection.confidence,
            _kf=kf,
        )
        obj.history.append((detection.cx, detection.cy))
        return obj

    # ------------------------------------------------------------------
    def predict(self):
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: "Detection"):
        meas = _xyxy_to_xywh(detection.bbox)
        self.mean, self.covariance = self._kf.update(self.mean, self.covariance, meas)
        self.hits += 1
        self.time_since_update = 0
        self.score = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.history.append((detection.cx, detection.cy))
        if len(self.history) > 60:
            self.history = self.history[-60:]

    def mark_missed(self):
        if self.state == TrackState.Confirmed:
            self.state = TrackState.Lost

    # ------------------------------------------------------------------
    @property
    def last_bbox(self) -> List[float]:
        """Return predicted bounding box as [x1, y1, x2, y2]."""
        return _xywh_to_xyxy(self.mean[:4])

    @property
    def cx(self) -> float:
        return float(self.mean[0])

    @property
    def cy(self) -> float:
        return float(self.mean[1])

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_lost(self) -> bool:
        return self.state == TrackState.Lost

    def is_deleted(self) -> bool:
        return self.state == TrackState.Removed


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _xyxy_to_xywh(bbox: List[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=np.float32)


def _xywh_to_xyxy(xywh: np.ndarray) -> List[float]:
    cx, cy, a, h = xywh[:4]
    w = a * h
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------

def _iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of [x1,y1,x2,y2] boxes."""
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)

    # Broadcast intersection
    xx1 = np.maximum(bboxes_a[:, None, 0], bboxes_b[None, :, 0])
    yy1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[None, :, 1])
    xx2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[None, :, 2])
    yy2 = np.minimum(bboxes_a[:, None, 3], bboxes_b[None, :, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-6)


def _linear_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hungarian assignment. Returns (row_indices, col_indices)."""
    if cost_matrix.size == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def _match(
    tracks: List[Track],
    detections: List[Detection],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associate tracks with detections using IoU + Hungarian algorithm.

    Returns:
        matches:           List of (track_idx, det_idx) tuples.
        unmatched_tracks:  Track indices with no valid detection.
        unmatched_dets:    Detection indices with no valid track.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    track_bboxes = np.array([t.last_bbox for t in tracks], dtype=np.float32)
    det_bboxes = np.array([d.bbox for d in detections], dtype=np.float32)

    iou = _iou_matrix(track_bboxes, det_bboxes)
    cost = 1.0 - iou

    row_ind, col_ind = _linear_assignment(cost)

    matches, unmatched_tracks, unmatched_dets = [], [], []

    matched_track_set = set()
    matched_det_set = set()

    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_track_set.add(r)
            matched_det_set.add(c)

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_set]
    unmatched_dets = [j for j in range(len(detections)) if j not in matched_det_set]

    return matches, unmatched_tracks, unmatched_dets


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    ByteTrack-inspired multi-object tracker.

    Two-stage matching:
      1. High-confidence detections  → active tracks (IoU matching).
      2. Low-confidence detections   → unmatched active tracks.

    Args:
        track_thresh:    Confidence threshold for high-score detections.
        low_det_thresh:  Minimum confidence to consider at all.
        match_thresh:    IoU threshold for successful match.
        track_buffer:    Max frames a lost track is kept before deletion.
        min_hits:        Hits required to confirm a tentative track.
    """

    def __init__(
        self,
        track_thresh: float = 0.50,
        low_det_thresh: float = 0.20,
        match_thresh: float = 0.80,
        track_buffer: int = 30,
        min_hits: int = 2,
    ):
        self.track_thresh = track_thresh
        self.low_det_thresh = low_det_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits

        self._kf = KalmanFilter()
        self.active_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.frame_count: int = 0

    def reset(self):
        """Reset tracker state (e.g. between video clips)."""
        global _GLOBAL_ID_COUNTER
        _GLOBAL_ID_COUNTER = 0
        self.active_tracks = []
        self.lost_tracks = []
        self.frame_count = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Process a new frame's detections and return confirmed/lost tracks.

        Args:
            detections: List of ``Detection`` from the current frame.

        Returns:
            List of *confirmed* (or recently lost) ``Track`` objects.
        """
        self.frame_count += 1

        # Split detections into high / low confidence
        high_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_dets  = [d for d in detections if self.low_det_thresh <= d.confidence < self.track_thresh]

        # ------ Predict all existing tracks ------
        for t in self.active_tracks + self.lost_tracks:
            t.predict()

        # ------ Stage 1: High-confidence ↔ Active tracks ------
        matches1, unmatched_tracks1, unmatched_high_dets = _match(
            self.active_tracks, high_dets, self.match_thresh
        )

        for ti, di in matches1:
            self.active_tracks[ti].update(high_dets[di])
            if self.active_tracks[ti].state == TrackState.Tentative and \
               self.active_tracks[ti].hits >= self.min_hits:
                self.active_tracks[ti].state = TrackState.Confirmed

        # ------ Stage 2: Low-confidence ↔ Unmatched active tracks ------
        remaining_tracks = [self.active_tracks[i] for i in unmatched_tracks1
                            if self.active_tracks[i].state == TrackState.Confirmed]

        matches2, still_unmatched_tracks, _ = _match(
            remaining_tracks, low_dets, self.match_thresh
        )

        for ti, di in matches2:
            remaining_tracks[ti].update(low_dets[di])

        # Tracks not matched in either stage → mark missed
        for t in remaining_tracks:
            if t.time_since_update > 0:
                t.mark_missed()

        # Tentative tracks that missed stage 1
        for ti in unmatched_tracks1:
            t = self.active_tracks[ti]
            if t.state == TrackState.Tentative:
                t.state = TrackState.Removed
            elif t.state == TrackState.Confirmed:
                t.mark_missed()

        # ------ Stage 3: New detections → create tentative tracks ------
        # Also try to match unmatched high dets against lost tracks
        if self.lost_tracks:
            matches3, _, unmatched_high_dets2 = _match(
                self.lost_tracks, [high_dets[i] for i in unmatched_high_dets],
                self.match_thresh,
            )
            revived = set()
            for ti, di in matches3:
                self.lost_tracks[ti].update([high_dets[i] for i in unmatched_high_dets][di])
                self.lost_tracks[ti].state = TrackState.Confirmed
                self.active_tracks.append(self.lost_tracks[ti])
                revived.add(ti)
            self.lost_tracks = [t for i, t in enumerate(self.lost_tracks) if i not in revived]
            unmatched_high_dets = [unmatched_high_dets[i] for i in unmatched_high_dets2]

        for di in unmatched_high_dets:
            new_track = Track.create(high_dets[di], self._kf, self.min_hits)
            self.active_tracks.append(new_track)

        # ------ Housekeeping ------
        # Move sufficiently-lost confirmed tracks to lost pool
        new_active, going_lost = [], []
        for t in self.active_tracks:
            if t.state == TrackState.Lost:
                going_lost.append(t)
            elif t.state != TrackState.Removed:
                new_active.append(t)

        self.lost_tracks.extend(going_lost)

        # Delete tracks that have been lost too long
        self.lost_tracks = [
            t for t in self.lost_tracks
            if t.time_since_update <= self.track_buffer
        ]

        self.active_tracks = new_active

        # Return all tracks currently considered output-worthy
        output = [
            t for t in self.active_tracks
            if t.state in (TrackState.Confirmed, TrackState.Tentative)
        ]
        return output

    @property
    def all_tracks(self) -> List[Track]:
        return self.active_tracks + self.lost_tracks
