"""
src/utils/visualization.py
==========================
Frame annotation drawing utilities for the Conveyor Belt CV System.

Provides:
  - draw_tracks()  — coloured bounding boxes, track IDs, trajectories
  - draw_zones()   — Entry / Tracking / Exit zone boundary lines
  - draw_hud()     — Heads-Up Display overlay (FPS, counts, status)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.counting import ProductState
from src.tracking import Track


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

# BGR colours
COLOUR_NORMAL       = (50, 200, 50)      # Green — Normal product
COLOUR_DEFECTIVE    = (30, 30, 220)      # Red   — Defective product
COLOUR_TRACKING     = (20, 200, 220)     # Yellow — In Tracking zone
COLOUR_TENTATIVE    = (150, 150, 150)    # Grey  — Tentative track
COLOUR_ENTRY_ZONE   = (255, 180, 0)      # Blue  — Entry line
COLOUR_EXIT_ZONE    = (0, 120, 255)      # Orange — Exit line
COLOUR_HUD_BG       = (15, 15, 15)       # Dark HUD background
COLOUR_TEXT         = (230, 230, 230)    # Light text

_STATE_COLOURS = {
    ProductState.DISCOVERED: COLOUR_TENTATIVE,
    ProductState.ENTRY:      COLOUR_ENTRY_ZONE,
    ProductState.TRACKING:   COLOUR_TRACKING,
    ProductState.EXIT:       COLOUR_EXIT_ZONE,
    ProductState.COUNTED:    COLOUR_NORMAL,
    ProductState.TERMINATED: COLOUR_TENTATIVE,
}

# Font
_FONT      = cv2.FONT_HERSHEY_DUPLEX
_FONT_SM   = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    state_map: Optional[Dict[int, ProductState]] = None,
    defect_ids: Optional[set] = None,
    draw_trajectory: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and track IDs on *frame* (in-place).

    Args:
        frame:           BGR uint8 image.
        tracks:          List of active ``Track`` objects.
        state_map:       Mapping track_id → ProductState for colour coding.
        defect_ids:      Set of track IDs classified as defective.
        draw_trajectory: If True, draw the recent trajectory polyline.

    Returns:
        The annotated frame (same array, modified in-place).
    """
    state_map  = state_map  or {}
    defect_ids = defect_ids or set()

    for track in tracks:
        tid = track.track_id
        x1, y1, x2, y2 = [int(v) for v in track.last_bbox]

        # Choose colour
        if tid in defect_ids:
            colour = COLOUR_DEFECTIVE
        else:
            state = state_map.get(tid, ProductState.DISCOVERED)
            colour = _STATE_COLOURS.get(state, COLOUR_TENTATIVE)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Track ID + state label
        state = state_map.get(tid, ProductState.DISCOVERED)
        label = f"ID:{tid} {track.class_name}"
        (tw, th), _ = cv2.getTextSize(label, _FONT_SM, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), _FONT_SM, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Trajectory polyline
        if draw_trajectory and len(track.history) > 1:
            pts = np.array(
                [(int(cx), int(cy)) for cx, cy in track.history[-30:]],
                dtype=np.int32,
            )
            cv2.polylines(frame, [pts], False, colour, 1, cv2.LINE_AA)

    return frame


def draw_zones(
    frame: np.ndarray,
    line1_px: int,
    line2_px: int,
    axis: str = "x",
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Draw semi-transparent zone regions and boundary lines.

    Args:
        frame:    BGR uint8 image.
        line1_px: Pixel coordinate of Entry/Tracking boundary.
        line2_px: Pixel coordinate of Tracking/Exit boundary.
        axis:     "x" (vertical boundary lines) or "y" (horizontal).
        alpha:    Transparency of zone fill (0=transparent, 1=opaque).

    Returns:
        Annotated frame.
    """
    H, W = frame.shape[:2]
    overlay = frame.copy()

    if axis == "x":
        # Entry zone fill (left of line1)
        cv2.rectangle(overlay, (0, 0), (line1_px, H), (255, 200, 50), -1)
        # Tracking zone fill (between lines)
        cv2.rectangle(overlay, (line1_px, 0), (line2_px, H), (50, 200, 255), -1)
        # Exit zone fill (right of line2)
        cv2.rectangle(overlay, (line2_px, 0), (W, H), (50, 50, 200), -1)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Boundary lines
        cv2.line(frame, (line1_px, 0), (line1_px, H), COLOUR_ENTRY_ZONE, 2)
        cv2.line(frame, (line2_px, 0), (line2_px, H), COLOUR_EXIT_ZONE, 2)

        # Zone labels
        _zone_label(frame, "ENTRY", 0, line1_px, H, "x")
        _zone_label(frame, "TRACKING", line1_px, line2_px, H, "x")
        _zone_label(frame, "EXIT", line2_px, W, H, "x")
    else:
        # Y-axis (horizontal conveyor view)
        cv2.rectangle(overlay, (0, 0), (W, line1_px), (255, 200, 50), -1)
        cv2.rectangle(overlay, (0, line1_px), (W, line2_px), (50, 200, 255), -1)
        cv2.rectangle(overlay, (0, line2_px), (W, H), (50, 50, 200), -1)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.line(frame, (0, line1_px), (W, line1_px), COLOUR_ENTRY_ZONE, 2)
        cv2.line(frame, (0, line2_px), (W, line2_px), COLOUR_EXIT_ZONE, 2)

        _zone_label(frame, "ENTRY", 0, line1_px, W, "y")
        _zone_label(frame, "TRACKING", line1_px, line2_px, W, "y")
        _zone_label(frame, "EXIT", line2_px, H, W, "y")

    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    total: int,
    normal: int,
    defective: int,
    frame_idx: int = 0,
    simulate_mode: bool = False,
) -> np.ndarray:
    """
    Draw a Heads-Up Display (HUD) panel in the top-right corner.

    Args:
        frame:         BGR uint8 image.
        fps:           Current processing FPS.
        total:         Total products counted.
        normal:        Normal products.
        defective:     Defective products.
        frame_idx:     Current frame index.
        simulate_mode: If True, show "DEMO MODE" badge.

    Returns:
        Annotated frame.
    """
    H, W = frame.shape[:2]

    # Panel dimensions
    panel_w, panel_h = 280, 130
    px1 = W - panel_w - 10
    py1 = 10

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px1 + panel_w, py1 + panel_h), COLOUR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Border
    cv2.rectangle(frame, (px1, py1), (px1 + panel_w, py1 + panel_h), (80, 80, 80), 1)

    # Title bar
    cv2.rectangle(frame, (px1, py1), (px1 + panel_w, py1 + 22), (40, 40, 40), -1)
    cv2.putText(frame, "Conveyor Monitor", (px1 + 6, py1 + 16), _FONT_SM, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    if simulate_mode:
        cv2.putText(frame, "DEMO", (px1 + panel_w - 52, py1 + 16), _FONT_SM, 0.45, (0, 200, 255), 1)

    # Stats
    stats = [
        (f"FPS:       {fps:5.1f}",        COLOUR_TEXT),
        (f"Frame:     {frame_idx}",         COLOUR_TEXT),
        (f"Total:     {total}",             (220, 220, 100)),
        (f"Normal:    {normal}",            (80, 200, 80)),
        (f"Defective: {defective}",         (80, 80, 220)),
    ]

    y = py1 + 38
    for text, colour in stats:
        cv2.putText(frame, text, (px1 + 8, y), _FONT_SM, 0.48, colour, 1, cv2.LINE_AA)
        y += 18

    return frame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _zone_label(
    frame: np.ndarray,
    text: str,
    start: int,
    end: int,
    height_or_width: int,
    axis: str,
):
    mid = (start + end) // 2
    if axis == "x":
        x, y = mid - 30, 22
    else:
        x, y = 8, mid + 6

    (tw, _), _ = cv2.getTextSize(text, _FONT_SM, 0.45, 1)
    # Background pill
    cv2.rectangle(frame, (x - 3, y - 14), (x + tw + 3, y + 3), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), _FONT_SM, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
