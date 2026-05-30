"""
scripts/generate_demo_video.py
==============================
Generates a synthetic MP4 video simulating products moving along a conveyor belt.

The script creates coloured shapes (circles and rectangles) that travel from
left to right across the frame.  A fraction of objects are rendered as
"defective" (irregular shapes / different colours) to exercise the full pipeline.

Usage:
    python scripts/generate_demo_video.py [--output demo/demo_conveyor.mp4]
                                          [--width 1280] [--height 720]
                                          [--fps 30] [--duration 60]
                                          [--num-products 8]
                                          [--defect-rate 0.15]
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Product simulation model
# ---------------------------------------------------------------------------

@dataclass
class SimProduct:
    pid: int
    x: float           # Current centre X
    y: float           # Current centre Y (fixed lane)
    speed: float       # Pixels per frame
    size: int          # Approx radius / half-side
    is_defective: bool
    colour: Tuple[int, int, int]       # BGR Normal colour
    defect_colour: Tuple[int, int, int] = (30, 30, 200)  # BGR Defective
    shape: str = "box"                 # "box" | "circle" | "diamond"
    noise_amp: float = 0.0             # Y-axis jitter amplitude
    _phase: float = field(default_factory=lambda: random.uniform(0, 2 * math.pi))

    def step(self):
        self.x += self.speed
        self.y += self.noise_amp * math.sin(self._phase)
        self._phase += 0.15

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) in integer pixels."""
        s = self.size
        return (
            int(self.x - s), int(self.y - s),
            int(self.x + s), int(self.y + s),
        )

    def draw(self, frame: np.ndarray):
        colour = self.defect_colour if self.is_defective else self.colour
        x1, y1, x2, y2 = self.bbox
        cx, cy, s = int(self.x), int(self.y), self.size

        if self.shape == "circle":
            cv2.circle(frame, (cx, cy), s, colour, -1)
            cv2.circle(frame, (cx, cy), s, (0, 0, 0), 1)
        elif self.shape == "diamond":
            pts = np.array([
                [cx,     cy - s],
                [cx + s, cy    ],
                [cx,     cy + s],
                [cx - s, cy    ],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [pts], colour)
            cv2.polylines(frame, [pts], True, (0, 0, 0), 1)
        else:  # box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)

        if self.is_defective:
            # Draw a red 'X' mark to visually indicate defect
            m = max(3, s // 3)
            cv2.line(frame, (cx - m, cy - m), (cx + m, cy + m), (0, 0, 255), 2)
            cv2.line(frame, (cx + m, cy - m), (cx - m, cy + m), (0, 0, 255), 2)


# ---------------------------------------------------------------------------
# Background (conveyor belt texture)
# ---------------------------------------------------------------------------

def _draw_belt(frame: np.ndarray, tick: int, speed: float = 4.0):
    """Draw a scrolling conveyor belt background."""
    H, W = frame.shape[:2]

    # Base belt colour (dark grey)
    frame[:] = (40, 40, 40)

    # Belt slats (repeating vertical bars)
    slat_w = 60
    offset = int(tick * speed) % slat_w
    for x in range(-slat_w + offset, W + slat_w, slat_w):
        cv2.rectangle(frame, (x, 0), (x + slat_w // 2, H), (48, 48, 48), -1)

    # Top & bottom belt rails
    cv2.rectangle(frame, (0, 0), (W, 18), (25, 25, 25), -1)
    cv2.rectangle(frame, (0, H - 18), (W, H), (25, 25, 25), -1)


# ---------------------------------------------------------------------------
# Zone annotation on demo video
# ---------------------------------------------------------------------------

def _draw_demo_zones(frame: np.ndarray, W: int, H: int, z1: int, z2: int):
    """Lightly mark the counting zones on the demo video."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),   (z1, H), (200, 160, 50), -1)
    cv2.rectangle(overlay, (z1, 0),  (z2, H), (50, 200, 220), -1)
    cv2.rectangle(overlay, (z2, 0),  (W,  H), (50, 50, 200),  -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.line(frame, (z1, 0), (z1, H), (220, 160, 30), 1)
    cv2.line(frame, (z2, 0), (z2, H), (30, 80, 220), 1)

    for label, lx in [("ENTRY", W // 8), ("TRACKING", (z1 + z2) // 2 - 32), ("EXIT", z2 + W // 16)]:
        cv2.putText(frame, label, (lx, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_demo_video(
    output_path: str = "demo/demo_conveyor.mp4",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_s: int = 60,
    num_lanes: int = 3,
    num_products: int = 8,
    defect_rate: float = 0.15,
    product_speed_range: Tuple[float, float] = (3.5, 7.0),
):
    """
    Generate a synthetic conveyor belt demo video.

    Args:
        output_path:          Path to the output MP4 file.
        width:                Frame width in pixels.
        height:               Frame height in pixels.
        fps:                  Output frames per second.
        duration_s:           Duration of the video in seconds.
        num_lanes:            Number of parallel product lanes.
        num_products:         Maximum simultaneous products on screen.
        defect_rate:          Fraction of products that are "defective".
        product_speed_range:  (min, max) pixels per frame for product speed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = fps * duration_s
    lane_ys = [int(height * (i + 1) / (num_lanes + 1)) for i in range(num_lanes)]

    # Zone boundaries (for annotation)
    z1 = int(width * 0.20)
    z2 = int(width * 0.80)

    # Spawn parameters
    product_size_range = (20, 40)
    spawn_interval_range = (fps // 2, fps * 2)  # frames

    COLOURS = [
        (50, 200, 80),    # Green
        (230, 180, 50),   # Teal
        (200, 80, 200),   # Purple
        (50, 180, 230),   # Orange
        (180, 50, 50),    # Blue
        (80, 220, 220),   # Yellow
    ]
    SHAPES = ["box", "circle", "diamond"]

    products: List[SimProduct] = []
    pid_counter = 0
    lane_spawn_timers = [random.randint(0, spawn_interval_range[1]) for _ in lane_ys]

    print(f"[Demo] Generating {duration_s}s demo video at {fps} FPS → '{output_path}'")

    for tick in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        _draw_belt(frame, tick)

        # Spawn new products per lane
        for li, (lane_y, timer) in enumerate(zip(lane_ys, lane_spawn_timers)):
            if timer <= 0 and len(products) < num_products * 2:
                is_def = random.random() < defect_rate
                colour = random.choice(COLOURS)
                speed = random.uniform(*product_speed_range)
                size = random.randint(*product_size_range)
                shape = random.choice(SHAPES)
                jitter = random.uniform(0.0, 2.5)
                products.append(
                    SimProduct(
                        pid=pid_counter,
                        x=float(-size),
                        y=float(lane_y + random.randint(-12, 12)),
                        speed=speed,
                        size=size,
                        is_defective=is_def,
                        colour=colour,
                        shape=shape,
                        noise_amp=jitter,
                    )
                )
                pid_counter += 1
                lane_spawn_timers[li] = random.randint(*spawn_interval_range)
            else:
                lane_spawn_timers[li] -= 1

        # Draw zones
        _draw_demo_zones(frame, width, height, z1, z2)

        # Step and draw products
        alive = []
        for prod in products:
            prod.step()
            prod.draw(frame)
            if prod.x < width + prod.size + 10:
                alive.append(prod)
        products = alive

        # HUD info
        progress = (tick + 1) / total_frames * 100
        cv2.putText(frame, f"DEMO | Frame {tick+1}/{total_frames} ({progress:.0f}%)",
                    (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

        writer.write(frame)

        # Console progress
        if (tick + 1) % (fps * 5) == 0:
            print(f"  Progress: {progress:.0f}%  (products spawned: {pid_counter})")

    writer.release()
    print(f"[Demo] Done! Video saved to '{output_path}' ({pid_counter} products generated).")
    return str(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic conveyor belt demo video")
    parser.add_argument("--output",       default="demo/demo_conveyor.mp4")
    parser.add_argument("--width",        type=int,   default=1280)
    parser.add_argument("--height",       type=int,   default=720)
    parser.add_argument("--fps",          type=int,   default=30)
    parser.add_argument("--duration",     type=int,   default=60,  help="Duration in seconds")
    parser.add_argument("--num-products", type=int,   default=8)
    parser.add_argument("--defect-rate",  type=float, default=0.15)
    args = parser.parse_args()

    generate_demo_video(
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_s=args.duration,
        num_products=args.num_products,
        defect_rate=args.defect_rate,
    )


if __name__ == "__main__":
    main()
