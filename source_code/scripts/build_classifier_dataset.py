"""
scripts/build_classifier_dataset.py
====================================
Builds the defect classifier dataset from existing ABO product images.

Strategy:
  - Normal    : random samples from data/train/images/ (clean catalogue photos)
  - Defective : same samples with synthetic degradation applied
                (scratches, blobs, channel corruption, heavy noise, blur)

Output structure:
  data/classifier/
  ├── train/
  │   ├── Normal/     (n_train images)
  │   └── Defective/  (n_train images)
  └── val/
      ├── Normal/     (n_val images)
      └── Defective/  (n_val images)

Usage:
    python scripts/build_classifier_dataset.py [--n-train 2000] [--n-val 400]
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Synthetic defect augmentation
# ---------------------------------------------------------------------------

def add_scratch(img: np.ndarray) -> np.ndarray:
    """Draw random thin dark lines across the image."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(random.randint(2, 6)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        color = (random.randint(0, 60),) * 3
        thickness = random.randint(1, 3)
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def add_blob(img: np.ndarray) -> np.ndarray:
    """Draw random dark or bright elliptical stains."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(random.randint(1, 4)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        rx, ry = random.randint(5, w // 4), random.randint(5, h // 4)
        color = (random.randint(0, 80),) * 3 if random.random() < 0.7 \
            else (random.randint(180, 255),) * 3
        cv2.ellipse(out, (cx, cy), (rx, ry), random.randint(0, 180),
                    0, 360, color, -1)
    overlay = cv2.addWeighted(img, 0.35, out, 0.65, 0)
    return overlay


def corrupt_channel(img: np.ndarray) -> np.ndarray:
    """Shift/scale one colour channel to simulate colour distortion."""
    out = img.copy().astype(np.float32)
    ch = random.randint(0, 2)
    out[:, :, ch] = np.clip(out[:, :, ch] * random.uniform(0.2, 0.6) +
                            random.uniform(-40, 40), 0, 255)
    return out.astype(np.uint8)


def add_noise(img: np.ndarray) -> np.ndarray:
    """Add heavy Gaussian noise."""
    noise = np.random.normal(0, random.uniform(25, 60), img.shape).astype(np.int16)
    out = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


def heavy_blur(img: np.ndarray) -> np.ndarray:
    """Apply strong motion or Gaussian blur."""
    k = random.choice([15, 21, 31])
    if random.random() < 0.5:
        return cv2.GaussianBlur(img, (k, k), 0)
    # Motion blur
    kernel = np.zeros((k, k))
    if random.random() < 0.5:
        kernel[k // 2, :] = 1
    else:
        np.fill_diagonal(kernel, 1)
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def make_defective(img: np.ndarray) -> np.ndarray:
    """Apply 2-4 random defect augmentations to produce a defective sample."""
    augmentations = [add_scratch, add_blob, corrupt_channel, add_noise, heavy_blur]
    chosen = random.sample(augmentations, k=random.randint(2, 4))
    for aug in chosen:
        img = aug(img)
    return img


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(n_train: int, n_val: int, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    src_images = Path(ROOT / "data" / "train" / "images")
    if not src_images.exists():
        print(f"[ERROR] Source images not found: {src_images}")
        print("  Download the ABO dataset first (see README).")
        raise SystemExit(1)

    all_imgs = list(src_images.glob("*.jpg"))
    total_needed = (n_train + n_val) * 2          # Normal + Defective
    if len(all_imgs) < n_train + n_val:
        print(f"[WARNING] Only {len(all_imgs)} source images, "
              f"reducing to {len(all_imgs) // 2} per split.")
        n_train = len(all_imgs) // 3
        n_val   = len(all_imgs) // 6

    random.shuffle(all_imgs)
    train_pool = all_imgs[:n_train]
    val_pool   = all_imgs[n_train: n_train + n_val]

    out_root = ROOT / "data" / "classifier"

    for split, pool in [("train", train_pool), ("val", val_pool)]:
        for cls in ["Normal", "Defective"]:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

        print(f"\n[{split.upper()}] Processing {len(pool)} images ...")
        for img_path in tqdm(pool, desc=f"  {split}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))

            stem = img_path.stem

            # Normal — save as-is
            cv2.imwrite(str(out_root / split / "Normal" / f"{stem}.jpg"), img)

            # Defective — synthesise degradation
            defective = make_defective(img)
            cv2.imwrite(str(out_root / split / "Defective" / f"{stem}_def.jpg"), defective)

    # Summary
    for split in ["train", "val"]:
        for cls in ["Normal", "Defective"]:
            n = len(list((out_root / split / cls).glob("*.jpg")))
            print(f"  {split}/{cls}: {n} images")

    print(f"\n✅ Classifier dataset ready at: {out_root}")


def main():
    parser = argparse.ArgumentParser(description="Build defect classifier dataset")
    parser.add_argument("--n-train", type=int, default=2000,
                        help="Normal/Defective images in train split (each)")
    parser.add_argument("--n-val",   type=int, default=400,
                        help="Normal/Defective images in val split (each)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    print(f"Building classifier dataset: {args.n_train} train / {args.n_val} val per class")
    build(args.n_train, args.n_val, args.seed)


if __name__ == "__main__":
    main()
