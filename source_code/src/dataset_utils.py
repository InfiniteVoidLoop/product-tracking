"""
src/dataset_utils.py
====================
Dataset utilities for the Conveyor Belt CV System.

Covers:
  - CLAHE + motion blur + Gaussian noise + HSV augmentation pipeline (Albumentations)
  - YOLODataset: torch Dataset for detection training
  - Helper to build the data.yaml required by Ultralytics
"""

import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# Augmentation Pipeline
# ---------------------------------------------------------------------------

def build_augmentation_pipeline(
    image_size: int = 640,
    p_motion_blur: float = 0.30,
    p_noise: float = 0.25,
    p_clahe: float = 0.40,
    p_brightness: float = 0.50,
    p_hsv: float = 0.40,
    training: bool = True,
):
    """
    Build an Albumentations augmentation pipeline suited for industrial
    conveyor belt environments.

    Args:
        image_size:      Target output resolution (square).
        p_motion_blur:   Probability of linear motion blur (simulates high-speed belts).
        p_noise:         Probability of Gaussian noise (simulates sensor/dust noise).
        p_clahe:         Probability of CLAHE (normalises factory lighting).
        p_brightness:    Probability of random brightness/contrast adjustment.
        p_hsv:           Probability of HSV colour jitter.
        training:        If False returns only resize+normalise (inference mode).

    Returns:
        An ``albumentations.Compose`` transform callable.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError(
            "albumentations is required. Install with: pip install albumentations"
        )

    if not training:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=114,
                ),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    return A.Compose(
        [
            # ---------- Spatial transforms ----------
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=114,
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.15,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.50,
            ),
            # ---------- Lighting / colour ----------
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=p_clahe),
            A.RandomBrightnessContrast(
                brightness_limit=0.30, contrast_limit=0.30, p=p_brightness
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=p_hsv
            ),
            # ---------- Industrial noise / blur ----------
            A.MotionBlur(blur_limit=(3, 15), p=p_motion_blur),
            A.GaussNoise(var_limit=(10, 80), p=p_noise),
            # ---------- Normalise ----------
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3
        ),
    )


# ---------------------------------------------------------------------------
# YOLO Dataset
# ---------------------------------------------------------------------------

class YOLODataset:
    """
    PyTorch-compatible Dataset for YOLO-format labelled images.

    Directory structure expected::

        root/
          images/  *.jpg | *.png
          labels/  *.txt   (YOLO format: class cx cy w h, normalised)

    Args:
        root:        Path to split folder (e.g. data/train/).
        transform:   Albumentations Compose pipeline.
        image_exts:  Accepted image file extensions.
    """

    def __init__(
        self,
        root: str | Path,
        transform=None,
        image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ):
        self.root = Path(root)
        self.transform = transform
        self.img_dir = self.root / "images"
        self.lbl_dir = self.root / "labels"

        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() in image_exts:
                lbl_path = self.lbl_dir / img_path.with_suffix(".txt").name
                self.samples.append((img_path, lbl_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, lbl_path = self.samples[idx]

        # Load image (BGR → RGB)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        bboxes, class_labels = [], []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = parts
                        bboxes.append([float(cx), float(cy), float(w), float(h)])
                        class_labels.append(int(cls))

        if self.transform:
            result = self.transform(
                image=img, bboxes=bboxes, class_labels=class_labels
            )
            img = result["image"]
            bboxes = result["bboxes"]
            class_labels = result["class_labels"]

        return {
            "image": img,
            "bboxes": bboxes,
            "class_labels": class_labels,
            "img_path": str(img_path),
        }


# ---------------------------------------------------------------------------
# data.yaml builder (Ultralytics format)
# ---------------------------------------------------------------------------

def build_data_yaml(
    dataset_root: str | Path,
    class_names: List[str],
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Generate a data.yaml file for Ultralytics YOLO training.

    Args:
        dataset_root:  Root folder containing train/ val/ test/ subdirectories.
        class_names:   Ordered list of class names.
        output_path:   Where to save the yaml file. Defaults to dataset_root/data.yaml.

    Returns:
        Absolute path to the saved yaml file.
    """
    dataset_root = Path(dataset_root).resolve()
    output_path = output_path or dataset_root / "data.yaml"
    output_path = Path(output_path)

    data = {
        "path": str(dataset_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"[dataset_utils] data.yaml written to: {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Augmented image generator (offline augmentation for small datasets)
# ---------------------------------------------------------------------------

def augment_and_save(
    source_dir: str | Path,
    output_dir: str | Path,
    multiplier: int = 5,
    image_size: int = 640,
):
    """
    Apply the augmentation pipeline ``multiplier`` times to every image in
    ``source_dir`` and write the results to ``output_dir``.

    Useful for boosting small custom datasets before YOLO training.

    Args:
        source_dir:  Directory with images/ and labels/ sub-folders.
        output_dir:  Destination directory (created if absent).
        multiplier:  How many augmented copies to generate per original image.
        image_size:  Target square resolution.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    transform = build_augmentation_pipeline(image_size=image_size, training=True)
    # Use a simpler, numpy-returning version (without ToTensorV2) for saving
    import albumentations as A

    save_transform = A.Compose(
        [t for t in transform.transforms if not t.__class__.__name__ == "ToTensorV2"],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3
        ),
    )

    dataset = YOLODataset(source_dir, transform=None)
    count = 0
    for img_path, lbl_path in dataset.samples:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = [], []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = parts
                        bboxes.append([float(cx), float(cy), float(w), float(h)])
                        class_labels.append(int(cls))

        for i in range(multiplier):
            try:
                res = save_transform(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_img = cv2.cvtColor(res["image"], cv2.COLOR_RGB2BGR)
                stem = f"{img_path.stem}_aug{i:03d}"
                cv2.imwrite(str(output_dir / "images" / f"{stem}.jpg"), aug_img)

                with open(output_dir / "labels" / f"{stem}.txt", "w") as f:
                    for cls, box in zip(res["class_labels"], res["bboxes"]):
                        f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")
                count += 1
            except Exception as e:
                print(f"[augment_and_save] Warning: {e}")

    print(f"[augment_and_save] Generated {count} augmented samples → {output_dir}")
