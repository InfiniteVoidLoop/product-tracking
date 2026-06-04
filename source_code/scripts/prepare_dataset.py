"""
scripts/prepare_dataset.py
==========================
Prepares the dataset for YOLO training:
  1. Creates the expected directory structure (data/train|val|test/images|labels).
  2. Optionally applies offline augmentation via Albumentations.
  3. Generates a data.yaml compatible with Ultralytics YOLO.

Usage:
    python scripts/prepare_dataset.py [--source data/raw]
                                      [--output data]
                                      [--class-names product]
                                      [--extract-frames 50]
                                      [--augment]
                                      [--multiplier 5]

Supply --source pointing to a flat folder of labelled YOLO images, or a folder
already structured as images/ + labels/ subdirectories. If --source points to a
video file, pass --extract-frames to sample frames before splitting.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_under_root(path: str | Path) -> Path:
    """Resolve project-local paths from source_code/, independent of shell cwd."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def extract_frames_from_video(
    video_path: Path,
    raw_dir: Path,
    frame_count: int,
):
    """
    Evenly sample frames from a video into a YOLO-style raw/images folder.

    Matching empty label files are written so downstream YOLO tooling sees a
    complete images/ + labels/ structure. Fill those labels after annotation.
    """
    if frame_count <= 0:
        print("[ERROR] --extract-frames must be greater than 0 for video sources.")
        sys.exit(1)

    if not video_path.exists():
        print(f"[ERROR] Video source does not exist: {video_path}")
        sys.exit(1)

    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV is required for video extraction. Install opencv-python.")
        sys.exit(1)

    img_dir = raw_dir / "images"
    lbl_dir = raw_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"[ERROR] Could not determine frame count for: {video_path}")
        cap.release()
        sys.exit(1)

    samples = min(frame_count, total_frames)
    if samples == 1:
        frame_indices = [0]
    else:
        frame_indices = [
            round(i * (total_frames - 1) / (samples - 1))
            for i in range(samples)
        ]

    saved = 0
    stem = video_path.stem
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  [WARNING] Could not read frame {frame_idx}; skipping.")
            continue

        image_name = f"{stem}_f{frame_idx:06d}.jpg"
        image_path = img_dir / image_name
        label_path = lbl_dir / image_path.with_suffix(".txt").name

        if not cv2.imwrite(str(image_path), frame):
            print(f"  [WARNING] Could not write image: {image_path}")
            continue

        label_path.touch()
        saved += 1

    cap.release()

    if saved == 0:
        print("[ERROR] No frames were extracted from the video.")
        sys.exit(1)

    print(f"  Extracted {saved} frames from {video_path} → {img_dir}")
    print("  Empty labels were created. Annotate them before training.")


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    # test gets the remainder
):
    """
    Split a source folder into train/val/test splits.

    Supports either:
      - YOLO format: source/images + source/labels
      - image-only format: source contains images directly or in subfolders

    For image-only sources, each image is labelled as one full-frame product.
    That matches object-centric datasets such as ABO product images.
    """
    import random

    yolo_img_dir = source_dir / "images"
    yolo_lbl_dir = source_dir / "labels"
    has_yolo_layout = yolo_img_dir.exists()

    img_dir = yolo_img_dir if has_yolo_layout else source_dir
    lbl_dir = yolo_lbl_dir if has_yolo_layout else None

    if not img_dir.exists():
        print(f"[ERROR] Source folder does not exist: {img_dir}")
        sys.exit(1)

    imgs = sorted(
        path for path in img_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not imgs:
        print(f"[ERROR] No images found in {img_dir}")
        sys.exit(1)

    if lbl_dir is None:
        print("  No labels/ folder found; creating full-frame YOLO labels.")
    elif not lbl_dir.exists():
        print(f"  No labels folder found at {lbl_dir}; creating empty labels.")

    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": imgs[:n_train],
        "val": imgs[n_train : n_train + n_val],
        "test": imgs[n_train + n_val :],
    }

    for split, split_imgs in splits.items():
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            rel_path = img_path.relative_to(img_dir)
            safe_stem = "_".join(rel_path.with_suffix("").parts)
            image_name = f"{safe_stem}{img_path.suffix.lower()}"
            label_name = f"{safe_stem}.txt"

            shutil.copy2(img_path, img_out / image_name)

            if lbl_dir and lbl_dir.exists():
                lbl_path = lbl_dir / rel_path.with_suffix(".txt")
                if not lbl_path.exists():
                    lbl_path = lbl_dir / img_path.with_suffix(".txt").name
                if lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_out / label_name)
                else:
                    (lbl_out / label_name).touch()
            elif lbl_dir is None:
                (lbl_out / label_name).write_text("0 0.5 0.5 1.0 1.0\n")
            else:
                (lbl_out / label_name).touch()

        print(f"  {split:5s}: {len(split_imgs)} images → {img_out}")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument("--source",       default="data/raw",
                        help="Source folder with images/ and labels/")
    parser.add_argument("--output",       default="data",
                        help="Output base folder (train/val/test will be created here)")
    parser.add_argument("--class-names",  nargs="+", default=["product"],
                        help="Space-separated list of class names (in class ID order)")
    parser.add_argument("--extract-frames", type=int, default=0,
                        help="If --source is a video, evenly extract this many frames first")
    parser.add_argument("--augment",      action="store_true",
                        help="Apply offline Albumentations augmentation to training set")
    parser.add_argument("--multiplier",   type=int, default=5,
                        help="Number of augmented copies per original image")
    parser.add_argument("--train-ratio",  type=float, default=0.70)
    parser.add_argument("--val-ratio",    type=float, default=0.20)
    args = parser.parse_args()

    source_dir = resolve_under_root(args.source)
    output_dir = resolve_under_root(args.output)

    print(f"[PrepareDataset] Source : {source_dir}")
    print(f"[PrepareDataset] Output : {output_dir}")
    print(f"[PrepareDataset] Classes: {args.class_names}")

    if source_dir.suffix.lower() in VIDEO_SUFFIXES:
        print("\n[Step 1] Extracting frames from video...")
        raw_dir = output_dir / "raw"
        extract_frames_from_video(source_dir, raw_dir, args.extract_frames)
        source_dir = raw_dir
    elif args.extract_frames:
        print("[ERROR] --extract-frames can only be used when --source is a video file.")
        sys.exit(1)

    # Step 2: split
    print("\n[Step 2] Splitting dataset...")
    split_dataset(source_dir, output_dir, args.train_ratio, args.val_ratio)

    # Step 3: augment (optional)
    if args.augment:
        print("\n[Step 3] Augmenting training set...")
        try:
            from src.dataset_utils import augment_and_save

            train_dir = output_dir / "train"
            aug_dir = output_dir / "train_augmented"
            augment_and_save(train_dir, aug_dir, multiplier=args.multiplier)

            # Merge augmented back into train
            for sub in ["images", "labels"]:
                for f in (aug_dir / sub).iterdir():
                    shutil.copy2(f, train_dir / sub / f.name)
            shutil.rmtree(aug_dir)
            print(f"  Augmented images merged into train/")
        except Exception as e:
            print(f"  [WARNING] Augmentation failed: {e}")
    else:
        print("\n[Step 3] Augmentation skipped (pass --augment to enable).")

    # Step 4: create data.yaml
    print("\n[Step 4] Generating data.yaml...")
    try:
        from src.dataset_utils import build_data_yaml

        yaml_path = build_data_yaml(output_dir, args.class_names)
    except Exception as e:
        # Fallback: write manually
        import yaml

        yaml_path = str(output_dir / "data.yaml")
        data = {
            "path": str(output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(args.class_names),
            "names": args.class_names,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"  data.yaml written to: {yaml_path}")

    print(f"\n[PrepareDataset] Done!")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Next:  python scripts/train_detector.py --data {yaml_path}")


if __name__ == "__main__":
    main()
