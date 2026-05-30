"""
scripts/train_classifier.py
===========================
Trains a MobileNetV3-Small defect classifier (Normal vs Defective).

Dataset format expected:
    data/classifier/
        train/
            Normal/     *.jpg | *.png
            Defective/  *.jpg | *.png
        val/
            Normal/
            Defective/

Usage:
    python scripts/train_classifier.py [--config config.yaml]
                                       [--data-dir data/classifier]
                                       [--epochs 30]
                                       [--batch 32]
                                       [--device cuda:0]

The trained weights are saved to models/classification/mobilenetv3_defect.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3 defect classifier")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--data-dir", default="data/classifier",
                        help="Dataset root containing train/ and val/ folders")
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--device",   default=None)
    parser.add_argument("--output",   default="models/classification/mobilenetv3_defect.pt")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError:
        print("[ERROR] PyTorch + torchvision required. Install with: pip install torch torchvision")
        sys.exit(1)

    data_dir = ROOT / args.data_dir
    if not data_dir.exists():
        print(
            f"[ERROR] Data directory not found: {data_dir}\n"
            "  Expected structure:\n"
            "    data/classifier/train/Normal/\n"
            "    data/classifier/train/Defective/\n"
            "    data/classifier/val/Normal/\n"
            "    data/classifier/val/Defective/"
        )
        sys.exit(1)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[TrainClassifier] Using device: {device}")

    # ---------------------------------------------------------------------------
    # Data transforms
    # ---------------------------------------------------------------------------
    input_size = 224
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    datasets_dict = {
        split: datasets.ImageFolder(str(data_dir / split), data_transforms[split])
        for split in ["train", "val"]
    }
    loaders = {
        split: DataLoader(
            ds, batch_size=args.batch, shuffle=(split == "train"),
            num_workers=4, pin_memory=True
        )
        for split, ds in datasets_dict.items()
    }

    class_names = datasets_dict["train"].classes
    num_classes = len(class_names)
    print(f"[TrainClassifier] Classes: {class_names} (n={num_classes})")
    print(f"  Train: {len(datasets_dict['train'])} | Val: {len(datasets_dict['val'])}")

    # ---------------------------------------------------------------------------
    # Model: MobileNetV3-Small with custom head
    # ---------------------------------------------------------------------------
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    # Replace classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    best_val_acc = 0.0
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_correct = 0
            total = 0

            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
                total           += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = running_correct / total

            print(
                f"  Epoch {epoch:>3}/{args.epochs} [{phase}]  "
                f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}"
            )

            if phase == "val":
                scheduler.step()
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save(model.state_dict(), str(output_path))
                    print(f"    ★ New best model saved (val_acc={best_val_acc:.4f})")

    print(f"\n[TrainClassifier] Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"  Weights saved to: {output_path}")
    print(f"\n  Update config.yaml → classifier.model_path and set simulate: false")


if __name__ == "__main__":
    main()
