#!/bin/bash
# run_full_training.sh
# This script prepares the dataset (if needed) and trains the YOLOv8 detector
# for the full 150 epochs to get production-ready weights.

# Ensure we are in the source_code directory
cd "$(dirname "$0")/.." || exit

# Activate the virtual environment
source venv/bin/activate

echo "========================================"
echo " Starting Full Object Detector Training "
echo "========================================"

# Optionally re-prepare the dataset (you can comment this out if data/data.yaml already exists)
# python scripts/prepare_dataset.py \
#    --source data \
#    --output data \
#    --class-names "Blue Basket" "Blue Panel" "Brown Table" "Brown Tray" "Orange Cone" "Part"

# Run the training script for 150 epochs.
# We override the batch size here (set to 8 or 16 depending on your GPU memory).
# If you get a CUDA Out of Memory error, lower the --batch argument.
python scripts/train_detector.py \
    --data data/data.yaml \
    --epochs 150 \
    --batch 8

echo "========================================"
echo " Training Complete! "
echo "========================================"
echo "Don't forget to update config.yaml with your new best weights:"
echo "  detection:"
echo "    model_path: \"runs/detect/conveyor_belt/weights/best.pt\""
