#!/bin/bash
# run_classifier_training.sh
# This script trains the MobileNetV3 image classifier to detect "Normal" vs "Defective" items.

# Ensure we are in the source_code directory
cd "$(dirname "$0")/.." || exit

# Activate the virtual environment
source venv/bin/activate

echo "================================================="
echo " Starting Defect Classifier Training (MobileNet) "
echo "================================================="
echo "Make sure your dataset is correctly organized in:"
echo "  data/classification/train/Normal/"
echo "  data/classification/train/Defective/"
echo "  data/classification/val/Normal/"
echo "  data/classification/val/Defective/"
echo ""
echo "Training for 30 epochs with batch size 32..."

# Run the classifier training script
python scripts/train_classifier.py \
  --data-dir data/classification \
  --epochs 30 \
  --batch 32 \
  --lr 0.001

echo "================================================="
echo " Training Complete! "
echo "================================================="
echo "The new weights have been automatically saved to:"
echo "  models/classification/mobilenetv3_defect.pt"
echo "Your app.py is already configured to use these new weights!"
