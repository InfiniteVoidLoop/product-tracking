#!/bin/bash
# run_demo.sh
# This script runs the full tracking and counting pipeline on a video file.

# Ensure we are in the source_code directory
cd "$(dirname "$0")/.." || exit

# Activate the virtual environment
source venv/bin/activate

echo "==================================================="
echo " Running Conveyor Belt Tracking Demo "
echo "==================================================="

# To see the processing happen live in a pop-up window (if your system supports GUI):
# python app.py --source data/videos/v1.mov

# If you just want it to process in the background and save the annotated output video
# without popping up a window, we use the --headless flag:
echo "Processing v1.mov in headless mode..."
python app.py --source data/videos/v1.mov --headless
mv data/output_annotated.mp4 data/v1_tracked_demo.mp4
echo "Saved annotated output to data/v1_tracked_demo.mp4!"

echo ""
echo "Processing v2.mov in headless mode..."
python app.py --source data/videos/v2.mov --headless
mv data/output_annotated.mp4 data/v2_tracked_demo.mp4
echo "Saved annotated output to data/v2_tracked_demo.mp4!"

echo "==================================================="
echo " All done! "
echo "==================================================="
