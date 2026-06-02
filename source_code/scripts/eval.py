"""
scripts/eval.py
===============
Benchmarking tool for the Conveyor Belt CV System.
Calculates Multi-Object Tracking Accuracy (MOTA) and Counting Accuracy against ground truth.

Usage:
    python scripts/eval.py --ground-truth data/ground_truth.json --predictions data/predictions.json
"""

import argparse
import json
import sys

def calculate_mota(ground_truth, predictions):
    """
    Calculate Multi-Object Tracking Accuracy (MOTA).
    MOTA = 1 - (sum(misses) + sum(false_positives) + sum(mismatches)) / sum(ground_truth_objects)
    """
    misses = 0
    fps = 0
    mismatches = 0
    gt_count = 0

    # Simplified mock calculation since we don't have bounding box IoU matching implemented here.
    # In a real scenario, we would compute IoU between GT and Predictions per frame
    # and use the Hungarian algorithm to match identities.
    
    # We will simulate the metrics based on provided json data if it contains the counts
    
    # For now, let's assume ground_truth and predictions contain frame-level stats
    # or just use dummy values if data is not structured.
    if "frames" in ground_truth and "frames" in predictions:
        for frame_idx, gt_frame in ground_truth["frames"].items():
            pred_frame = predictions["frames"].get(frame_idx, {})
            gt_objects = gt_frame.get("objects", [])
            pred_objects = pred_frame.get("objects", [])
            
            gt_count += len(gt_objects)
            # Mock calculation:
            diff = len(pred_objects) - len(gt_objects)
            if diff > 0:
                fps += diff
            elif diff < 0:
                misses += abs(diff)
            
            mismatches += pred_frame.get("id_switches", 0)

    if gt_count == 0:
        return 0.0
        
    mota = 1.0 - (misses + fps + mismatches) / gt_count
    return mota

def calculate_counting_accuracy(gt_count, pred_count):
    """
    Calculate Counting Accuracy.
    Accuracy = 1 - abs(predicted_count - ground_truth_count) / ground_truth_count
    """
    if gt_count == 0:
        return 1.0 if pred_count == 0 else 0.0
    return max(0.0, 1.0 - abs(pred_count - gt_count) / gt_count)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tracking and Counting KPI")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON file")
    
    args = parser.parse_args()
    
    try:
        with open(args.ground_truth, 'r') as f:
            gt_data = json.load(f)
        with open(args.predictions, 'r') as f:
            pred_data = json.load(f)
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load JSON data: {e}")
        sys.exit(1)
        
    # Extract total counts
    gt_total = gt_data.get("total_count", 0)
    pred_total = pred_data.get("total_count", 0)
    
    mota = calculate_mota(gt_data, pred_data)
    counting_acc = calculate_counting_accuracy(gt_total, pred_total)
    
    print("====================================")
    print("        Evaluation Results          ")
    print("====================================")
    print(f"Ground Truth Total Count: {gt_total}")
    print(f"Predicted Total Count   : {pred_total}")
    print(f"Counting Accuracy       : {counting_acc:.2%}")
    print(f"MOTA                    : {mota:.2%}")
    print("====================================")

if __name__ == "__main__":
    main()
