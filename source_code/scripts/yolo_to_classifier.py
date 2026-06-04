import os
import cv2
from pathlib import Path

def convert_yolo_to_classifier(yolo_dir="data", output_dir="data/classifier", classes_txt="data/classes.txt"):
    yolo_path = Path(yolo_dir)
    out_path = Path(output_dir)
    
    # Load class names
    classes_file = yolo_path / "classes.txt"
    if not classes_file.exists():
        classes_file = Path(classes_txt)
        
    if not classes_file.exists():
        print(f"[ERROR] Could not find classes file at {classes_file}")
        return
        
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    for split in ["train", "val", "test"]:
        split_img_dir = yolo_path / split / "images"
        split_lbl_dir = yolo_path / split / "labels"
        
        if not split_img_dir.exists():
            continue
            
        print(f"Processing split: {split}...")
        
        for img_path in split_img_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
                
            lbl_path = split_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w, _ = img.shape
            
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # Convert normalized YOLO coordinates to absolute pixel coordinates
                x_center = cx * w
                y_center = cy * h
                width = bw * w
                height = bh * h
                
                x1 = int(max(0, x_center - width / 2))
                y1 = int(max(0, y_center - height / 2))
                x2 = int(min(w, x_center + width / 2))
                y2 = int(min(h, y_center + height / 2))
                
                # Ensure valid crop
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = img[y1:y2, x1:x2]
                
                # Ensure the folder for this class exists
                cls_name = class_names[cls_id]
                cls_out_dir = out_path / split / cls_name
                cls_out_dir.mkdir(parents=True, exist_ok=True)
                
                crop_out_path = cls_out_dir / f"{img_path.stem}_{idx}.jpg"
                cv2.imwrite(str(crop_out_path), crop)
                
    print(f"\n[Success] Dataset successfully extracted to {out_path}")
    print("You can now train the classifier using:")
    print("  python scripts/train_classifier.py --data-dir data/classifier")

if __name__ == "__main__":
    convert_yolo_to_classifier()
