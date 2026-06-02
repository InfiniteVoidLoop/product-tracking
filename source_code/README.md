# Conveyor Belt Product Detection & Tracking System

**Real-time CV pipeline for product detection, tracking, counting, and defect inspection on conveyor belts.**

> **Authors:** Do Duy Loi (23120293) · Trinh Chan Duy (23120419) · Dang Vo Hong Phuc (23120155)
> **Course:** Major Project — University of Science, Ho Chi Minh City, 2026

---

## Project Structure

```
source_code/
├── app.py                      # Main pipeline application
├── config.yaml                 # Central configuration
├── requirements.txt
│
├── src/                        # Core modules
│   ├── detection.py            # YOLO wrapper
│   ├── tracking.py             # ByteTrack (pure Python)
│   ├── counting.py             # Virtual Counting Zones
│   ├── defect_classifier.py    # MobileNetV3 defect classifier
│   ├── database.py             # SQLite logging
│   ├── dataset_utils.py        # Augmentation pipeline
│   └── utils/
│       ├── visualization.py    # Drawing: zones, tracks, HUD
│       └── video_utils.py      # VideoSource, ThreadedCapture, VideoWriter
│
├── scripts/
│   ├── train_detector.py       # Train YOLO detector
│   ├── train_classifier.py     # Train defect classifier
│   ├── run_inference.py        # Run detection on a video
│   ├── prepare_dataset.py      # Split & augment dataset
│   └── generate_demo_video.py  # Generate synthetic test video
│
├── dashboard/app.py            # Streamlit monitoring dashboard
├── deployment/trt_export.py    # ONNX / TensorRT export
│
├── data/
│   ├── train/ val/ test/       # ABO dataset (YOLO format, not in git)
│   ├── conveyor_dataset/       # 35 labelled real conveyor frames
│   │   ├── train/ val/ test/
│   │   ├── raw/                # Raw unlabelled frames
│   │   └── data.yaml
│   └── data.yaml
│
├── demo/
│   ├── demo-conveyor.mp4           # Input demo video
│   └── demo-conveyor-detected.mp4  # Output after inference
│
└── models/
    └── detection/yolov8s.pt    # Pretrained YOLOv8s weights
```

---

## Setup

```bash
cd source_code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data Download

Dataset files are not tracked by git. Download before training.

### ABO Product Dataset (278k images — for detector training)

```bash
# Download (~3.1 GB)
wget https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/archives/abo-images-small.tar -P data/

# Extract
mkdir -p temp_images
tar -xf data/abo-images-small.tar -C temp_images/

# Organise into YOLO splits
python scripts/prepare_dataset.py \
    --source temp_images/images/small \
    --output data \
    --class-names product

# Cleanup
rm -f data/abo-images-small.tar && rm -rf temp_images/
```

### Conveyor-specific dataset (35 labelled frames — already included)

The `data/conveyor_dataset/` folder is already structured and ready to use.

---

## Workflow: Train → Test on Demo Video

### Step 1 — Train the Object Detector

```bash
python scripts/train_detector.py \
    --data  data/conveyor_dataset/data.yaml \
    --model yolov8s.pt \
    --epochs 20 \
    --batch  16 \
    --device 0
```

> After training completes, move the best weights into `models/`:
> ```bash
> cp runs/detect/conveyor_belt/weights/best.pt models/detection/conveyor_best.pt
> rm -rf runs/
> ```
> Then update `config.yaml` → `detection.model_path: "models/detection/conveyor_best.pt"`

---

### Step 2 — Train the Defect Classifier

**2a. Generate the classifier dataset** (auto-builds Normal + Defective pairs from ABO images):

```bash
python scripts/build_classifier_dataset.py --n-train 2000 --n-val 400
```

This creates:
```
data/classifier/
├── train/
│   ├── Normal/     2000 clean product images
│   └── Defective/  2000 synthetically degraded images
└── val/
    ├── Normal/      400 images
    └── Defective/   400 images
```

**2b. Train the classifier:**

```bash
python scripts/train_classifier.py \
    --data-dir data/classifier \
    --epochs   10 \
    --batch    32
```

> Weights saved to: `models/classification/mobilenetv3_defect.pt`

---

### Step 3 — Test on the Demo Video

Run inference with both trained models:

```bash
python scripts/run_inference.py \
    --model  models/detection/conveyor_best.pt \
    --source demo/demo-conveyor.mp4 \
    --output demo/demo-conveyor-detected.mp4 \
    --conf   0.25
```

The annotated video is saved to `demo/demo-conveyor-detected.mp4`.

To also activate the defect classifier, update `config.yaml` first:

```yaml
classifier:
  simulate: false
  model_path: "models/classification/mobilenetv3_defect.pt"
```

Then run the full live pipeline instead:

```bash
python app.py --source demo/demo-conveyor.mp4
```

Controls: `q` = quit · `r` = reset counters

---

## Live Pipeline (Camera / RTSP)

```bash
# Webcam
python app.py --source 0

# RTSP stream
python app.py --source rtsp://192.168.1.100/live

# Headless (no display, for server / SSH)
python app.py --source demo/demo-conveyor.mp4 --headless

# Disable output recording
python app.py --no-record
```

---

## Monitoring Dashboard

```bash
# In a separate terminal
streamlit run dashboard/app.py
# → Open http://localhost:8501
```

Shows: live counts · throughput chart · defect rate trend · alert log

---

## Configuration Reference

| Section | Key | Default | Description |
|---|---|---|---|
| `source` | `path` | `demo/demo-conveyor.mp4` | Input video path |
| `detection` | `model_path` | `models/detection/yolov8s.pt` | YOLO weights |
| `detection` | `conf_threshold` | `0.35` | Detection confidence |
| `counting` | `zone_start` | `0.20` | Entry zone (fraction of width) |
| `counting` | `zone_end` | `0.80` | Exit zone (fraction of width) |
| `classifier` | `simulate` | `true` | `false` once classifier is trained |
| `app` | `headless` | `false` | Skip display window |
| `app` | `record_output` | `true` | Save annotated video |

---

## Edge Deployment (TensorRT)

```bash
python deployment/trt_export.py \
    --model  runs/detect/conveyor_belt/weights/best.pt \
    --format engine \
    --fp16
```

Then set `config.yaml` → `detection.model_path: "models/detection/best.engine"`

---

## Pipeline Architecture

```
Video Source
    │
    ▼
[Capture Thread]      frame_queue
    │
    ▼
[Processing Thread]   YOLO → ByteTrack → Zone Counter
    ├──────────────▶  [Main Thread]  display + VideoWriter
    └──────────────▶  crop_queue
                          │
                          ▼
                  [Classifier Thread]  MobileNetV3 → db_queue
                                           │
                                           ▼
                                   [DB Thread]  SQLite
```

---

## References

1. Jocher, G., et al. — *YOLOv8 by Ultralytics* (2023)
2. Zhang, Y., et al. — *ByteTrack*, ECCV 2022
3. Bergmann, P., et al. — *MVTec AD*, CVPR 2019
4. Collins, J., et al. — *ABO Dataset*, CVPR 2022
