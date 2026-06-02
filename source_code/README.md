# Conveyor Belt Product Detection & Tracking System

**Real-time computer vision pipeline for automated product detection, multi-object tracking, counting, and defect inspection on manufacturing conveyor belts.**

> **Authors:** Do Duy Loi (23120293) · Trinh Chan Duy (23120419) · Dang Vo Hong Phuc (23120155)
> **Course:** Major Project — University of Science, Ho Chi Minh City, 2026

---

## Project Structure

```
source_code/
├── app.py                          # Main application entry point
├── config.yaml                     # Central configuration (edit this first)
├── requirements.txt                # Python dependencies
├── README.md
│
├── src/                            # Core pipeline modules
│   ├── detection.py                # YOLO wrapper (Ultralytics + ONNX)
│   ├── tracking.py                 # ByteTrack multi-object tracker (pure Python)
│   ├── counting.py                 # Virtual Counting Zones state machine
│   ├── defect_classifier.py        # MobileNetV3 defect classifier
│   ├── database.py                 # Thread-safe SQLite logging
│   ├── dataset_utils.py            # Albumentations augmentation pipeline
│   └── utils/
│       ├── visualization.py        # Annotation drawing (zones, tracks, HUD)
│       └── video_utils.py          # VideoSource, ThreadedCapture, VideoWriter
│
├── scripts/
│   ├── prepare_dataset.py          # Split & augment a YOLO dataset
│   ├── train_detector.py           # Train YOLO object detector
│   ├── train_classifier.py         # Train MobileNetV3 defect classifier
│   ├── run_inference.py            # Run detection on a video file
│   └── generate_demo_video.py      # Generate a synthetic test video
│
├── dashboard/
│   └── app.py                      # Streamlit live monitoring dashboard
│
├── deployment/
│   ├── trt_export.py               # Export to ONNX / TensorRT
│   └── requirements.txt
│
├── data/
│   ├── train/ val/ test/           # ABO product image dataset (YOLO format)
│   ├── conveyor_dataset/           # Conveyor-specific labelled dataset (35 imgs)
│   │   ├── train/ val/ test/
│   │   ├── raw/                    # Raw unlabelled frames from conveyor videos
│   │   └── data.yaml
│   └── data.yaml                   # Main dataset config
│
├── demo/
│   ├── demo-conveyor.mp4           # Synthetic conveyor demo video
│   ├── conveyor_video.mp4          # Real conveyor footage (62 MB)
│   └── conveyor_nuts_2.mp4         # Real conveyor footage (nuts)
│
├── models/
│   └── detection/
│       └── yolov8s.pt              # Pretrained YOLOv8s weights
│
└── plan/                           # Project planning documents
```

---

## Data Setup

Dataset files are **not tracked by git** (too large). Follow the steps below to download and place them correctly.

---

### Option A — ABO Product Image Dataset (278k images, recommended for detector pre-training)

The project uses the [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) small-image dataset.

```bash
cd source_code

# Download the archive (~3.1 GB)
wget https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/archives/abo-images-small.tar -P data/

# Extract into a temporary folder
mkdir -p temp_images
tar -xf data/abo-images-small.tar -C temp_images/

# Organise into train / val / test splits with YOLO labels
python scripts/prepare_dataset.py \
    --source temp_images/images/small \
    --output data \
    --class-names product

# Clean up the archive and temp folder
rm -f data/abo-images-small.tar
rm -rf temp_images/
```

After this you will have:
```
data/
├── train/images/   (~278k .jpg)
├── train/labels/   (~278k .txt)
├── val/images/     (~79k  .jpg)
├── val/labels/     (~79k  .txt)
├── test/images/    (~39k  .jpg)
├── test/labels/    (~39k  .txt)
└── data.yaml
```

---

### Option B — Conveyor-Specific Dataset (35 labelled real frames, for fine-tuning)

These frames are extracted from the conveyor videos included in `demo/`.

```bash
cd source_code

# Extract frames from the real conveyor video
python scripts/prepare_dataset.py \
    --source data/conveyor_dataset/raw \
    --output data/conveyor_dataset \
    --class-names product

# The labelled split is now ready at:
# data/conveyor_dataset/train/  (24 images)
# data/conveyor_dataset/val/    (7 images)
# data/conveyor_dataset/test/   (4 images)
```

> **Tip:** Use Option A to pre-train the detector on a large variety of products, then fine-tune on Option B with real conveyor footage for the best accuracy.

---

### Skipping Data Download (demo / inference only)

If you only want to run inference on the demo video without training, **no dataset download is needed**. The pretrained `models/detection/yolov8s.pt` is sufficient.

---

## Quick Start

### 1 — Set up the Python environment

```bash
cd source_code

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2 — Run detection on the demo video (no training needed)

Uses pretrained `yolov8s.pt` in simulate mode (no custom weights required):

```bash
python scripts/run_inference.py \
    --model models/detection/yolov8s.pt \
    --source demo/demo-conveyor.mp4 \
    --output demo/demo-conveyor-detected.mp4 \
    --conf 0.25
```

The annotated video is saved to `demo/demo-conveyor-detected.mp4`.

### 3 — Run the full live pipeline

Opens an OpenCV window with zone overlays, tracking IDs, defect flags, and a HUD counter:

```bash
python app.py
```

**Key controls:**
- `q` — quit
- `r` — reset counters

**Common overrides:**

```bash
# Use a specific video file
python app.py --source demo/conveyor_video.mp4

# Webcam
python app.py --source 0

# RTSP stream
python app.py --source rtsp://192.168.1.100/live

# Headless (no display window, for SSH / server)
python app.py --headless

# Override confidence threshold
python app.py --conf 0.4

# Do not record output video
python app.py --no-record
```

### 4 — Launch the monitoring dashboard

```bash
# In a separate terminal (venv must be active)
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) — the dashboard auto-refreshes and shows:
- Live product count (Total / Normal / Defective)
- Throughput-per-minute bar chart
- Defect rate trend line
- Alert log and product table

---

## Training Your Own Detector

### Step 1 — Prepare your dataset

Your labelled images should be in YOLO format (`images/` + `labels/` folders).

```bash
python scripts/prepare_dataset.py \
    --source data/raw           \    # folder with images/ and labels/
    --output data/my_dataset    \
    --class-names product       \
    --augment                         # optional: Albumentations augmentation
```

Or use the existing `data/conveyor_dataset/` (35 labelled real conveyor frames).

### Step 2 — Train the detector

```bash
python scripts/train_detector.py \
    --data data/conveyor_dataset/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 16 \
    --device 0          # GPU index; use "cpu" if no GPU
```

Best weights are saved to `runs/detect/conveyor_belt/weights/best.pt`.

### Step 3 — Update config

Edit `config.yaml`:
```yaml
detection:
  model_path: "runs/detect/conveyor_belt/weights/best.pt"
```

---

## Training the Defect Classifier

Organise crop images into class folders:

```
data/classifier/
├── train/
│   ├── Normal/       ← normal product crops
│   └── Defective/    ← defective product crops
└── val/
    ├── Normal/
    └── Defective/
```

Then train:

```bash
python scripts/train_classifier.py \
    --data-dir data/classifier \
    --epochs 30 \
    --batch 32
```

Weights saved to `models/classification/mobilenetv3_defect.pt`.

Update `config.yaml`:
```yaml
classifier:
  simulate: false
  model_path: "models/classification/mobilenetv3_defect.pt"
```

---

## Edge Deployment (Jetson / GPU Server)

Export to TensorRT FP16 for ~2× speed improvement:

```bash
# Export to ONNX first
python deployment/trt_export.py \
    --model runs/detect/conveyor_belt/weights/best.pt \
    --format onnx

# Export to TensorRT engine
python deployment/trt_export.py \
    --model runs/detect/conveyor_belt/weights/best.pt \
    --format engine \
    --fp16 \
    --validate
```

Update `config.yaml` → `detection.model_path: "models/detection/best.engine"`

---

## Configuration Reference (`config.yaml`)

| Section | Key | Default | Description |
|---|---|---|---|
| `source` | `type` | `"file"` | `"file"`, `"webcam"`, or `"rtsp"` |
| `source` | `path` | `"demo/demo-conveyor.mp4"` | Video file path |
| `detection` | `model_path` | `"models/detection/yolov8s.pt"` | YOLO weights (.pt / .onnx / .engine) |
| `detection` | `conf_threshold` | `0.35` | Minimum detection confidence |
| `counting` | `zone_start` | `0.20` | Entry zone boundary (fraction of width) |
| `counting` | `zone_end` | `0.80` | Exit zone boundary (fraction of width) |
| `classifier` | `simulate` | `true` | `true` = random predictions (demo mode) |
| `app` | `headless` | `false` | Skip display window |
| `app` | `record_output` | `true` | Save annotated video |

---

## Pipeline Architecture

```
Video Source (file / webcam / RTSP)
        │
        ▼
[Thread 1: Capture]       VideoSource → frame_queue
        │
        ▼
[Thread 2: Processing]    YOLO detect → ByteTrack → Zone State Machine
        ├── result_queue → [Main Thread] → cv2 display + VideoWriter
        └── crop_queue
                │
                ▼
[Thread 3: Classifier]    MobileNetV3 → defect result → db_queue
                │
                ▼
[Thread 4: Database]      SQLite inserts (products / alerts / throughput)
```

All threads share a `threading.Event` stop signal and communicate through bounded `queue.Queue` instances with non-blocking drops to prevent back-pressure.

---

## References

1. Jocher, G., et al. — *YOLOv8 by Ultralytics* (2023)
2. Zhang, Y., et al. — *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022
3. Bergmann, P., et al. — *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019
4. Collins, J., et al. — *ABO: Dataset and Benchmarks for Real-World 3D Object Understanding*, CVPR 2022
