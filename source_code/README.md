# Conveyor Belt Product Detection & Tracking System

**A real-time computer vision pipeline for automated product detection, multi-object tracking, counting, and defect inspection on manufacturing conveyor belts.**

> **Group:** Do Duy Loi (23120293) · Trinh Chan Duy (23120419) · Dang Vo Hong Phuc (23120155)  
> **Course:** Major Project — University of Science, Ho Chi Minh City, 2026

---

## 🏗️ Architecture Overview

```
Camera / Video File
      │
      ▼
Capture Thread ──────────────► frame_queue
                                    │
                                    ▼
Processing Thread  (YOLO + ByteTrack + Virtual Counting Zones)
                  ├─────────────► result_queue  → Main thread (cv2 display)
                  └─────────────► crop_queue
                                    │
                                    ▼
Classifier Thread  (MobileNetV3 Defect Inspection)
                  └─────────────► db_queue
                                    │
                                    ▼
DB Thread          (SQLite ingestion)
                                    │
                                    ▼
Streamlit Dashboard (live monitoring)
```

---

## 📁 Project Structure

```
source_code/
├── app.py                          # 🚀 Main application entry point
├── config.yaml                     # ⚙️  Central configuration file
├── requirements.txt
│
├── src/                            # Core Python modules
│   ├── dataset_utils.py            # Albumentations augmentation pipeline
│   ├── detection.py                # YOLO detector wrapper
│   ├── tracking.py                 # ByteTrack MOT (pure Python)
│   ├── counting.py                 # Virtual Counting Zones state machine
│   ├── defect_classifier.py        # MobileNetV3 defect classifier
│   ├── database.py                 # SQLite manager
│   └── utils/
│       ├── visualization.py        # Frame annotation drawing
│       └── video_utils.py          # VideoSource / ThreadedCapture / VideoWriter
│
├── scripts/
│   ├── generate_demo_video.py      # 🎬 Generate synthetic conveyor belt video
│   ├── prepare_dataset.py          # 📦 Prepare/split/augment YOLO dataset
│   ├── train_detector.py           # 🔬 Train YOLO detector
│   └── train_classifier.py         # 🔬 Train defect classifier
│
├── deployment/
│   ├── trt_export.py               # ⚡ Export to ONNX / TensorRT
│   └── requirements.txt
│
├── dashboard/
│   └── app.py                      # 📊 Streamlit monitoring dashboard
│
├── data/                           # Datasets & database (created at runtime)
│   ├── train/ val/ test/
│   └── conveyor.db
│
├── models/                         # Model weight storage
│   ├── detection/
│   └── classification/
│
└── plan/                           # Planning documents (see plan/README.md)
```

---

## ⚡ Quick Start (Demo Mode)

### 1. Set up virtual environment

```bash
cd source_code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate a synthetic demo video

```bash
python scripts/generate_demo_video.py --output demo/demo_conveyor.mp4 --duration 60
```

### 3. Run the main pipeline

```bash
python app.py --source demo/demo_conveyor.mp4
```

The system will:
- Auto-download `yolov8s.pt` from Ultralytics on first run
- Run detection + ByteTrack + Virtual Counting Zones
- Display annotated video with zone overlays, track IDs, and HUD
- Classify product crops in simulation mode (random defect assignment)
- Log all events to `data/conveyor.db`
- Record annotated output to `data/output_annotated.mp4`

Press **`q`** to quit, **`r`** to reset counters.

### 4. Launch the monitoring dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Configuration

Edit [`config.yaml`](config.yaml) to adjust:

| Section | Key | Description |
|---|---|---|
| `source` | `type` | `"file"`, `"webcam"`, `"rtsp"` |
| `source` | `path` | Path to video file |
| `detection` | `model_path` | Path to YOLO `.pt` / `.onnx` / `.engine` |
| `detection` | `conf_threshold` | Minimum detection confidence (0.35) |
| `counting` | `zone_start` | Entry/Tracking boundary (fraction of width, default 0.20) |
| `counting` | `zone_end` | Tracking/Exit boundary (fraction of width, default 0.80) |
| `classifier` | `simulate` | `true` = random predictions (demo), `false` = real model |
| `app` | `headless` | `true` = no display window (server mode) |
| `app` | `record_output` | `true` = save annotated MP4 |

---

## 🔬 Training Your Own Models

### Detector (YOLO)

```bash
# 1. Prepare your labelled dataset
python scripts/prepare_dataset.py \
    --source data/raw \
    --class-names product \
    --augment

# 2. Train
python scripts/train_detector.py \
    --model yolov11m.pt \
    --epochs 150

# 3. Update config.yaml with the best weights path
#    detection.model_path: runs/detect/conveyor_belt/weights/best.pt
```

### Defect Classifier (MobileNetV3)

```bash
# Organise your crops:
# data/classifier/train/Normal/     (*.jpg)
# data/classifier/train/Defective/  (*.jpg)
# data/classifier/val/Normal/
# data/classifier/val/Defective/

python scripts/train_classifier.py \
    --data-dir data/classifier \
    --epochs 30

# Update config.yaml:
#    classifier.model_path: models/classification/mobilenetv3_defect.pt
#    classifier.simulate: false
```

---

## ⚡ Edge Deployment (Jetson / GPU Server)

```bash
# Export to ONNX
python deployment/trt_export.py \
    --model runs/detect/conveyor_belt/weights/best.pt \
    --format onnx

# Export to TensorRT FP16
python deployment/trt_export.py \
    --model runs/detect/conveyor_belt/weights/best.pt \
    --format engine \
    --fp16 \
    --validate
```

Update `config.yaml` → `detection.model_path: models/detection/best.engine`

---

## 📊 Pipeline Modules

| Module | File | Description |
|---|---|---|
| **Detection** | `src/detection.py` | Ultralytics YOLO wrapper + ONNX fallback |
| **Tracking** | `src/tracking.py` | ByteTrack: Kalman Filter + 2-stage Hungarian matching |
| **Counting** | `src/counting.py` | Virtual Zone state machine (Entry→Tracking→Exit) |
| **Classifier** | `src/defect_classifier.py` | MobileNetV3-Small binary defect classifier |
| **Database** | `src/database.py` | Thread-safe SQLite with throughput logging |
| **Visualization** | `src/utils/visualization.py` | Zone overlays, track annotation, HUD |
| **Video** | `src/utils/video_utils.py` | VideoSource, ThreadedCapture, VideoWriter |

---

## 🧪 Technical Highlights

- **ByteTrack**: Two-stage matching (high + low confidence detections) prevents ID switches under occlusion with Kalman Filter prediction
- **Virtual Counting Zones**: Products must traverse Entry → Tracking → Exit in sequence; anti-double-count cache of 200 IDs
- **Optimal Viewpoint Crops**: Classifier triggered only when track centre is within 5% of Tracking zone midpoint (lens-distortion minimum)
- **Multi-threaded**: 4-thread consumer-producer pipeline with bounded queues to prevent frame drops at 60+ FPS
- **TensorRT FP16**: ~2× speedup over PyTorch on Jetson devices (target >60 FPS)

---

## 📚 References

1. Jocher, G., et al. (2023). *YOLOv8 by Ultralytics*. https://github.com/ultralytics/ultralytics
2. Zhang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
3. Wojke, N., et al. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric*. ICIP 2017.
4. MVTec Software GmbH. *D2S Dataset for Instance Segmentation*. https://www.mvtec.com/company/research/datasets/d2s/
5. Bergmann, P., et al. (2019). *MVTec AD Dataset for Anomaly Detection*. CVPR 2019.
