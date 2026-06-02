# Conveyor Belt CV System Analysis & Remediation Plan

## Part 1 — Codebase Review

### 1. High-Level Architecture
The system is a multi-threaded Python application designed for real-time edge processing. It divides tasks across four background daemon threads to prevent I/O blocking and maintain high FPS:
1. **Capture Thread:** Reads video frames into a queue.
2. **Processing Thread:** Runs YOLO detection, custom ByteTrack tracking, and Virtual Zone counting.
3. **Classifier Thread:** Runs MobileNetV3 defect classification on cropped product images.
4. **Database Thread:** Writes aggregated counts and alerts to SQLite.
The main thread handles the synchronization and OpenCV visualization loop.

### 2. Processing Pipeline
- **Video Input:** Handled by `VideoSource` and `ThreadedCapture`, supporting file, webcam, and RTSP.
- **Detection:** Frame is passed to a unified `Detector` wrapper (YOLOv8 or ONNX).
- **Tracking:** Detections are fed into `ByteTracker`, which uses a two-stage Hungarian matching algorithm with a Kalman Filter (predicting `[cx, cy, a, h]`).
- **Counting:** `ZoneCounter` assigns states (`ENTRY` → `TRACKING` → `EXIT`) based on normalized axis coordinates. Valid traversals increment counts and trigger optimal-viewpoint crops.
- **Output:** Crops are classified; results are visualized via HUD overlays and logged to SQLite.

### 3. Main Modules
- `app.py`: Orchestrator and UI visualization loop.
- `src/detection.py`: Wrapper for Ultralytics YOLO and ONNX inference.
- `src/tracking.py`: Pure-Python ByteTrack multi-object tracker implementation.
- `src/counting.py`: Virtual counting zones state machine and logic.
- `src/defect_classifier.py`: Torchvision-based MobileNetV3 classifier.
- `src/utils/`: Handlers for video I/O and frame annotations.

### 4. Key Algorithms, Models, and Libraries
- **Detection:** Ultralytics YOLOv8 architecture (COCO pre-trained or custom).
- **Tracking:** Kalman Filter (constant velocity) + Hungarian algorithm (IoU bipartite matching) based on ByteTrack.
- **Classification:** MobileNetV3-Small.
- **Libraries:** OpenCV, PyTorch, NumPy, SciPy (for `linear_sum_assignment`), ONNX Runtime.

### 5. Technical Debt and Code Quality
- **Silent Fallbacks:** `src/detection.py` silently falls back to a generic `yolov8s.pt` model if the specified weights path doesn't exist, heavily impacting system correctness without throwing an error.
- **Rigid Tracking Assumptions:** The custom `KalmanFilter` in `src/tracking.py` expects pedestrian-like movement (constant aspect ratio).
- **Camera Coupling:** `ZoneCounter` relies strictly on horizontal (`x`) or vertical (`y`) straight lines. Slight camera rotations or diagonal conveyors will break the state machine.

---

## Part 2 — Root Cause Analysis

### 1. Objects not detected at all / Detects workers' hands
* **Symptom:** The system misses actual products and falsely flags body parts.
* **Relevant Files:** `src/detection.py` (lines 130-134), `config.yaml`
* **Root Cause:** The `config.yaml` specifies `models/detection/conveyor_best.pt`. If this custom model file is missing, `src/detection.py` silently catches the error and auto-downloads/loads the generic COCO `yolov8s.pt`. The COCO model doesn't know about specific factory products, but it does confidently detect "person" (workers' hands/arms).
* **Confidence Level:** 100% (High)
* **Proposed Fix:** Remove the silent fallback in `src/detection.py`. Throw an explicit `FileNotFoundError`. If a custom model is required, ensure it is trained and deployed. Furthermore, set the `classes` filter in `config.yaml` to strictly include only the desired product class IDs.

### 2. Conveyor runs vertically, but horizontal movement assumed
* **Symptom:** Vertical movement does not register zone transitions.
* **Relevant Files:** `config.yaml`, `src/counting.py`
* **Root Cause:** In `config.yaml`, the `counting.axis` parameter defaults to `"x"`. In `src/counting.py`, this extracts horizontal normalized coordinates. For a vertical conveyor, the horizontal coordinate changes very little, meaning objects never traverse the Entry → Tracking → Exit zones.
* **Confidence Level:** 100% (High)
* **Proposed Fix:** Change `counting.axis: "y"` in `config.yaml`. Set `direction` to `"positive"` (top-to-bottom) or `"negative"` (bottom-to-top) matching the physical setup.

### 3. Detection heavily dependent on direction / Strict Tracking Heuristics
* **Symptom:** System struggles with rotating objects or slight backward jitter.
* **Relevant Files:** `src/tracking.py` (lines 73-80, 244-249), `src/counting.py` (lines 263-272)
* **Root Cause:** 
  1. **Tracking:** The `KalmanFilter` models bounding boxes using `[cx, cy, a, h]` (where `a` = width/height). This is optimized for upright pedestrians. As products rotate on a belt, their bounding box aspect ratio `a` flips wildly. The Kalman Filter expects low variance on `a` (`1e-2`), causing predictions to heavily deviate from measurements and resulting in dropped tracks / ID switches.
  2. **Counting:** The `_direction_ok` function in `src/counting.py` strictly mandates monotonic movement (`delta >= -0.10`). Rotations causing the bounding box center to shift backward slightly can trigger track termination.
* **Confidence Level:** High
* **Proposed Fix:** 
  - Change the Kalman state vector in `tracking.py` from `[cx, cy, a, h]` to `[cx, cy, w, h]` (independent width and height) OR dramatically increase the process/measurement noise covariance for `a` and `h` to allow for rapid bounding box resizing due to object rotation.
  - Relax the `_direction_ok` delta threshold in `counting.py`, or evaluate direction based on smoothed velocity rather than comparing strictly to `first_seen_coord`.

---

## Part 3 — Fix Plan

### Critical Fixes
1. **Fix Silent Model Fallback**
   * *Description:* Modify `src/detection.py` to raise an exception if the custom weights file is missing, rather than loading COCO `yolov8s.pt`. Ensure custom weights exist.
   * *Impact:* Immediate resolution of false positives (hands) and missed products.
   * *Complexity:* Low
   * *Dependencies:* None
   * *Effort:* 1 hour
2. **Configure Vertical Axis Mapping**
   * *Description:* Update `config.yaml` to set `counting.axis: "y"` and configure the correct conveyor direction.
   * *Impact:* Zone logic will correctly process the vertical video feed.
   * *Complexity:* Low
   * *Dependencies:* None
   * *Effort:* 30 mins

### Important Improvements
3. **Adapt Kalman Filter for Rotations**
   * *Description:* Refactor `KalmanFilter` in `src/tracking.py` to model `[cx, cy, w, h]` instead of aspect ratio `a`. Update transition matrices and covariance values accordingly.
   * *Impact:* Substantial reduction in ID switches and dropped tracks for rotating products.
   * *Complexity:* Medium
   * *Dependencies:* None
   * *Effort:* 3-4 hours
4. **Relax Counting Direction Constraints**
   * *Description:* Update `_direction_ok` in `src/counting.py` to tolerate higher coordinate jitter caused by bounding box reshaping during rotation.
   * *Impact:* Prevents false track terminations.
   * *Complexity:* Low
   * *Dependencies:* Fix #3
   * *Effort:* 1 hour

### Nice-to-Have Enhancements
5. **TensorRT Optimization**
   * *Description:* Export the custom YOLO model to TensorRT (`.engine`) as outlined in the documentation to ensure stable 60+ FPS inference.
   * *Impact:* Reduces latency and improves tracker temporal consistency.
   * *Complexity:* High
   * *Dependencies:* Custom YOLO weights finalized.
   * *Effort:* 1-2 days

---

## Part 4 — Parallel Work Plan (Team of 3)

### Engineer A: Detection & Model Improvements
* **Responsibilities:** Train/verify the custom YOLO model for products. Ensure workers' hands are completely ignored. Export models to optimized formats (ONNX/TensorRT). Fix `src/detection.py` fallback logic.
* **Deliverables:** Finalized `conveyor_best.pt`, updated `src/detection.py`.
* **Parallelizable Tasks:** Model training, code updates in `detection.py`.
* **Dependencies:** Requires annotated dataset from Engineer C.

### Engineer B: Tracking & Conveyor-Motion Logic
* **Responsibilities:** Update `config.yaml` to handle vertical axes. Completely rewrite `src/tracking.py` (Kalman Filter) to handle rotating bounding boxes `[cx, cy, w, h]`. Update `src/counting.py` to handle directional jitter smoothly.
* **Deliverables:** Updated `config.yaml`, `tracking.py`, and `counting.py`.
* **Parallelizable Tasks:** 100% independent. Can use dummy data or existing flawed detections to validate tracking logic robustness.
* **Dependencies:** None.

### Engineer C: Evaluation, Testing & Tooling
* **Responsibilities:** Label raw video data for Engineer A. Write a test suite/script (`eval.py`) that calculates Multi-Object Tracking Accuracy (MOTA) and Counting Accuracy against ground truth.
* **Deliverables:** Annotated YOLO dataset, ground-truth counts for test videos, `eval.py` benchmarking tool.
* **Parallelizable Tasks:** 100% independent.
* **Integration Points:** Engineer A needs the dataset immediately. Engineer B will use `eval.py` to measure if the new Kalman Filter actually reduced ID switches.

---

## Recommended Implementation Order
1. **Eng A & B & C:** Kickoff. Eng C starts annotating a small validation set immediately.
2. **Eng A:** Patches `src/detection.py` to disable COCO fallback.
3. **Eng B:** Changes `config.yaml` to `axis: "y"` and starts rewriting the Kalman Filter.
4. **Eng C:** Delivers initial dataset to Eng A. Starts writing `eval.py`.
5. **Eng A:** Trains custom YOLO weights.
6. **Eng B:** Finishes tracking fixes and `_direction_ok` patches.
7. **Eng C:** Runs `eval.py` against Eng A's model and Eng B's tracker to verify KPIs.
8. **Eng A:** Explores TensorRT optimization for production deployment.

## Top 5 Fixes for Immediate Performance Improvement
1. **Disable silent fallback to COCO** (`src/detection.py:130`) — Stop detecting "person"/hands.
2. **Switch axis to "y"** (`config.yaml:46`) — Fix zone state machine for vertical video.
3. **Refactor Kalman Filter to [cx, cy, w, h]** (`src/tracking.py`) — Prevent ID switches from aspect ratio inversion during object rotation.
4. **Increase covariance noise for bounding box dimensions** (`src/tracking.py:73`) — Tell the Kalman Filter to trust the detector's box size over its own strict predictions.
5. **Relax jitter tolerance in direction checking** (`src/counting.py:270`) — Prevent tracks from terminating instantly if a rotation shrinks the box and shifts the center slightly backwards.
